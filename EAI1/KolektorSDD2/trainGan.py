import torch
import collections
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import random
from  Gan import visualize as visualize
from Gan import loader
import Gan.model as model
import argparse
seed = 1
random.seed(seed)
torch.manual_seed(seed)
def main(args):
    if args.device is not None:
       device = torch.device("cuda:"+args.device if (torch.cuda.is_available()) else "cpu")
    else:
       device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    dataloader = loader.loadKolektorSDD2(args)
    D,G,C = model.get_model(args, device)
    if args.trained:
        state_dict = torch.load(args.resume)
        G.load_state_dict(state_dict['generator'])
        D.load_state_dict(state_dict['discriminator'])
    train_criterion = nn.BCELoss()
    fixed_noise = torch.randn(16, args.ls, 1, 1, device=device)
    optimizerD = optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.b, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.b, 0.999))
    img_list = []
    G_loss = []
    D_loss = []
    iters = 0
    for epoch in range(args.e):
        for i, data in enumerate(dataloader, 0):
            realImages = data[0].to(device)
            batchSize = realImages.size(0)
            realImages= realImages.unfold(2, 128, 128).unfold(3,128,128).reshape(8 * batchSize, 3, 128, 128)
            batchSize = 8* batchSize
            D.zero_grad()
            real_label = torch.FloatTensor(batchSize).uniform_(0.98, 1.0).to(device)
            output = D(realImages).view(-1)
            errD_real = train_criterion(output, real_label)
            D_x = output.mean().item()

            noise = torch.randn(batchSize, args.ls, 1, 1, device=device)
            fake_data = G(noise)
            fake_label = torch.full((batchSize,), 0, device=device).type(torch.float)
            output = D(fake_data.detach()).view(-1)
            errD_fake = train_criterion(output, fake_label)

            D_G_z1 = output.mean().item()
            errD = errD_real+errD_fake
            errD.backward()
            optimizerD.step()
            for q in range(10):
                G.zero_grad()
                real_label = torch.full((batchSize,), 1, device=device).type(torch.float)
                fake_data = G(noise)
                output = D(fake_data).view(-1)
                errG = train_criterion(output, real_label)
                pixelLoss= torch.sum(abs(fake_data-realImages))/(128*128*batchSize)
                loss= errG+pixelLoss
                loss.backward()
                D_G_z2 = output.mean().item()
                optimizerG.step()
            if i % 4 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_P: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.e, i, len(dataloader),
                         errD.item(), errG.item(),pixelLoss, D_x, D_G_z1, D_G_z2))
            G_loss.append(errG.item())
            D_loss.append(errD.item())
            if (iters % 100 == 0) or ((epoch == args.e - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake_data = G(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_data, padding=2, normalize=True))
            iters += 1
        if epoch % args.s == 0:
            torch.save({
                'generator': G.state_dict(),
                'discriminator': D.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'params': args
            }, './models/model_final__{}.pth'.format(epoch))
    torch.save({
        'generator': G.state_dict(),
        'discriminator': D.state_dict(),
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict(),
        'params': args
    }, './models/model_final.pth')


    visualize.plotLosses(G_loss,D_loss)
    visualize.imagesRally(img_list)
if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-p', '--path', default="./testimages", type=str, help='path to the dataSet')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default= "0", type=str, help='indices of GPUs to enable (default: all)')
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target')
    options = [
        CustomArgs(['--trained', '--takes bool if trained already'], default=False, type=bool, target=('trained',)),
        CustomArgs(['--lr_g', '--learning_rate_g'],default=0.02, type=float, target=('lr_g',)),
        CustomArgs(['--lr_d', '--learning_rate_d'],default=0.000002, type=float, target=('lr_d',)),
        CustomArgs(['--bs', '--batchsize'], default=8, type=int, target=('data_loader',)),
        CustomArgs(['--iml', '--imagesizelength'],default=512, type=int, target=('data_loader',)),
        CustomArgs(['--imb', '--imagesizebreadth'], default=256, type=int, target=('data_loader',)),
        CustomArgs(['--inc', '--inputchannel'],default=3,type=int, target=('discrimnator',)),
        CustomArgs(['--ouc', '--outputchannel'],default=3,type=int, target=('generator',)),
        CustomArgs(['--up', '--upscalar'],default=64, type=int, target=('discriminator')),
        CustomArgs(['--dw', '--downscalar'],default=64, type=int, target=('generator',)),
        CustomArgs(['--e', '--epochs'], default=10,type=int, target=('models',)),
        CustomArgs(['--key', '--comet_key'],default="",type=str, target=('comet', 'api')),
        CustomArgs(['--offline', '--comet_offline'],default="",type=str, target=('comet', 'offline')),
        CustomArgs(['--b', '--beta'], default=0.5,type=float, target=('optimizer',)),
        CustomArgs(['--s', '--saveepoch'],default=1, type=int, target=('models',)),
        CustomArgs(['--ls', '--latentsize'], default=100, type=int, target=('models',)),

    ]
    for opt in options:
        args.add_argument(*opt.flags, default=opt.default, type=opt.type)
    args = args.parse_args()

    main(args)




