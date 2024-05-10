import argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
from Gan import loader
import Gan
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='E:/EAI1/KolektorSDD2/models/model_good.pth',
                    help='Checkpoint to load path from')
args = parser.parse_args()
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# load the model and get the weights for the generator
state_dict = torch.load(args.load_path, map_location=device)
params = state_dict['params']
params.device = device
params.path = 'E:/EAI1/KolektorSDD2/testimages/classification'
params.bs = 16
D, netG, Classication = Gan.get_model(params, device)
netG.load_state_dict(state_dict['generator'])

# load the images from the dataset using the dataloader without image shuffling
dataloader = loader.loadKolektorSDD2(params, device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(Classication.parameters(), lr=0.02, betas=(0.5, 0.99))


num_epochs = 100
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0
    totalimages = 0
    count = 0
    for images, labels in dataloader:
        count += 1
        size = len(labels)
        labels = labels.type(torch.float32).to(device)

        # now create the noise vector number will be equal to 8*batch size
        noise = torch.randn(8 * size, params.ls, 1, 1, device=device, requires_grad=True)
        optimizernoise = optim.Adam([noise], lr=0.1, betas=(0.9, 0.99))
        images = images.unfold(2, 128, 128).unfold(3, 128, 128).reshape(8 * size, 3, 128, 128).to(device)

        for i in range(15):
            optimizernoise.zero_grad()
            generated_img = netG.eval()(noise)
            error = torch.sum(abs(images - generated_img)) / (128 * 128 * 8 * size)
            error.backward()
            optimizernoise.step()
        generated_img = netG.eval()(noise).detach()
        images = (images - generated_img).reshape(size, 3, 512, 256)

        optimizer.zero_grad()
        logits = Classication(images.to(device))
        labels = labels.unsqueeze(1)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        correct = torch.sum(((logits >= 0.0).type(torch.float32).view(-1) == labels.view(-1)).type(torch.float32))
        running_corrects += correct
        totalimages += len(labels)
        running_loss += loss.item()
        if count % 100 == 0:
            print(count)
    avg_loss = running_loss / len(dataloader)
    avg_acc = running_corrects / totalimages
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_acc:.4f}')

    # test loader
    params.path = 'E:/EAI1/KolektorSDD2/testimages/test'
    test_loader = loader.loadKolektorSDD2(params, device)
    # testing
    total_correct_test = 0
    total_test = 0

    count = 0
    for inputs, labels in test_loader:
        size = len(labels)
        labels = labels.type(torch.float32).to(device)
        noise = torch.randn(8 * size, params.ls, 1, 1, device=device, requires_grad=True)
        inputs = inputs.unfold(2, 128, 128).unfold(3, 128, 128).reshape(8 * size, 3, 128, 128).to(device)
        for i in range(15):
            optimizernoise.zero_grad()
            generated_img = netG.eval()(noise)
            error = torch.sum(abs(inputs - generated_img)) / (128 * 128 * 8 * size)
            error.backward()
            optimizernoise.step()

            # print('noise', noise[0][0])

        generated_img = netG.eval()(noise).detach()

        inputs = (inputs - generated_img).reshape(size, 3, 512, 256)
        with torch.no_grad():
            logits = Classication(inputs.to(device))
            labels = labels.unsqueeze(1)
            loss = loss_function(logits, labels)
        correct = torch.sum(((logits >= 0.0).type(torch.float32).view(-1) == labels.view(-1)).type(torch.float32))
        running_corrects += correct
        totalimages += len(labels)
        running_loss += loss.item()
        print(f'Iteration {count + 1}/{len(dataloader)}, Test Loss: {loss:.4f}, Test Accuracy: {correct / size:.4f}')
        count += 1

        avg_test_loss = running_loss / len(dataloader)
        avg_test_acc = running_corrects / totalimages

        if epoch % 10 == 0:
            print("Saving model")
            torch.save({
                'classication': Classication.state_dict(),
                'optimizer': optimizer.state_dict(),
                'params': args
            }, 'E:/EAI1/KolektorSDD2/checkpoint/model_final__{}.pth'.format(epoch))

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Average Test Loss: {avg_test_loss:.4f}, Average Test Accuracy: {avg_test_acc:.4f}')

torch.save({
    'classication': Classication.state_dict(),
    'optimizer': optimizer.state_dict(),
    'params': args
}, 'E:/EAI1/KolektorSDD2/outputs/model_final.pth')

print('Finished Training')

