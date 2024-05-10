import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from Gan import loader
import random
import Gan
from Gan.model import Generator

parser = argparse.ArgumentParser()
parser.add_argument('-load_path', default='/home/farooq/Music/EAI1/KolektorSDD2/models/model_good.pth', help='Checkpoint to load path from')
parser.add_argument('-num_output', default=8, help='Number of generated outputs')
args = parser.parse_args()
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")

print(args.load_path)
state_dict = torch.load(args.load_path,map_location=device)
print("loaaded dict")
params = state_dict['params']
params.device = "cuda:0"
params.path = '/home/farooq/Music/EAI1/KolektorSDD2/testimages'
params.bs =4
D,netG = Gan.get_model(params,device)
netG.load_state_dict(state_dict['generator'])
print(netG)
print(args.num_output)
dataloader = loader.loadKolektorSDD2(params,device)
image = (next(iter(dataloader))[0]).unfold(2, 128, 128).unfold(3,128,128).reshape(8 * params.bs, 3, 128, 128)
noise = torch.randn(8 * params.bs, params.ls, 1, 1, device=device)
with torch.no_grad():
    generated_img = netG(noise).detach().cpu()

plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(torch.cat((image,generated_img),0), padding=50, normalize=True), (1,2,0)))
plt.savefig('test2.png',dpi = 300)
plt.show()
plt.close()
