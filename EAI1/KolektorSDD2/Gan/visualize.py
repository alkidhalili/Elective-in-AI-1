import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib
#matplotlib.use('TkAgg')
import numpy as np
import matplotlib.animation as animation

def showImages(batch,device,message):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title(message)
    plt.imshow(np.transpose(vutils.make_grid( batch[0].to(device)[: 64], padding=2, normalize=True,scale_each = True).cpu(), (1, 2, 0)))
    plt.show()
    plt.close()


def plotLosses(G_loss,D_loss):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_loss, label="G")
    plt.plot(D_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def imagesRally(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    anim = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    plt.show()
    anim.save('rally.gif', dpi=80, writer='pillow')