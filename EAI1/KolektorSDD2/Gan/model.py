import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init(w):
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)


class Generator(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.latentSize = args.ls
        self.downScalar = args.dw
        self.outChannels = args.ouc

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(self.latentSize, self.downScalar*16,kernel_size=4, stride=1, padding=0, bias=False)
        # self.tconv1 = nn.Conv2d(self.outChannels, self.downScalar * 16, kernel_size=(24,400), stride=(24,400), padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(self.downScalar*16)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.downScalar*16, self.downScalar*8, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.downScalar*8)

        self.tconv3 = nn.ConvTranspose2d(self.downScalar * 8, self.downScalar * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.downScalar * 4)

        self.tconv4 = nn.ConvTranspose2d(self.downScalar * 4, self.downScalar * 2, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.downScalar * 2)
        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv5 = nn.ConvTranspose2d(self.downScalar*2, self.downScalar, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.downScalar)



        # # Input Dimension: (ngf*2) x 16 x 16
        # self.tconv6 = nn.ConvTranspose2d(self.downScalar*2, self.downScalar, 4, 2, 1, bias=False)
        # self.bn6 = nn.BatchNorm2d(self.downScalar)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv7 = nn.ConvTranspose2d(self.downScalar, self.outChannels, 4, 2, 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.relu(self.bn5(self.tconv5(x)))
        # x = F.relu(self.bn6(self.tconv6(x)))


        x = torch.tanh(self.tconv7(x))

        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inChannels =args.inc
        self.upScalar =args.up
        #256
        self.conv1 = nn.Conv2d(self.inChannels, self.upScalar, 4, 2, 1, bias=False)

        # #128
        # self.conv2 = nn.Conv2d(self.upScalar, self.upScalar*2, 4, 2, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(self.upScalar*2)
        # self.drop1 = nn.Dropout(0.1)
        #64
        self.conv3 = nn.Conv2d(self.upScalar, self.upScalar*2,4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.upScalar*2)
        self.drop2 = nn.Dropout(0.05)
        #32
        self.conv4 = nn.Conv2d(self.upScalar*2, self.upScalar*4, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.upScalar*4)
        self.drop3 = nn.Dropout(0.1)
        #16
        self.conv5 = nn.Conv2d(self.upScalar * 4, self.upScalar * 8, 4,2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.upScalar * 8)
        self.drop4 = nn.Dropout(0.1)
        #8
        self.conv6 = nn.Conv2d(self.upScalar * 8, self.upScalar * 16, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.upScalar * 16)
        #4
        self.conv7 = nn.Conv2d(self.upScalar*16, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        # x = self.drop1(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = self.drop2(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = self.drop3(x)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)
        x = self.drop4(x)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2, True)



        x = torch.sigmoid(self.conv7(x))

        return x


# Define the classifier

class Classification(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.inChannels =3
        self.upScalar =8
        #512x 256  input
        self.conv1 = nn.Conv2d(self.inChannels, self.upScalar, 4, 2, 1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # #128
        # self.conv2 = nn.Conv2d(self.upScalar, self.upScalar*2, 4, 2, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(self.upScalar*2)
        # self.drop1 = nn.Dropout(0.1)
        #128x64  after  conv1
        self.conv3 = nn.Conv2d(self.upScalar, self.upScalar*2,4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.upScalar*2)
        self.drop2 = nn.Dropout(0.05)
        #64x32  after  conv3
        self.conv4 = nn.Conv2d(self.upScalar*2, self.upScalar*4, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.upScalar*4)
        self.drop3 = nn.Dropout(0.1)
        #32x16 after conv4
        self.conv5 = nn.Conv2d(self.upScalar * 4, self.upScalar * 8, 4,2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(self.upScalar * 8)
        self.drop4 = nn.Dropout(0.1)
        #16x8 after conv5
        self.conv6 = nn.Conv2d(self.upScalar * 8, self.upScalar * 16, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(self.upScalar * 16)
        #8x4 after conv6

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.upScalar*16 * 8 * 4, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear( 10, 1)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = self.maxpool1(x)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        # x = self.drop1(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = self.drop2(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = self.drop3(x)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2, True)
        x = self.drop4(x)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2, True)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x




def get_model(args,device):
    G = Generator(args).to(device)
    G.apply(weights_init)
    D = Discriminator(args).to(device)
    D.apply(weights_init)
    C= Classification(args).to(device)

    return D,G,C


