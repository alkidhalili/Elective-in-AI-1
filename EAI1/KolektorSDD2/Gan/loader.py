import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset


def loadKolektorSDD2(args):
    transform_train = transforms.Compose([
        transforms.Resize((args.iml, args.imb)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor()
    ])
    dataset_classification = dset.ImageFolder(root=args.path, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(dataset_classification, batch_size=args.bs, shuffle=True)
    return dataloader

