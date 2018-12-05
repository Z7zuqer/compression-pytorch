import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

valDir="/share4/public/classification_data/imagenet1k/val/"
trainDir="/share4/public/classification_data/imagenet1k/train/"

def PILLoad(imgPath,resizeH,resizeW,transform,phase):
    path=""
    if phase=="train":
        path=trainDir+imgPath
    else:   
        path=valDir+imgPath
    with Image.open(path) as img:
        image = img.convert('RGB')
    image = transform(image)
    return image

class MyValDataset(Dataset):
    def __init__(self,shuffle):
        fh=open("/share4/public/classification_data/imagenet1k/meta/val.txt","r")
        imgs=[]
        self.transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,])
        self.shuffle=shuffle
        for line in fh:
            line=line.strip('\n')
            line=line.rstrip()
            words=line.split()
            labelList=int(words[1])
            imageList=words[0]
            imgs.append((imageList,labelList))
        self.imgs=imgs
    def __getitem__(self,item):
        image,label=self.imgs[item]
        img=PILLoad(image,224,224,self.transform,"test")
        return img,label
    def __len__(self):
        return len(self.imgs)

class MyTrainDataset(Dataset):
    def __init__(self,shuffle):
        fh=open("/share4/public/classification_data/imagenet1k/meta/train.txt","r")
        imgs=[]
        self.transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,])
        self.shuffle=shuffle
        for line in fh:
            line=line.strip('\n')
            line=line.rstrip()
            words=line.split()
            labelList=int(words[1])
            imageList=words[0]
            imgs.append((imageList,labelList))
        self.imgs=imgs
    def __getitem__(self,item):
        image,label=self.imgs[item]
        img=PILLoad(image,224,224,self.transform,"train")
        return img,label
    def __len__(self):
        return len(self.imgs)
