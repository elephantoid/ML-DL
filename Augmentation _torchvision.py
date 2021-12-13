#import
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import pandas as pd
import os
from skimage import io
from torch.utils.data import (
    Dataset,
    DataLoader,
)
class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


root_dir='C:/Users/all7j/PycharmProjects/datasets/cats_dogs/resized/'
csv_file_path="C:/Users/all7j/PycharmProjects/torch_tutorials/DogsandCats.csv"

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.RandomCrop((224,224)),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0,1.0,1.0]) # (value - mean)/ std
])
dataset = CatsAndDogsDataset(csv_file=csv_file_path, root_dir=root_dir, transform=my_transforms)
img_num=0
for _ in range(10)
    for img, label in dataset:
        save_image(img, 'img'+str(img_num)+".png")
        img_num+=1
        print(img.shape)