import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import datasets, transforms as tv

# Dataset class to preprocess your data and labels
# You can do all types of transformation on the images in this class


class bird_dataset(Dataset):
    # You can read the train_list.txt and test_list.txt files here.
    def __init__(self,root,file_path):
        self.normalize = tv.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = tv.Compose([tv.Resize(256), tv.CenterCrop(224), tv.ToTensor()])
        self.img = []
        self.labels = []
        f = open(root + file_path, "r")
        lines = f.readlines()
        for line in lines:
            image_addr = root + "images/" + line.split(' ')[0]
            image = Image.open(image_addr)
            image = image.convert('RGB')
            image_convert = self.transform(image)
            # the pixel within the image is normalized to [0,1]
            self.img.append(image_convert)
            self.labels.append(line.split(' ')[1])
        f.close()
        
        # get mean and std of all images
        self.mean = torch.mean(torch.stack(self.img),dim=0)
        self.std = torch.std(torch.stack(self.img),dim=0)
            

    def __len__(self):
        return len(self.img)


    def __getitem__(self, item):
        image = self.img[item]
        # normalization by z score
        image = (image - self.mean) / self.std
        # normalization using mean and std (0.485, 0.456, 0.406), (0.229, 0.224, 0.225).
        image = self.normalize(image)
        currClass = int(self.labels[item])
        return image, currClass

