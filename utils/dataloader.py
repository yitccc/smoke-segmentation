import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root):
        imgs_full_root = os.path.join(image_root, 'images')
        labs_full_root = os.path.join(gt_root, 'labels')
        image_names = os.listdir(imgs_full_root)
        label_names = os.listdir(labs_full_root)
        self.images = [os.path.join(imgs_full_root, im_name) for im_name in image_names]
        self.gts = [os.path.join(labs_full_root, lab_name) for lab_name in label_names]
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip()
            ]
        )

    def __getitem__(self, index):
        
        image_idx = self.images[index]
        image = cv2.imread(image_idx)
        image = cv2.resize(image, (256, 192))
        # image = image.transpose(2,1,0)
        gt_idx = self.gts[index]
        gt = cv2.imread(gt_idx, 0)
        gt = cv2.resize(gt, (256, 192))
        gt = gt / 255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        # image = image / 255.0
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=4, pin_memory=True):

    dataset = SkinDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root):
        imgs_full_root = os.path.join(image_root, 'images')
        labs_full_root = os.path.join(gt_root, 'labels')
        image_names = os.listdir(imgs_full_root)
        label_names = os.listdir(labs_full_root)
        self.images = [os.path.join(imgs_full_root, im_name) for im_name in image_names]
        self.gts = [os.path.join(labs_full_root, lab_name) for lab_name in label_names]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image_idx = self.images[self.index]
        image = cv2.imread(image_idx)
        image = cv2.resize(image, (256, 192), interpolation=cv2.INTER_NEAREST)
        # image = image / 255.0
        # image = image.transpose(2, 1, 0)
        image = self.transform(image).unsqueeze(0)
        # image = image / 255.0
        gt_idx = self.gts[self.index]
        gt = cv2.imread(gt_idx, 0)
        gt = cv2.resize(gt, (256, 192), interpolation=cv2.INTER_NEAREST)
        gt = gt/255.0
        self.index += 1

        return image, gt


if __name__ == '__main__':
    path = 'data/'
    tt = SkinDataset(path+'data_train.npy', path+'mask_train.npy')

    for i in range(50):
        img, gt = tt.__getitem__(i)

        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)
        plt.savefig('vis/'+str(i)+".jpg")
 
        plt.imshow(gt[0])
        plt.savefig('vis/'+str(i)+'_gt.jpg')
