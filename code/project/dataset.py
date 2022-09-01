import os
import csv
import cv2
import dlib
import torch
import random
import skimage
import numpy as np
import pandas as pd
import albumentations as A
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class FaceDataset(Dataset):
    def __init__(self, option, image_label, transforms):
        self.option = option
        self.df = image_label
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        assert index <= len(self), 'index range error'

        image_dir = os.path.join(self.option.data.base_path, self.df.iloc[index, ]['path'])
        image_id = self.df.iloc[index, ]['real'].astype(np.int64)

        image = cv2.imread(image_dir, cv2.COLOR_BGR2RGB)
        image = np.array(image)
        target = torch.as_tensor(image_id, dtype=torch.long)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        image = image/255.0

        return image, target

def get_train_valid_dataloader(options):

    train_df = pd.read_csv(options.data.train)
    train_df['path'] = train_df['path'].map(lambda x : x[2:])

    train, valid = train_test_split(train_df, test_size=options.data.test_proportions)

    w = options.input_size.height
    h = options.input_size.width

    transforms_train = A.Compose([
        A.Resize(w, h),
        ToTensorV2(),
    ])

    transforms_valid = A.Compose([
        A.Resize(w, h),
        ToTensorV2(),
    ])

    train_dataset = FaceDataset(options, image_label=train, transforms=transforms_train)
    valid_dataset = FaceDataset(options, image_label=valid, transforms=transforms_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=options.data.random_split, num_workers=options.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=options.batch_size, shuffle=options.data.random_split, num_workers=options.num_workers)

    return train_dataloader, valid_dataloader, train_dataset, valid_dataset
    