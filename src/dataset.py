import torch
import pandas as pd
import cv2
import albumentations
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class GTSRDataset(Dataset):
    def __init__(self, images, labels, tfms=None):
        self.images = images
        self.labels = labels

        # apply augmentations
        if tfms == 0: # if validating
            self.aug = albumentations.Compose([
                # 48x48 resizing is required
                albumentations.Resize(48, 48, always_apply=True),

            ])
        else: # if training
            self.aug = albumentations.Compose([
                # 48x48 resizing is required
                albumentations.Resize(48, 48, always_apply=True),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = plt.imread(self.images[index]+'.ppm')
        image = image / 255.
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1))
        label = self.labels[index]

        return {
            'image': torch.tensor(image, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

df = pd.read_csv('../../input/german_traffic_sign/GTSRB/data.csv', 
                 nrows=5000)

X = df.image_path.values
y = df.label.values

(xtrain, xtest, ytrain, ytest) = train_test_split(X, y, 
                                test_size=0.10, random_state=42)
print(f"Training instances: {len(xtrain)}")
print(f"Validation instances: {len(xtest)}")

train_data = GTSRDataset(xtrain, ytrain, tfms=1)
val_data = GTSRDataset(xtest, ytest, tfms=0)


batch_size = 256
train_data_loader = DataLoader(
    train_data, 
    batch_size=batch_size,
    shuffle=True,
    # num_workers=1,
)
val_data_loader = DataLoader(
    val_data, 
    batch_size=batch_size,
    shuffle=False,
    # num_workers=1,
)

# visualization
visualize = False
if visualize:
    for i in range(1):
        sign_df = pd.read_csv(
            '../../input/german_traffic_sign/GTSRB/Final_Training/signnames.csv'
            )
        sample = train_data[i]
        image = sample['image']
        label = sample['label']
        image = np.array(np.transpose(image, (1, 2, 0)))
        plt.imshow(image)
        plt.title(str(sign_df.loc[int(label), 'SignName']))
        plt.show()