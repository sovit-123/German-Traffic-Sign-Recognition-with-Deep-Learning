import glob as glob
import albumentations
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os

from model import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# load the model checkpoint
checkpoint = torch.load('../outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])

# read all image paths
root_dir = '../../input/german_traffic_sign/GTSRB/Final_Test/Images/'

# read the test dataframe
test_df = pd.read_csv(
    '../../input/german_traffic_sign/GTSRB/Final_Test/GTSRB_Final_Test_GT/GT-final_test.csv',
    delimiter=';', nrows=10
    )
# change index to filename for easier access to labels
gt_df = test_df.set_index('Filename', drop=True)

# read sign label dataframes
sign_df = pd.read_csv(
        '../../input/german_traffic_sign/GTSRB/Final_Training/signnames.csv'
        )

aug = albumentations.Compose([
                # 48x48 resizing is required for this network model
                albumentations.Resize(48, 48, always_apply=True),
            ])


for i in range(len(test_df)):
    image_path = root_dir+test_df.loc[i, 'Filename']
    image = plt.imread(image_path)
    orig = image.copy()

    model.eval()
    with torch.no_grad():
        image = image / 255.
        image = aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float).to(device)
        image = image.unsqueeze(0)
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
         
    # get the prediction label
    label = sign_df.loc[int(preds), 'SignName']
    # get the ground truth label
    filename = image_path.split('/')[-1]
    gt_id = gt_df.loc[filename].ClassId
    gt_label = sign_df.loc[int(gt_id), 'SignName']

    # image = image.detach().cpu().numpy()
    # image = image.squeeze(0)
    # image = np.transpose(image, (1, 2, 0))
    # plt.imshow(image)
    # plt.title('Image that the model sees')
    # plt.show()
    
    plt.imshow(orig)
    plt.title(f"Prediction - {str(label)}\nGround Truth - {str(gt_label)}")
    plt.axis('off')
    plt.show()
    plt.close()