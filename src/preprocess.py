import pandas as pd
import os
import numpy as np

from tqdm import tqdm

root_path = '../../input/german_traffic_sign/GTSRB/Final_Training/Images/'

# get all the image folder paths
all_paths = os.listdir(root_path)
all_paths.sort()
# print(all_paths)

# create a new dataframe 
data = pd.DataFrame()
labels = []
counter = 0
# store all images in the dataframe
for i, path in tqdm(enumerate(all_paths), total=len(all_paths)):
    all_images = os.listdir(root_path+all_paths[i])
    for image in all_images:
        if image.split('.')[1] == 'ppm':
            image_name = image.split('.')[0]
            data.loc[counter, 'image_path'] = f"{root_path}{all_paths[i]}/{image_name}"
            data.loc[counter, 'label'] = i
            counter += 1

# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# save the new dataframe
data.to_csv('../../input/german_traffic_sign/GTSRB/data.csv', index=False)

print(data.head(5))