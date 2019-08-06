import pandas as pd
from os import listdir
import os
import cv2

count = 0
SIZE_THRESHOLD = 12.0*1024

def validate_item(row,img_dir,images,total):
    global count
    left_img_name = f'{row.iloc[0]}.jpg'
    right_img_name = f'{row.iloc[1]}.jpg'
    count += 1
    print(f'{count}/{total}\r', end="")
    return left_img_name in images and right_img_name in images

#we drop votes related to images smaller than a threshold since they are noisy.
def filter_by_size(img_dir,row):
    left_img_name = f'{row.iloc[0]}.jpg'
    right_img_name = f'{row.iloc[1]}.jpg'
    return not(os.path.getsize(img_dir+left_img_name) <= SIZE_THRESHOLD or os.path.getsize(img_dir+right_img_name) <= SIZE_THRESHOLD)
    
if __name__ == '__main__':
    PLACE_PULSE_PATH ='votes.csv'
    IMAGES_PATH= 'placepulse/'
    CROPPED_PATH = 'pp_cropped/'
    data = pd.read_csv(PLACE_PULSE_PATH)
    imgs = set(listdir(IMAGES_PATH))
    total = len(data)
    print(total)
    data = data[data.apply(lambda x: validate_item(x,IMAGES_PATH,imgs,total) and filter_by_size(IMAGES_PATH,x), axis=1)]
    print()
    print(len(data))
    data.to_csv('votes_clean.csv', header=True, index=False)