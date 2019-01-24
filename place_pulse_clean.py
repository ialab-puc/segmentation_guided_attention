import pandas as pd
from os import listdir

count = 0

def validate_item(row,img_dir,images):
    global count
    left_img_name = f'{row.iloc[0]}.jpg'
    right_img_name = f'{row.iloc[1]}.jpg'
    count += 1
    print(count)
    return left_img_name in images and right_img_name in images

if __name__ == '__main__':
    PLACE_PULSE_PATH ='votes.csv'
    IMAGES_PATH= 'placepulse/'
    data = pd.read_csv(PLACE_PULSE_PATH)
    imgs = set(listdir(IMAGES_PATH))
    print(len(data))
    data = data[data.apply(lambda x: validate_item(x,IMAGES_PATH,imgs), axis=1)]
    print(len(data))
    data.to_csv('votes_clean.csv', header=True, index=False)