import numpy as np
import pandas as pd
import os
import cv2
from sklearn.utils import shuffle
import pickle


def train_image_reader(img_path,x1_path,y1_path):
    ''' Reads the content of a folder, and processes it into pkl files'''
    ### Dictionary for emotions
    feel_names = ['angry','disgust','fear','happy','neutral','sad','surprise']
    feel_names_dict = {feel_name:i for i, feel_name in enumerate(feel_names)}
    ### Pixels and label into arrays:
    main_dir = img_path
    folders = ['angry','disgust','fear','happy','neutral','sad','surprise']
    img_dict = {'image':[],'label':[]}

    for folder in folders:
        folder_path = os.path.join(main_dir,folder)
        files_folder = os.listdir(folder_path)
        for file in files_folder:
            file_name = os.path.basename(file)
            img_dict['image'].append(file_name)
            img_dict['label'].append(folder)

    df_train = pd.DataFrame(img_dict)
    x1 = []
    y1 = []

    for i,row in df_train.iterrows():
        print(f'Opening image #{i} ,{row['label']}')
        image_path = os.path.join(img_path, row['label'],row['image'])
        image = cv2.imread(image_path,flags = cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        x1.append(image)
        y1.append(row['label'])

    ### Lists
    x1 = np.array(x1)
    y1 = np.array(y1)

    ### Vectorization of y1 and scalarization of x1
    map_emotions = np.vectorize(feel_names_dict.get)
    y1 = map_emotions(y1)
    x1 = x1/255


    ###Shuffle and dumping into a pickle
    x1, y1 = shuffle(x1, y1, random_state=42)
    pickle.dump(x1,open(x1_path,'wb'))
    pickle.dump(y1,open(y1_path,'wb'))
    return x1, y1, df_train

def test_image_reader(img_path,x1_path):
    ### Pixels and label into arrays:
    main_dir = img_path
    img_dict = {'image':[]}
    files_folder = os.listdir(img_path)

    # for folder in folders:
    #     folder_path = os.path.join(main_dir,folder)
        
    for file in files_folder:
        file_name = os.path.basename(file)
        img_dict['image'].append(file_name)

    df_test = pd.DataFrame(img_dict)
    x1_test = []

    for i,row in df_test.iterrows():
        print(f'Opening image #{i}')
        image_path = os.path.join(img_path, row['image'])
        image = cv2.imread(image_path,flags = cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        x1_test.append(image)


    ### Lists
    x1_test = np.array(x1_test)
    ### Vectorization of y1 and scalarization of x1
    x1_test = x1_test/255
    df_test['pixels'] = x1_test.tolist()

    ###Shuffle and dumping into a pickle
    pickle.dump(x1_test,open(x1_path,'wb'))
    return x1_test, df_test