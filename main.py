# This is a sample Python script.
from random import shuffle


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
import cv2
import time
import random
import shutil
from skimage.feature import hog
import os
import numpy as np
import sys
from skimage.measure import shannon_entropy
from  keras.src.models import Sequential
from keras.src.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
sys.setrecursionlimit(10000)
from sklearn.preprocessing import StandardScaler
input_path='D:\\dataset\\asl_alphabet_train\\asl_alphabet_train'
output_path='D:\\dataset\\asl_alphabet_train\\augmented_images'

data=[]
labels=[]
def data_preprocess(input_path,output_path):
    if not os.path.exists(output_path):
        print('Directory does not exist')
        os.makedirs(output_path)
    for class_name in os.listdir(input_path):
        class_path=os.path.join(input_path,class_name)
        if os.path.isdir(class_path):
            output_class_path=os.path.join(output_path,class_name)
            if os.path.isdir(output_class_path):
                shutil.rmtree(output_class_path)
            if not os.path.isdir(output_class_path):
                os.makedirs(output_class_path)
        for img_name in os.listdir(class_path):
            image_path=os.path.join(class_path,img_name)
            output_img_path=os.path.join(output_class_path,img_name)
            img=cv2.imread(image_path)/255.0
            if img is not None:
            # cv2.imwrite(output_img_path,img)
              for i in range(3):
                  aug_img=augment_data(img)
                  aug_img_name=f"{img_name.split('.')[0]}_aug{i+1}.jpg"
                  # aug_img_path=os.path.join(output_class_path,aug_img_name)
                  # cv2.imwrite(aug_img_path,(aug_img*255).astype(np.uint8))
                  data.append(aug_img)
                  labels.append(aug_img_name)
                  # print(np.array(data))
    return np.array(data), np.array(labels)
def augment_data(img):
        transformations=[
            lambda x: cv2.rotate(x,cv2.ROTATE_90_CLOCKWISE),
            lambda x: cv2.rotate(x,cv2.ROTATE_90_COUNTERCLOCKWISE),
            lambda x: cv2.flip(x,1),
            lambda x: cv2.flip(x,0),
            lambda x: cv2.convertScaleAbs(x,alpha=1.2,beta=10)
        ]
        return random.choice(transformations)(img)
data_preprocess(input_path,output_path)
label_encoder=LabelEncoder()
encoded_labels=label_encoder.fit_transform(labels)
x_train,x_temp,y_train,y_temp=train_test_split(data,encoded_labels,test_size=0.2,random_state=42,stratify=encoded_labels)
x_val,x_test,y_val,y_test=train_test_split(x_temp,y_temp,test_size=0.1,random_state=42,stratify=y_temp)
num_classes=len(np.unique(y_train))
print(num_classes)
model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(200,200,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(num_classes,activation='softmax')
])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10,batch_size=32)
test_loss,test_acc=model.evaluate(x_test,y_test)
