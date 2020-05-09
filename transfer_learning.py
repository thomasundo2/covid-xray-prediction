#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelEncoder



# In[3]:


import tensorflow as tf

# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(2020)

# Make the augmentation sequence deterministic
aug.seed(2020)

IMG_SIZE = 250


# In[4]:


df = pd.read_csv('train.csv', index_col = 'id', usecols = ['id','filename'])
y = pd.read_csv('train.csv', index_col = 'id', usecols = ['id','label'])
# convert y into numerical labels 
def label_to_numeric(x):
    if x=='normal':
        return 0
    if x=='bacterial':
        return 1
    if x=='viral':
        return 2
    if x=='covid':
        return 3
y['label'] = y['label'].apply(label_to_numeric)


# split into a training and test (validation) set
X_train_filename, X_val_filename, y_train, y_val = train_test_split(df, y, test_size=0.1, random_state = 2015)
X_train_filename.reset_index(drop = True, inplace = True)
X_val_filename.reset_index(drop = True, inplace = True)
y_train.reset_index(drop = True, inplace = True)
y_val.reset_index(drop = True, inplace = True)

from keras.utils import to_categorical


# In[5]:


# reading in data and preprocessing 
X_train = []
for index, rows in X_train_filename.iterrows():
    # different preprocessing 
    #img = load_ben_color(X_train_filename['filename'][index])
    img = cv2.imread(X_train_filename['filename'][index])
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    X_train.append(img)
    
X_val = []
for index, rows in X_val_filename.iterrows():
    # different preprocessing 
    #img = load_ben_color(X_train_filename['filename'][index])
    img = cv2.imread(X_val_filename['filename'][index])
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    X_val.append(img)


# In[ ]:





# In[6]:


X_train = np.array(X_train)
X_val = np.array(X_val)
onehot_train = to_categorical(LabelEncoder().fit_transform(y_train))
onehot_valid = to_categorical(LabelEncoder().fit_transform(y_val))


# In[7]:


y_train


# In[8]:


for i in range(len(onehot_train)):
    print(onehot_train[i])
    print(np.array(y_train)[i])


# In[9]:


# Source: https://github.com/Tahsin-Mayeesha/udacity-mlnd-deeplearning-capstone

vgg16_base = VGG16(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(IMG_SIZE, IMG_SIZE,3))

print('Adding new layers...')
output = vgg16_base.get_layer(index = -1).output  
output = Flatten()(output)
output = Dense(4096,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.5)(output)
output = Dense(4, activation='softmax')(output)


# In[10]:


vgg16_model = Model(vgg16_base.input, output)
for layer in vgg16_model.layers[:19]:
    layer.trainable = False

vgg16_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics =["accuracy"])


# In[11]:


train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()


# In[12]:


callbacks = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
# autosave best Model
best_model_file = "./data_augmented_weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)


# In[13]:


history = vgg16_model.fit_generator(train_datagen.flow(X_train, onehot_train, batch_size=10), nb_epoch=10,                   
              validation_data=val_datagen.flow(X_val,onehot_valid,batch_size=10,shuffle=False),callbacks = [callbacks,best_model])


# In[23]:


# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model_acc.png")
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("model_loss.png")
plt.show()


# In[20]:


vgg16_model.summary()


# In[14]:


X_test_filename = pd.read_csv('test.csv', index_col = 'id', usecols = ['id','filename'])

# process the data for testing
X_test = []
for index, rows in X_test_filename.iterrows():
    # different preprocessing 
    #img = load_ben_color(X_train_filename['filename'][index])
    print(X_test_filename['filename'][index])
    img = cv2.imread(X_test_filename['filename'][index])
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    X_test.append(img)
    
X_test = np.array(X_test)


# In[18]:


preds = vgg16_model.predict(X_test)


# In[16]:


# convert the proba into actual labels for the kaggle 

label_num = np.argmax(preds, axis=-1)  
# prepare for submission
def label_to_numeric(x):
    if x==0:
        return 'normal'
    if x==1:
        return 'bacterial'
    if x==2:
        return 'viral'
    if x== 3:
        return 'covid'
label = np.array(pd.DataFrame(label_num)[0].apply(label_to_numeric))
data_id = np.arange(len(label))
submission = pd.DataFrame(data_id, columns = ["Id"])
submission['label'] = label
submission.to_csv("submission.csv", index = None)


# In[1]:


preds


# In[ ]:


for i, element in enumerate(preds):
    print(np.array(submission['label'])[i])
    print(element)


# In[ ]:




