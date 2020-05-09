#!/usr/bin/env python
# coding: utf-8

# In[2]:


# References:
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution
# https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/


# In[3]:


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


# In[4]:


import tensorflow as tf

# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(2020)

# Make the augmentation sequence deterministic
aug.seed(2020)


# In[5]:


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


# In[6]:





# In[7]:


# Get the counts for each class
cases_count = y['label'].value_counts()
print(cases_count)

# Plot the results 
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Bacterial(1)','Viral(2)','Covid(3)'])
plt.show()


# In[ ]:





# In[8]:


def get_samples(num_label):
    return (X_train_filename[y_train['label']==num_label]['filename'].iloc[:5]).tolist()


normal_samples = get_samples(0)
pneumonia_b_samples = get_samples(1)
pneumonia_v_samples = get_samples(2)
covid_samples = get_samples(3)

IMG_SIZE = 256

# Concat the data in a single list and del the above two list
samples =  normal_samples + pneumonia_b_samples + pneumonia_v_samples + covid_samples 
del pneumonia_b_samples,pneumonia_v_samples, covid_samples, normal_samples

# Plot the data 
f, ax = plt.subplots(4,5, figsize=(30, 30))
for i in range(20):
    img = imread(samples[i])
    #img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Normal")
    elif 5<= i < 10:
        ax[i//5, i%5].set_title("Bacterial")
    elif 10 <= i < 15:
        ax[i//5, i%5].set_title("Viral")
    else:
        ax[i//5, i%5].set_title("Covid")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
print("without preprocessing")
plt.savefig("before.png")
plt.show()


# In[9]:


# an interesting method of color preprocessing

def crop_image(img,tol=7):
    # img is image data
    # tol  is tolerance
        
    mask = img>tol
    if not(img[np.ix_(mask.any(1),mask.any(0))].all() == img.all()):
        print("cropped!")
    return img[np.ix_(mask.any(1),mask.any(0))]

f, ax = plt.subplots(4,5, figsize=(30, 30))
for i in range(20):
    img = imread(samples[i])
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = crop_image_from_gray(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , IMG_SIZE/10) ,-4 ,128) # the trick is to add this line
    img = crop_image(img, tol = 150)

    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Normal"+ str(img.shape))
    elif 5<= i < 10:
        ax[i//5, i%5].set_title("Bacterial"+ str(img.shape))
    elif 10 <= i < 15:
        ax[i//5, i%5].set_title("Viral"+ str(img.shape))
    else:
        ax[i//5, i%5].set_title("Covid"+ str(img.shape))
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
print("with preprocessing")
plt.savefig("after.png")
plt.show()


# In[ ]:





# In[126]:


def load_ben_color(filename, sigmaX=10):
    
    img = cv2.imread(filename)
    if(img.shape[0] <256 or img.shape[1] < 256):
        print("bad image")
    img = crop_image(img, tol = 15)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , IMG_SIZE/10) ,-4 ,128)
    return img


# # Preprocessing 
# After EDA, we will now start actually preprocessing our data

# In[127]:


from sklearn.utils import class_weight
y_train_num = np.array(y_train['label'])
y_val_num = np.array(y_val['label'])
class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train_num),y_train_num)


# In[128]:


class_weights


# In[129]:


# for keras to run we will turn our into categorical
y_train = to_categorical(y_train_num, num_classes=None)
y_val = to_categorical(y_val_num, num_classes=None) 


# In[130]:


y_val


# In[131]:


# Augmentation sequence 
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=20), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness


# In[132]:


def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size
    
    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,4), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Initialize a counter
    i =0
    while True:
        np.random.shuffle(indices)
        # Get the next batch 
        count = 0
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['filename']
            label = data.iloc[idx]['label']
            
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=4)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            
            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.
            
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            
            # generating more samples of the undersampled class
            if label==4 and count < batch_size-2:
                aug_img1 = seq.augment_image(img) #using the augmentation seq we used earlier
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32)/255.
                aug_img2 = aug_img2.astype(np.float32)/255.

                batch_data[count+1] = aug_img1
                batch_labels[count+1] = encoded_label
                batch_data[count+2] = aug_img2
                batch_labels[count+2] = encoded_label
                count +=2
            
            else:
                count+=1
            
            if count==batch_size-1:
                break
            
        i+=1
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0


# # Building the CNN

# In[133]:


def build_model():
    input_img = Input(shape=(IMG_SIZE,IMG_SIZE,3), name='ImageInput')
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='pool1')(x)
    
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3,3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='pool2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(4, activation='softmax', name='fc3')(x)
    
    model = Model(inputs=input_img, outputs=x)
    return model


# In[134]:


model =  build_model()
model.summary()


# In[135]:


# Open the VGG16 weight file
f = h5py.File('vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# Select the layers for which you want to set weight.

w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']
model.layers[1].set_weights = [w,b]

w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
model.layers[2].set_weights = [w,b]

w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
model.layers[4].set_weights = [w,b]

w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
model.layers[5].set_weights = [w,b]

f.close()
model.summary()    


# In[136]:


# opt = RMSprop(lr=0.0001, decay=1e-6)
opt = Adam(lr=0.0001, decay=1e-5)
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)


# In[137]:



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
    
 


# In[138]:


X_train = np.array(X_train)
X_val = np.array(X_val)
y_train = np.array(y_train)
y_val = np.array(y_val)


# In[139]:


X_train


# In[140]:


print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)


# In[141]:


X_train_filename['label'] =y_train_num


# In[43]:


batch_size = 5
nb_epochs = 20

# Get a train data generator
X_train_gen = data_gen(data=X_train_filename, batch_size=batch_size)

# Define the number of training steps
nb_train_steps = X_train.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(X_val)))


# In[44]:


len(y_train)


# In[29]:


history = model.fit_generator(X_train_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                               validation_data=(X_val, y_val),callbacks=[es, chkpt],
                               class_weight=class_weights)


# # Trying out a different type of data gen

# In[142]:


# this is the augmentation configuration we will use for training

'''
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
'''

train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

test_datagen = ImageDataGenerator()


# In[143]:


callbacks = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
# autosave best Model
best_model_file = "./test.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

history = model.fit_generator(train_datagen.flow(X_train, y_train, batch_size=8), nb_epoch=10,                   
              validation_data=val_datagen.flow(X_val,y_val,batch_size=8,shuffle=False),callbacks = [callbacks,best_model])


# In[144]:


len(X_val)


# In[145]:


X_test_filename


# In[146]:


X_test_filename = pd.read_csv('test.csv', index_col = 'id', usecols = ['id','filename'])

X_test = []
for index, rows in X_test_filename.iterrows():
    # different preprocessing 
    #img = load_ben_color(X_train_filename['filename'][index])
    img = cv2.imread(X_test_filename['filename'][index])
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    if img.shape[2] ==1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.
    X_test.append(img)
    
X_test = np.array(X_test)


# In[147]:


preds = model.predict(X_test)


# In[148]:


label_num = np.argmax(preds, axis=-1)  
preds


# In[149]:


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


# In[150]:


submission.head()
submission.to_csv("submission.csv", index = None)


# In[151]:


submission


# In[104]:


y_val


# In[ ]:




