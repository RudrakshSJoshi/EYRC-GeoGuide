#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import tensorflow as tf
import random
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import timm

tf.random.set_seed(42)
np.random.seed(42)


# In[2]:


class_names = ['Combat', 'DestroyedBuildings', 'Fire','Humanitarian Aid and rehabilitation',"Military vehicles and weapons"]
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (256, 256)


# In[3]:


data = "/kaggle/input/new-dataset/dataset/train"
images = []
labels = []
for folder in os.listdir(data):
    label = class_names_label[folder]
    for file in tqdm(os.listdir(os.path.join(data, folder))):
                
                # Get the path name of the image
        img_path = os.path.join(os.path.join(data, folder), file)
                
                # Open and resize the img
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE) 
                
                # Append the image and its corresponding label to the output
        images.append(image)
        labels.append(label)


# In[4]:


t_data = "/kaggle/input/new-dataset/dataset/test"
t_images = []
for file in tqdm(os.listdir(t_data)):
        img_path = os.path.join(t_data, file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE) 
        t_images.append(image)
t_images = np.array(t_images)


# In[5]:


import matplotlib.pyplot as plt
def display_examples(class_names, images, labels):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])  # Ensure that images[i] is a valid image data
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(class_names[labels[i]])
    plt.show()
display_examples(class_names, images, labels)


# In[6]:


images = np.array(images ,dtype = 'float32')/255
labels = np.array(labels, dtype = 'int32')


# In[7]:

train_images ,test_images,train_labels,test_labels= train_test_split(images,labels,test_size=0.2,random_state =42)
n_train = train_images.shape[0]
n_test = test_images.shape[0]
print ("Number of training examples: {}".format(n_train))
print ("Number of test examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))


# In[8]:


datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# In[9]:
unique_labels, label_counts = np.unique(train_labels, return_counts=True)
print("Label vs Counts for training data")
# Print the unique labels and their counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# In[10]:


label_counts = {0: 60, 1: 68, 2: 69, 3: 57, 4: 60}

max_count = 70# Perform data augmentation for each class to match the maximum count
augmented_images = []
augmented_labels = []
for label, count in label_counts.items():
    if count < max_count:
        label_indices = [i for i, l in enumerate(train_labels) if l == label]
        for i in range(max_count - count):
            for idx in label_indices:
                augmented_img = datagen.flow(np.expand_dims(train_images[idx], 0), batch_size=1)
                augmented_images.append(augmented_img[0][0])
                augmented_labels.append(label)

# Combine the original and augmented data
train_images = np.concatenate((train_images, augmented_images), axis=0)
train_labels = np.concatenate((train_labels, augmented_labels), axis=0)


# In[11]:


unique_labels, label_counts = np.unique(train_labels, return_counts=True)
print("Label vs Counts for training data")
# Print the unique labels and their counts
for label, count in zip(unique_labels, label_counts):
    print(f"Label: {label}, Count: {count}")


# In[14]:


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),  
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(5, activation='softmax')])
# Compile the model with appropriate loss and metrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=3, restore_best_weights=True)


# In[15]:


def lr_schedule(epoch):
    if epoch < 5:
        return 0.0001
    elif epoch < 10:
        return 0.00005
    elif epoch < 15:
        return 0.00001
    else:
        return 0.000005

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


# In[16]:

class PrintAccuracyCallback(Callback):
    def __init__(self, test_images, test_labels):
        self.test_images = test_images
        self.test_labels = test_labels

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.test_images)
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(y_pred == self.test_labels)
        acc = acc / self.test_labels.shape[0] * 100
        print(f"Accuracy after epoch {epoch + 1}: {acc}%")

# Assuming you have the 'model' and other required variables set up
history = model.fit(train_images, train_labels, batch_size=32, epochs=5, callbacks=[lr_scheduler,PrintAccuracyCallback(test_images, test_labels), early_stopping])


# In[17]:


y_pred = model.predict(test_images)
y_pred = np.argmax(y_pred,axis=1)
acc= np.sum(y_pred==test_labels)
acc = acc/test_labels.shape[0]*100
print(acc)


# In[18]:


incorrect_indices = np.where(y_pred != test_labels)[0]
misclassified_labels = test_labels[incorrect_indices]
misclassified_predictions = y_pred[incorrect_indices]
unique_classes, class_counts = np.unique(misclassified_labels, return_counts=True)

# Print the class-wise number of incorrect predictions
for cls, count in zip(unique_classes, class_counts):
    print(f"Class {cls} has {count} incorrect predictions.")


# In[25]:


y_pred = model.predict(t_images)
y_pred_indices = np.argmax(y_pred, axis=1)
class_names = ['Combat','DestroyedBuildings', 'Fire', 'Humanitarian Aid and rehabilitation', 'Military vehicles and weapons']
class_names_label = {i: class_name for i, class_name in enumerate(class_names)}

# Print images with labels
num_images = len(t_images)
num_rows = (num_images + 2) // 3  # To ensure at most 3 images per row

fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows))
for i, ax in enumerate(axes.flat):
    if i < num_images:
        ax.imshow(t_images[i])
        ax.set_title(class_names_label[y_pred_indices[i]])
        ax.axis('off')  
    else:
        ax.axis('off')  

plt.tight_layout()
plt.show()


# In[26]:


model.save("D:\\HP\\users\\OneDrive\\Desktop\\eyantra\\Task_1A\\task_2b_evaluator\\hello.h5")


# In[24]:


model.save_weights("D:\\HP\\users\\OneDrive\\Desktop\\eyantra\\Task_1A\\task_2b_evaluator\\weights.h5")


# In[ ]:

