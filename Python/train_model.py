# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%                                               %%%%%
# %%%%%          BISMILLAH HIRRAHMA NIRRAHEEM         %%%%%
# %%%%%                                               %%%%%
# %%%%%         Programmed By: Muzammil Behzad        %%%%%
# %%%%% Center for Machine Vision and Signal Analysis %%%%%
# %%%%%              University of Oulu               %%%%%
# %%%%%                 Oulu, Finland                 %%%%%
# %%%%%                                               %%%%%
# %%%%%        Email: muzammil.behzad@oulu.fi         %%%%%
# %%%%%                                               %%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# import stuff
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# import more stuff
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# paths for thw two dataset folders
mask = r'\mask'
no_mask = r'\no_mask'

# store images from the dataset
images = []
for file in list(os.listdir(mask)):
    images.append(cv2.imread(mask+'/'+file))
    
for file in list(os.listdir(no_mask)):
    images.append(cv2.imread(no_mask+'/'+file)) 
    
images = np.array(images)

# store labels (mask = 1,0 and  no mask = 0,1)
n_mask = len(list(os.listdir(mask)))
n_no_mask = len(list(os.listdir(no_mask)))
labels = np.zeros(( n_mask  + n_no_mask, 2  ))
labels[:n_mask,0] = 1
labels[n_mask:,1] = 1
print('y shape',labels.shape)
print('x shape',images.shape)

# split data for training
from random import shuffle
data_size = labels.shape[0]
index = list(range(data_size))
shuffle(index)
index = np.array(index)
images =   images[index]
labels =   labels[index,:]
x_train =  images[:int(0.8*data_size)]
y_train =  labels[:int(0.8*data_size),:] 
x_test =   images[int(0.8*data_size):]
y_test =   labels[int(0.8*data_size):,:]

from random import randint
i = randint(0, len(x_test))
plt.imshow(x_test[i][:,:,::-1])
print((np.array(['With Mask','Without Mask'])[y_test[i] == 1])[0])

# initialize network parameters
img_rows, img_cols = 160, 160
input_shape = (img_rows, img_cols, 3)
batch_size = 1
num_classes = 2
epochs = 15

# creat a keras model from scratch
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.summary()

# train the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# save the model
modelsave_name = 'quarantine_face_mask.h5'
model.save(modelsave_name)

# restore the trained model (you can skip training the model by using just this)
new_model = keras.models.load_model(modelsave_name)

# test the model
from random import randint
i = randint(0, 552)
m = new_model.predict(x_test[i].reshape(-1,160,160,3)) == new_model.predict(x_test[i].reshape(-1,160,160,3)).max()
plt.imshow(x_test[i][:,:,::-1])
print(np.array(['With Mask','Without Mask'])[m[0]])