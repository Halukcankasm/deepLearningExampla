import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


#read train
train = pd.read_csv("train.csv")
print(train.shape)
print(train.head())

#read test
test=pd.read_csv("test.csv")
print(test.shape)
print(test.head())

#put labels into y_train variable
Y_train = train["label"]

#Drop 'label' column
X_train = train.drop(labels = ["label"],axis =1)

#%%visualize number of digits clases
plt.figure(figsize=(8,5))
#figsize=(15,7) x ve y deki boyutları

g= sns.countplot(Y_train,palette="icefire")
plt.title("Number of digit clases")
print(Y_train.value_counts()) #Hangi sayıdan kaç tane var

#%%#plot some samples

img = X_train.iloc[5]
img_=np.array(img)#matrix haline çevir
img = img_.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.show()

#%%plot some samples
img = X_train.iloc[3]
img=np.array(img)
img = img.reshape((28,28))
plt.title("train.iloc[3,0]")
plt.imshow(img,cmap='gray')
plt.show()

#%%Normalization , Reeshape and Label Encoding

#Train and test image (28x28)
#reshape all data to 28x28x1 3D , x1=grayScale

#Label encoding
#Kerasın anlaması için 1,2,3,4,5,6,7,8,9 sayılarını label encoding yapmamız lazım
#2=>[0,0,1,0,0,0,0,0,0,0]
#4=>[0,0,0,0,1,0,0,0,0,0]


#Normalization , herhangi bir sayıyı 0-1 arasında sınırlandırma
X_train = X_train / 255.0
test = test / 255.0
print("x_train sahape:",X_train.shape)
print("test sahpe:",test.shape)

#Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("x_train sahape:",X_train.shape)
print("test sahpe:",test.shape)


#Label Encoding
from keras.utils.np_utils import to_categorical #convert to one_hot_encoding
Y_train = to_categorical(Y_train,num_classes=10)


#%% Train-Test

from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.1,random_state=42)

print("x_train shape",X_train.shape)
print("x_test shape",X_val.shape)
print("y_train shape",Y_train.shape)
print("y_test shape",Y_val.shape)

#%%

# 
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

model = Sequential()
#
model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
#
model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
# fully connected
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#%% Define the optimizer
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

#%% Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])           


#%% Epoch and BatchSize
epochs = 10  # for better result increase the epochs
batch_size = 250

#%%data augmentation

# data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#%% Fit the model , Train

history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),
                              epochs=epochs,validation_data=(X_val,Y_val),
                              steps_per_epoch=X_train.shape[0]// batch_size)








































