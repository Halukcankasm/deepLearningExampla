from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import matplotlib.pyplot as plt
from glob import glob #Kaçtane Sınıfım olduğunu öğrenmek için

train_path = "fruits_360/train/"
test_path = "fruits_360/test/"

img = load_img(train_path +"Apple Braeburn/0_100.jpg" )
plt.imshow(img)
plt.axis("off")
plt.show()


x= img_to_array(img)
print(x.shape)
"""
(100, 100, 3)
->x eksenindeki pixsel sayısı , 100 , x.shape[0]
->y eksenindeki pixel sayısı 100, x.shape[1]
->RGB , 3 , x.shape[3]
"""

className = glob(train_path+ '/*')
"""
train_path içerisine gir ve bütün dosyaları gez ve
className içerisine yükle
"""
numberOfClass = len(className)
print("Number Of Class:",numberOfClass)

#%% Create Model [Cnn]


#
model = Sequential()
model.add(Conv2D(filters = 32,kernel_size=(3,3),input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

#
model = Sequential()
model.add(Conv2D(filters = 32,kernel_size=(3,3),input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

#
model = Sequential()
model.add(Conv2D(filters = 32,kernel_size=(3,3),input_shape=x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

#
model.add(Flatten())

#ANN
model.add(Dense(units=1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(units=numberOfClass)) #output
model.add(Activation("softmax"))
         
model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics = ["accuracy"])



batch_size = 32
#Her iterasyonda 32 tane resmimi train edicem

#%% Data Generation - Train - Test

train_datagen=ImageDataGenerator(rescale = 1./255,
                   shear_range=0.3,
                   #Resmi belli bir açı ile sağa ve sola döndürür
                   horizontal_flip=True,
                   #Resmi random bir şekilde sağa veya sola çevir
                   zoom_range = 0.3
                   )


test_datagen=ImageDataGenerator(rescale = 1./255)
                   
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=x.shape[:2],#(100,100)
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode ="categorical")

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=x.shape[:2],#(100,100)
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode ="categorical")


model.fit_generator(
    generator=train_generator,
    steps_per_epoch = 1600 // batch_size,
    epochs = 100,
    validation_data = test_generator,
    validation_steps = 800 // batch_size)


"""
    resim sayım = 400
    400//batch_size=32 = 12
    12 seferde tüm resimlerimi train ediyorum
    32+32+32+.................+32 = 12 kez 
    tek seferde 32 resmi train ediyorum
    32 lik paketler halinde
    Ama biz data- generate ediyoruz , sayısını bilmiyoruz
    1600//32 = 50 , yani biz 50 kez 32 resmi train etmek istiyoruz
    Elimizdeki resim sayısı 400 , geriye kalan 1200 tane resim  generation dan 
    geliyor
"""

















