import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

"""
Data olarak 2062 taneden oluşan image işaret dili datası kullnacağız
0,1,2,3,4,5,6,7,8,9 sayılarının işaret dillerine karşılık gelen imageler
Sadece 0-1 classifle edilecek
204-408 => 0 
822-1027=> 1
X.npy = Resimler
Y.npy = Resimlern matetiksel ifadelere karşılık gelen sayıları
"""
x_l=np.load('X.npy')#(64,64) lük 2062 tane array
y_l=np.load('Y.npy')

img_size=64

plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(img_size,img_size))
"""
plt.subplot(1,2,1),2.paremetre kaç tane subplot olacağı , 2 tane olsun
3.paremetre hangisinde olsun , 1.si olsun
"""

plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(img_size,img_size))

X=np.concatenate((x_l[204:409],x_l[822:1027]),axis=0)#(410,64,64)
"""X in içerisine 0-1 resilerini attık"""


z= np.zeros(205)
o=np.zeros(205)
Y=np.concatenate((z,o),axis=0).reshape(X.shape[0],1)
"""Resimlerin label birleştirdik"""
#%%Train-Test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.15,random_state=42)

number_of_train=x_train.shape[0] #348
"""
x_train(348,64,64)
x_train.shape[0] =>348
x_train.shape[1] => 64
x_train.shape[2] => 64 
"""
number_of_test=x_test.shape[0] #62

x_train_flatten=x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[1])
"""X datasını 2D hale getimemiz lazım"""
x_test_flatten=x_test.reshape(number_of_test,x_test.shape[1]*x_test.shape[1])

x_train=x_train_flatten.T
x_test=x_test_flatten.T
y_train=y_train.T
y_test=y_test.T
""" Çarpım için Transpozunu aldık"""


#%%ANN Keras ile İplement

# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T



from keras.wrappers.scikit_learn import KerasClassifier
"""
Bir data üzerinde Keras ile classifier yapmak için
kullanmamız gereken modul ve algoritma
"""

from sklearn.model_selection import cross_val_score
"""
Bir data sette cross_val_score =2 dediğimiz zaman 5'e bölüyor
4 bölümü train, 1 bölümü test olarak kabul ediyor
Daha sonra bu işlemi 5 kez ard arda tekrarlıyor ,
4 bölümlük kısmı train , 1 bölümlük kısmı test olacak şekilde
5 farklı train,test seti tanımlıyor ve bunlarım hepsi için bir accuracy buluyor
accuracy ortalamasını olarak ortalama bir accuracy değeri elde etmiş oluyoruz
"""

from keras.models import Sequential#initialize neural network library
"""
parametreler = weight(w) ve bias(b)
Sequential kullanarak w ve b initialize etmiş olucağız
"""

from keras.layers import Dense
"""
Layerlarımızı build , yanı construck(kurmak,inşaa etmek) etmek
"""

def build_classifier():
#->Neural Networkumu oluşturacak yapı olacak    
    classifier = Sequential() #initialize neural network
# """
# Sequential() metodu çağrılacak, benim için bir Neural Network Yapısı inşaa et
# classifier bir neural network olduğunu burada define et
# """


    classifier.add(Dense(units=8,kernel_initializer='uniform',activation= 'relu', input_dim = x_train.shape[1]))
# """
# 1.Hidden Layer
# classifier.add(Dense())-> Bir tane Layer oluştur
# units=8 -> 1.Hidden Layer da 8 tane node olucak
# kernel_initializer='uniform' -> weightlerimi initalize ediliyor , 'uniform' bir şekilde dağılsın,random bir şekilde dağılsın
# activation='relu' , acti. fun. 'relu' olsun daha önceden sigmoid kullanmıştık
# input_dim=x_train.shape[1] -> input dimension , yani vereceğim datadakı sample sayısı (4096)
# """    
    

    classifier.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))
# """
# 2.Hidden Layer
# units=4 -> 4 tane node oluşsun
# kernel_initializer='uniform' -> w ve b random bir şekilde initialize edilsin
# activation='relu' -> Activation function relu olsun
# """    
    

    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
# """
# 3.Hidden Layer
# units=1 -> 1 tane node oluşsun
# kernel_initializer='uniform' -> w random bir şekilde dağılsın
# activation='sigmoid' -> en son act. fun sigmoid olsun , yani Output Layer ekledim
# """    

#--> Bu kısma kadar Foward Propagation tanımlandı

    

    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# """
# ->loss='binary_crossentropy' = loss(eror) bulmak için bu metodu kul.
# ->Bacward Propagation için kullanacağımız metod => optimizer='adam'
# 'adam' metodunda learning rate sabit olmuyor ve daha adaktif oluyor , adapte olarak daha fazla hızlı öğreniyor
# memormızı daha iyi kullanıyoruz
# momentumlu bir şekilde learning rate değişen hali
# ->metrics=['accuracy']
# Değerlendirme metodu , modeli değerlendirirken accuracy kullanacağımızı define ediliyor
# """    
    return classifier

#--> Artık Classier buil edildi ve bundan sonra bunu çağırmamız lazım

classifier=KerasClassifier(build_fn=build_classifier,epochs=100)
"""
->Classier buil edildi ve  çağırmamız lazım , Bunu KerasClassifier ile gerçekleştiriyoruz
->build_fn=build_classifier -> Buil edilen classifier 
->epochs=100 = number of iteration , tekrarlama sayım
"""

#-->Bundan sonra geriye birtek eğitmek kalıyor
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=3)
"""
->cross_val_score ile birden fazla accuracies veriyor ve daha efektif bir sonuç elde etmiş oluyoruz
->estimator=classifier = benim kullanacağım classifierım
->,X=x_train,y=y_train , X ve Y parametreleri
->cv=3 , Benim için 3 kez accuracy bul ve bunların ortalamasını alıyorum
"""
mean=accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))
