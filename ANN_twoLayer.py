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
x_l=np.load('X.npy')
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

X=np.concatenate((x_l[204:409],x_l[822:1027]),axis=0)
"""X in içerisine 0-1 resilerini attık"""


z= np.zeros(205)
o=np.zeros(205)
Y=np.concatenate((z,o),axis=0).reshape(X.shape[0],1)
"""Resimlerin label birleştirdik"""
#%%Train-Test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

number_of_train=x_train.shape[0]
number_of_test=x_test.shape[0]

x_train_flatten=x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[1])
"""X datasını 2D hale getimemiz lazım"""
x_test_flatten=x_test.reshape(number_of_test,x_test.shape[1]*x_test.shape[1])

x_train=x_train_flatten.T
x_test=x_test_flatten.T
y_train=y_train.T
y_test=y_test.T
""" Çarpım için Transpozunu aldık"""


#%%Initializin parameters weights and bias


def initialize_parameters_and_layer_sizes_NN(x_train,y_train):
    parameters={"weight1":np.random.randn(3,x_train.shape[0])*0.1,
                "bias1": np.zeros((3,1)),
                "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                "bias2": np.zeros((y_train.shape[0],1))}
                #Dictionary şeylide tanımlıyoruz
    
                #"weight1":np.random.randn(3,x_train.shape[0])*0.1,
                #sayımı küçültüp random sayılar vermemizi sağlıyor
                #3 olmasının sebebi 3 tane node var
                
    return parameters

#%%Foward Propagation
    
def forward_propagation_NN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    

    #Dictionary olarak tanımlıyoruz
    
    return A2,cache


#%%Lost an Cost Function
    
def compute_cost_NN(A2,Y,parameters):
    logprobs=np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost


#%%Bacward Propagation
    
def bacward_propagation_NN(parameters,cache,X,Y):
    
    dZ2=cache["A2"]-Y
    dW2=np.dot(dZ2.cache["A1".T])/X.shape[1]
    db2=np.sum(dZ2,axis=1,keepdims=True)/X.shape[1]
    dZ1=np.dot(parameters["weight2"].T,dZ2)*(1-np.power(cache["A1"],2))
    dW1=np.dot(dZ1,X.T)/X.shape[1]
    db1=np.sum(dZ1,axis=1,keepdims=True)/X.shape[1]
    grads = {"dweight1":dW1,
             "dbias1":db1,
             "dweight2":dW2,
             "dbias2":db2}
    return grads 


#%%Update Parameters
    

def update_parameters_NN(parameters,grads,learning_rate=0.01):
    parameters={"weight1":parameters["weight"]-learning_rate*grads["dweight1"],
                "bias1":parameters["bias1"]-learning_rate*grads["dbias1"],
                "weight2":parameters["weight2"]-learning_rate*grads["dweight2"],
                "bias2":parameters["bias2"]-learning_rate*grads["dbias2"],}
                #Dictionary şeylide tanımlıyoruz
    return parameters


#%%Prediction with learning parameters
    
def predict_NN(parameters,x_test):
    
    A2,cache = forward_propagation_NN(x_test, parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    #z>0.5, y_head=1 , z<0 y_head=0
    
    for i in range(A2.shape[1]):
        if A2[0,i]<=0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=1
            
    return Y_prediction            
       

#%%Create ANN model

# 2 - Layer neural network
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)         
#num_iterations=2500 , w ve b update 2500 defa             
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

    



















































































