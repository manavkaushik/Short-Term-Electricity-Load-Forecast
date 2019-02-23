#!/usr/bin/env python
# coding: utf-8

# In[560]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[712]:


def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        
        parameters['W' + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l - 1])) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# In[713]:


parameters = initialize_parameters([56,100,1])

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# In[714]:


def linear_forward(A, W, b):
    
    Z = np.dot(W, A) + b
  
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


# In[715]:


#test

Z, cache = linear_forward(X_train.T, parameters["W1"], parameters["b1"])
Z


# In[716]:


def sigmoid(Z):
    
    A = 1 / (1 + np.exp(- Z))
    
    return A, Z


# In[717]:


def relu(Z):
    
    a = Z * (Z >= 0)
    
    return a, Z


# In[718]:


def linear_activation_forward(A_prev, W, b, activation):
   
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
       
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
       
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    

    return A, cache


# In[719]:


#test

A, cache = linear_activation_forward(X_train.T, parameters["W1"], parameters["b1"], activation= "relu")
A.shape


# In[720]:


def L_model_forward(X, parameters):
   

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implementing [LINEAR -> RELU]*(L-1). Adding "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation= "relu")
        caches.append(cache)
    
    #print(A.min())
        
    
    # Implementing LINEAR -> RELU. Adding "cache" to the "caches" list.
    
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation= "relu")
    #AL, cache = linear_forward(A, parameters["W" + str(L)], parameters["b" + str(L)])
    caches.append(cache)
   
    
    assert(AL.shape == (1,X.shape[1]))
    #print (AL)
            
    return AL, caches


# In[721]:


#test

AL, cache = L_model_forward(X_train.T, parameters)


# In[722]:


AL.min()


# In[723]:


def compute_cost(AL, Y, cache, layers_dims, lambd):
    
    m = Y.shape[1]
    
    cost = - (np.sum(np.square(Y - AL))) / (2 * m)
    
    assert(cost.shape == ())
    
    return cost


# In[724]:


def L2_regularization_cost(cache, y, layers_dims, lambd):
    
    total = 0
    for i in range(0,len(layers_dims) - 1):
        total = total + np.sum(np.square(cache[i][1]))
        
    L2_cost = total * lambd / (2 * y.shape[1])
    
    return L2_cost


# In[725]:


L2 = L2_regularization_cost(cache, y_train.T, layers_dims= [57, 10, 1], lambd= 0.7)
L2.shape


# In[726]:


#test

C = compute_cost(AL, y_train.T, cache, [57, 10, 1], lambd = 0.7)
C.shape


# #test
# 
# C = compute_cost(AL, y_train.T)
# C
# 

# In[727]:


C.shape


# In[728]:


def linear_backward(lambd, dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (np.dot(dZ, A_prev.T)) / m + W * lambd / m
    db = (np.sum(dZ, axis=1).reshape(b.shape)) / m
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


# ### test

# In[729]:


def sigmoid_back(dA, activation_cache):
    m,q=sigmoid(activation_cache) #m stores the sigmoid return values , q stores the buffer returned
    n=1-m
    
    dZ = dA*m*n
    
    return dZ


# sigmoid_back(np.array(([3,4,5],[8,9,0])),np.array(([1,2,3],[4,4,4])))

# In[730]:


def relu_back(dA, activation_cache):
    
    a =  1 * (activation_cache >= 0 )
    
    dZ = dA * a
    
    return dZ


# In[731]:


def linear_activation_backward(lambd, dA, cache, activation):
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
       
        dZ = relu_back(dA, activation_cache)
        dA_prev, dW, db = linear_backward(lambd, dZ, linear_cache)
       
        
    elif activation == "sigmoid":
        
        dZ = sigmoid_back(dA, activation_cache)
        dA_prev, dW, db = linear_backward(lambd, dZ, linear_cache)
        
    
    return dA_prev, dW, db


# In[732]:


#test


# In[733]:


def L_model_backward(AL, Y, caches, lambd):
   
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = Y - AL
   
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(lambd, dAL, caches[L-1], activation= "relu")
    
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(lambd, grads["dA" + str(l+1)], caches[l], activation= "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] =  db_temp
        

    return grads


# In[734]:


#test


# In[735]:


def update_parameters(parameters, grads, learning_rate):
    
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
   
    return parameters


# In[736]:


#test


# In[737]:


def L_layer_model(X, Y, layers_dims, lambd, a0, decay_rate, num_iterations , print_cost=True):
   
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. 
    parameters = initialize_parameters(layers_dims)
    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        
        learning_rate = a0 / (1 + decay_rate * i)

        # Forward propagation: 
        AL, caches = L_model_forward(X, parameters)
       
        # Compute cost.
        cost = compute_cost(AL, Y, caches, layers_dims, lambd)
        
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches, lambd)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
   
    
    return parameters


# # TRAIN STARTS FROM HERE ..... MODEL FUNTIONS DONE

# In[738]:


#test


#Y, b = sigmoid(np.array(Y))

layers_dims = [56,500, 10, 1]

new_para = L_layer_model(X_train.T,y_train.T,layers_dims, lambd = 0.9, a0 = 0.1, decay_rate = 1, num_iterations = 500, print_cost=True)


# In[291]:


df = pd.read_excel("2017_weather_partial.xlsx")


# In[297]:


df.head()


# df.drop('Day', axis = 1, inplace =True)
# df.drop('Year', axis = 1, inplace =True)
# df.drop('Month', axis = 1, inplace =True)
# df.drop('Time', axis = 1, inplace =True)

# In[298]:


ser1=pd.Series(np.random.rand(35040))
ser2=pd.Series(np.random.rand(35040))
ser3=pd.Series(np.random.rand(35040))
ser4=pd.Series(np.random.rand(35040))
ser5=pd.Series(np.random.rand(35040))

m = np.zeros((5,2))
m[0][0] = df['Cloud_Cover'].mean()
m[0][1] = df['Cloud_Cover'].var()
m[1][0] = df['Temperature(C)'].mean()
m[1][1] = df['Temperature(C)'].var()
m[2][0] = df['ApparentTemperature(C)'].mean()
m[2][1] = df['ApparentTemperature(C)'].var()
m[3][0] = df['Humidity'].mean()
m[3][1] = df['Humidity'].var()
m[4][0] = df['WindSpeed(m/s)'].mean()
m[4][1] = df['WindSpeed(m/s)'].var()


# In[299]:


for i in range (35040):
    ser1[i]= ((df.iloc[i][0] - m[0][0]) / m[0][1]) 
    ser2[i]= ((df.iloc[i][1] - m[1][0]) / m[1][1]) 
    ser3[i]= ((df.iloc[i][2] - m[2][0]) / m[2][1]) 
    ser4[i]= ((df.iloc[i][3] - m[3][0]) / m[3][1]) 
    ser5[i]= ((df.iloc[i][4] - m[4][0]) / m[4][1]) 


# In[ ]:


n = []
n.append(ser1.max())
n.append(ser2.max())
n.append(ser3.max())
n.append(ser4.max())
n.append(ser5.max())

n


# df['Cloud_Cover'] = ser1 / n[0]
# df['Temperature(C)'] = ser2 / n[1]
# df['ApparentTemperature(C)'] = ser3 / n[2]
# df['Humidity'] = ser4 / n[3]
# df['WindSpeed(m/s)'] = ser5 / n[4]

# df['Cloud_Cover'] = ser1 / n[0]
# 
# df['ApparentTemperature(C)'] = ser3 / n[2]
# df['Humidity'] = ser4 / n[3]
# df['WindSpeed(m/s)'] = ser5 / n[4]

# In[431]:


df.head()


# #np.mean(df['Cloud_Cover'])
# df.drop('Temperature(C)', axis = 1, inplace =True)
# df.head()

# df.drop("Time", axis = 1, inplace = True)
# df.drop("Day", axis = 1, inplace = True)
# df.drop("Year", axis = 1, inplace = True)
# df.drop("Month", axis = 1, inplace = True)

# In[417]:


K = np.zeros((32000, 56))
for i in range(0,32000):
    for j in range(0, 56):
        K[i][j]=df.iloc[i][j]
    if i % 1000 == 0:
        print(i)


# In[453]:


dfY = pd.read_excel("final2017LoadData.xlsx")


# In[690]:


M = np.zeros((32000, 1))
for i in range(0,32000):
    M[i][0] = (dfY.iloc[i][2]/100000)


# In[691]:


X = np.array(K.T)
Y = np.array(M.T)


# In[692]:


Y.min()


# # Train-Test Split

# In[590]:


from sklearn.model_selection import train_test_split


# In[693]:


#X should be m*n where m is number of examples and n is number of factors
#Y should be m*1 where m is number of examples


# In[698]:


X_train, X_test, y_train, y_test = train_test_split(X.T,Y.T, test_size=0.1)


# In[699]:


X_dev , X_test , y_dev , y_test = train_test_split(X_test,y_test,test_size=0.3)


# In[700]:


X_train.shape


# In[595]:


#print(y_train.shape ,'Helllllo\n\n\n', X_dev.shape ,'Helllllo\n\n\n', X_test.shape)


# # Test Case
# 
# K_test = np.zeros((1,60))
# 
# for i in range(0, 60):
#     if i == 57:
#         K_test[0][i] = df.iloc[32002][i] / 10000000
#     elif i == 58:
#         K_test[0][i] = df.iloc[32002][i] / 10000
#     elif i == 59:
#         K_test[0][i] = df.iloc[32002][i] / 10000
#     else:
#         K_test[0][i] = df.iloc[32002][i]
# 
# Y_test, b = sigmoid((dfY.iloc[32002][2]) / 10000)
# X_test = np.array(K_test.T)

# In[745]:


# Test

Y_hat, extra = L_model_forward(X_test.T, new_para)

Y_hat
a = (abs(float(y_test[200][0]) - float(Y_hat.T[1][0])) * 100) / y_test[1][0]
e=[]
for i in range(960):
    e.append((abs(float(y_test[i][0]) - float(Y_hat.T[i][0])) * 100) / y_test[i][0])

#print("The error in your prediction is: " + str(e) + " %")


# In[747]:


Y_hat


# In[744]:


sum(e)/len(e)


# In[44]:


error=[]
for i in range(960):
    m = - np.log((1 / (float(Y_hat.T[i][0]))) - 1) * 10000
    n = - np.log((1 / (float(y_test[i][0]))) - 1) * 10000
    error.append((abs(float(m) - float(n)) * 100) / float(n))
print("The error in your prediction is: " + str(sum(error)/960) + " %")


# In[95]:


y_train.T.shape


# In[97]:


y_train


# In[103]:


X_train.T


# In[104]:


y_train.T


# In[429]:


X_train.T


# In[121]:





# In[ ]:




