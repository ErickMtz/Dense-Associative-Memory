    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:38:54 2018

@author: ErickMtz
"""

import math
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    prior = os.nice(-20)
    print(prior)
except OSError as err:
    print(err)


n = 30
T = 650
m = 29

K = 2000 #Number of memories
p = 0.9 # 0.6 <= p >= 0.95 Momentum


B = 1/np.float_power(T,n)
f = np.vectorize(lambda x:0 if x<0 else np.float_power(x,n), otypes=[float])    
f1 = np.vectorize(lambda x:0 if x<0 else np.float_power(x,n-1), otypes=[float])
tanh = np.vectorize(lambda x:math.tanh(x))



# Read digits from MNIST database
from mnist import MNIST
mndata = MNIST('samples')
images, labels = mndata.load_training()
height = 28
width = 28
N = height*width
z = [((np.asarray(x)-127.5)/127.5).tolist() for _, x in sorted(zip(labels, images))]
nImages = [labels.count(x) for x in sorted(set(labels))]


nExamplesPerClass = 5000
imagesPerClassInMinibatch = 100
Nc = 10

X = np.array([z[x:x+nExamplesPerClass] for x in np.cumsum([0]+nImages[0:-1])]) #Training set

Te = np.array([z[x+5000:x+5100] for x,y in zip(np.cumsum([0]+nImages[0:-1]),np.cumsum(nImages))]) #Testing set

tX = np.array(sorted((np.identity(Nc)*2-1).tolist() * imagesPerClassInMinibatch, reverse=True)) #Training labels set
#tT = sorted(np.repeat(np.identity(10, dtype=np.int), repeats = [n-5000 for n in nImages], axis=0).tolist(), reverse=True) #Testing labels set

# Training
W = np.random.normal(-0.3, 0.3, (K, N+Nc))
#V = np.random.uniform(-1,1,(K,N+Nc))
V = np.concatenate((X[:, np.random.choice(nExamplesPerClass, int(K/Nc), replace=False)].reshape(K,N), np.array(sorted((np.identity(Nc)*2-1).tolist() * int(K/Nc), reverse=True))),axis=1)
np.random.shuffle(V)


nEpochs = 3000

eo = 0.01 # 0.01<= eo >= 0.04
fe = 0.998

M = imagesPerClassInMinibatch*Nc #Minibatch size
nUpdates = int(len(X)*len(X[0])/M)
print(nEpochs, nUpdates)

from numba import jit

@jit(nopython=True, parallel=True)
def CalculateMinibatch(K, M, N, Nc, B, n, m, dW, c, W, v, tX):
    for A in range(M):
        aux = np.dot(W[0:K,0:N], np.transpose(v[A]))
        for i in range(len(aux)):
            if aux[i] >= 0:    
                aux[i] = math.pow(aux[i], n)
            else:
                aux[i] = 0
        c[A] = B * np.sum(W[0:K,N:N+Nc] * (aux).reshape(K,1),axis=0)
        for i in range(Nc):
            c[A][i] = math.tanh(c[A][i])
    
    aux1 = np.power(c - tX, 2*m-1) * (1 - np.power(c, 2))
    for miu in range(K):
        aux2 = np.dot(W[miu,0:N], np.transpose(v))
        for i in range(len(aux2)):
            if aux2[i] >= 0:    
                aux2[i] = math.pow(aux2[i], n-1)
            else:
                aux2[i] = 0
        dW[miu,0:N] = 2*m*B*n*np.sum(np.sum(aux1 * W[miu,N:N+Nc] * aux2.reshape(M,1),axis=1).reshape(M,1) * v, axis=0)
        dW[miu,N:N+Nc]= 2*m*B*np.sum(aux1 * (aux2*np.dot(W[miu,0:N], np.transpose(v))).reshape(M,1), axis=0)


error_training = np.zeros((nEpochs))+100
error_testing = np.zeros((nEpochs))+100
error_obj = np.zeros((nEpochs))+100
for epoch in range(nEpochs):
    e = eo*np.float_power(fe,epoch) 
    obj_func = 0
    for t in range(nUpdates):
        print("epoch =", epoch+1, ",update =",t)
        v = np.array([i[t*imagesPerClassInMinibatch:t*imagesPerClassInMinibatch+imagesPerClassInMinibatch] for i in X]).reshape(M, N)
        
        dW = np.zeros((K,N+Nc))
        c = np.zeros((M,Nc)) - 1
        ## MINIBATCH
        CalculateMinibatch(K, M, N, Nc, B, n, m, dW, c, W, v, tX)
        
        for miu in range(K):
            for I in range(N+Nc):
                V[miu][I] = p*V[miu][I] - dW[miu][I]
                W[miu][I] = W[miu][I] + (e*V[miu][I]/np.max(abs(V[miu])))   
                if W[miu][I] < -1:
                    W[miu][I] = -1
                elif W[miu][I] > 1:
                    W[miu][I] = 1
        obj_func += np.sum(np.power(c-tX,2*m))
    confusion_matrix = np.zeros((Nc,Nc), dtype=int)
    for i in range(Nc):
        for j in range(len(X[i])):
            tsn = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1])
            c = np.zeros(Nc) - 1 
            for alpha in range(Nc):
                tsn[alpha] = 1
                eng1 = np.sum(f(np.dot(np.concatenate((X[i][j], tsn), axis=0), np.transpose(W))))
                tsn[alpha] = -1
                eng2 = np.sum(f(np.dot(np.concatenate((X[i][j], tsn), axis=0), np.transpose(W))))
                c[alpha] = tanh(B*(eng1-eng2))
            confusion_matrix[i][np.argmax(c)] += 1
    print("training error",(1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100)
    error_training[epoch] = (1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100
    
    confusion_matrix = np.zeros((Nc,Nc), dtype=int)
    
    for i in range(Nc):
        for j in range(len(Te[i])):
            tsn = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1])
            c = np.zeros(Nc) - 1 
            for alpha in range(Nc):
                tsn[alpha] = 1
                eng1 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
                tsn[alpha] = -1
                eng2 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
                c[alpha] = tanh(B*(eng1-eng2))
            confusion_matrix[i][np.argmax(c)] += 1
    print("testing error",(1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100)
    error_testing[epoch] = (1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100
    
    error_obj[epoch] = obj_func
    print(obj_func)

np.save('error_training', error_training, delimiter=",")
np.save('error_testing', error_testing, delimiter=",")
np.save('error_obj', error_obj, delimiter=",")
np.savetxt('W', W, delimiter=",")
print("END TRAINING")

#Testing

for i in range(25):
    plt.imsave(str(i),W[i,0:N].reshape(width, height), vmin=-1, vmax=1,  cmap="coolwarm", format="png")


confusion_matrix = np.zeros((Nc,Nc), dtype=int)

for i in range(Nc):
    for j in range(len(Te[i])):
        tsn = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1])
        c = np.zeros(Nc) - 1 
        for alpha in range(Nc):
            tsn[alpha] = 1
            eng1 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
            tsn[alpha] = -1
            eng2 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
            c[alpha] = tanh(B*(eng1-eng2))
        confusion_matrix[i][np.argmax(c)] += 1
error1 = ((1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100)
print(error1)
np.savetxt('confusionMatrix', confusion_matrix, delimiter=",")
