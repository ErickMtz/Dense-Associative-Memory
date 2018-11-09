    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:38:54 2018

@author: ErickMtz
"""

import math
import numpy as np
import os
try:
    prior = os.nice(-20)
    print(prior)
except OSError as e:
    print(e)


n = 30
T = 700
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
T = np.array([z[x+5000:y] for x,y in zip(np.cumsum([0]+nImages[0:-1]),np.cumsum(nImages))]) #Testing set

tX = np.array(sorted(np.identity(Nc).tolist() * imagesPerClassInMinibatch, reverse=True)) #Training labels set
#tT = sorted(np.repeat(np.identity(10, dtype=np.int), repeats = [n-5000 for n in nImages], axis=0).tolist(), reverse=True) #Testing labels set


# Training
W = np.random.normal(-0.3, 0.3, (K, N+Nc))
V = np.concatenate((X[:, np.random.choice(nExamplesPerClass, int(K/Nc))].reshape(K,N), np.array(sorted(np.identity(Nc).tolist() * int(K/Nc), reverse=True))),axis=1)
np.random.shuffle(V)


nEpochs = 3000

eo = 0.01 # 0.01<= eo >= 0.04
fe = 0.998

M = imagesPerClassInMinibatch*Nc #Minibatch size
nUpdates = int(len(X)*len(X[0])/M)
print(nEpochs, nUpdates)

for epoch in range(nEpochs):
    e = eo*np.float_power(fe,epoch) 
    for t in range(nUpdates):
        print("epoch = ", epoch, t)
        v = np.array([i[t*imagesPerClassInMinibatch:t*imagesPerClassInMinibatch+imagesPerClassInMinibatch] for i in X]).reshape(M, N)
        
        ## MINIBATCH
        dW = np.zeros((K,N+Nc))
        c = np.zeros((M,Nc)) - 1
        for A in range(M):
            c[A] = tanh(B * np.sum(W[0:K,N:N+Nc] * f(np.dot(W[0:K,0:N], np.transpose(v[A]))).reshape(K,1),axis=0))
 
        aux1 = np.float_power(c - tX, 2*m-1) * (1 - np.float_power(c, 2))
        for miu in range(K):
            aux2 = np.dot(W[miu,0:N], np.transpose(v))
            dW[miu,0:N] = 2*m*B*n*np.sum(np.sum(aux1 * W[miu,N:N+Nc] * f1(aux2).reshape(M,1),axis=1).reshape(M,1) * v, axis=0).reshape(1,N)
            dW[miu,N:N+Nc]= 2*m*B*np.sum(aux1 * f(aux2).reshape(M,1), axis=0)


        for miu in range(K):
            for I in range(N+Nc):
                V[miu][I] = p*V[miu][I] - dW[miu][I]
                W[miu][I] = W[miu][I] + (e*V[miu][I]/np.max(V[miu]))   
                if W[miu][I] < -1:
                    W[miu][I] = -1
                elif W[miu][I] > 1:
                    W[miu][I] = 1

np.savetxt('W', W, delimiter=",")
print("END TRAINING")

#Testing
import matplotlib.pyplot as plt
for i in range(25):
    plt.imsave(str(i),W[i,0:N].reshape(width, height), vmin=-1, vmax=1,  cmap="coolwarm", format="png")

# W = np.loadtxt("W.txt", delimiter=",")

confusion_matrix = np.zeros((Nc,Nc), dtype=int)

for i in range(Nc):
    for j in range(len(T[i])):
        tsn = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1])
        c = np.zeros(Nc) - 1 
        for alpha in range(Nc):
            tsn[alpha] = 1
            eng1 = np.sum(f(np.dot(np.concatenate((T[i][j], tsn), axis=0), np.transpose(W))))
            tsn[alpha] = -1
            eng2 = np.sum(f(np.dot(np.concatenate((T[i][j], tsn), axis=0), np.transpose(W))))
            c[alpha] = tanh(B*(eng1-eng2))
        confusion_matrix[i][np.argmax(c)] += 1
    
np.savetxt('confusionMatrix', confusion_matrix, delimiter=",")
