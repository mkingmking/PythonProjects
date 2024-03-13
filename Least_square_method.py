import numpy as np
import pandas as pd


data = {
"meal"  : [1,1,2,0,1,3],
"salad" : [0,1,2,1,0,1],
"price" : [9,16,30,5,11,39]
}

df = pd.DataFrame(data)


X = df.iloc[:, :-1].values

y = df.iloc[:,2].values.reshape(-1, 1)




def leastsquare(X, y):
    XTX = np.dot(X.T, X)# To be changed
    XTXINV = np.linalg.inv(XTX)  # To be changed
    XTY = np.dot(X.T, y) # To be changed 
    w = np.dot(XTXINV,XTY )  # To be changed
    return w


w = leastsquare(X, y)
print(w)



n = 6 # To be changed # sample size
d = 2 # To be changed # feature size

#n, d = X.shape
alpha = 0.01
Js = []
w = np.random.rand(d ,1) # To be changed # random initial values


#y.reshape(-1, 1)
for i in range(1000):
    # Compute prediciton of the model
    y_hat = np.dot(X, w)
    # Compuete derivative
    
    dw = (1/n) * np.dot(X.T, (y_hat -  y))
    # Gradient Descent
    w = w - alpha * dw
    if i%100:
        Js.append((1 / (2 * n)) * np.sum(np.power(y - y_hat,2)))


print("w: {}\n".format(w))



