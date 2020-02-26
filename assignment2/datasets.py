import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def artificial(seed=42):
    '''Returns the 400 samples from the artificial dataset
    '''
    np.random.seed(seed)
    z = np.random.uniform(low=-1, high=1, size=(400,2))
    result = np.zeros((400,))
    result[np.where(z[:,0]>=0.7)] = 1

    result[np.where((z[:,0]<=0.3) & (z[:,1] >= -0.2 - z[:,0]) )] = 1

    return z, result

def iris():
    '''Returns the IRIS dataset
    '''
    data = load_iris()
    scaler = MinMaxScaler(feature_range=(-1,1))
    z = scaler.fit_transform(data.data)
    return z, data.target

if __name__ == "__main__":
    x, y = artificial()

    plt.figure(figsize=(10, 10))
    ind = np.where(y == 0)
    plt.scatter(x[ind,0], x[ind,1], color='blue', label='Class 0')

    ind = np.where(y == 1)
    plt.scatter(x[ind,0], x[ind,1], color='red', label='Class 1')
    plt.xlabel('z_1', fontsize=18)
    plt.ylabel('z_2', fontsize=18)
    plt.legend(loc='upper left', fontsize=15)
    plt.grid()
    plt.savefig('ex3scatter.pdf')
