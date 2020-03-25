import numpy as np
from matplotlib import pyplot as plt

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

inputPath = "data/input/"
outputPath = "data/output/"

SPLIT = 200

TS = 0

data = np.loadtxt(inputPath + "electricity_normal.txt")[:,TS]
pred = np.loadtxt(outputPath + str(TS) + "pred.txt")

pred = np.diag(pred, 1) -2

truth = data[-(SPLIT-1):]

print(rmse(truth,pred))

plt.plot(truth, 'b', label='truth')
plt.plot(pred, 'r')
plt.show()



