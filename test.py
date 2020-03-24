import numpy as np
from matplotlib import pyplot as plt

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

inputPath = "data/input/"
outputPath = "data/output/"

SPLIT = 200

data = np.loadtxt(inputPath + "electricity_normal.txt")[:, 3]
pred = np.loadtxt(outputPath + str(SPLIT) + "_prediction.txt")

pred = np.diag(pred, 1)
truth = data[-SPLIT:]

print(rmse(truth,pred))

plt.plot(truth, 'b', label='truth')
plt.plot(pred, 'r')
plt.show()



