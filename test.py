import numpy as np
from matplotlib import pyplot as plt
import stumpy as mp

# def rmse(y, y_pred):
#     return np.sqrt(np.mean(np.square(y - y_pred)))

# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def normalize(v):
#     norm = np.linalg.norm(v)
#     if norm == 0: 
#        return v
#     return v / norm

# def center(dat):
#     mean = np.mean(dat)
#     if mean > 0:
#         return dat - mean


# outputPath = "data/output/"

# SPLIT = 199

# TS = 0

# data = np.loadtxt(inputPath + "electricity_normal.txt")[:,TS]
# pred = np.loadtxt(outputPath + str(TS) + "pred.txt")

# # pred2 = np.mean(np.tril(pred,0),1)


# truth = data[-(SPLIT-1):]

# print("diag {}".format(1))
# print("RMSE {}".format(rmse(truth,pred)))
# print("MAPE {}".format(mape(truth,pred)))

# plt.plot(truth, 'b', label='truth')
# plt.plot(pred, 'r', label='pred')
# plt.show()
inputPath = "data/input/"
data = np.loadtxt(inputPath + "electricity_normal.txt")[:,3]

window_size = 25  # Approximately, how many data points might be found in a pattern
matrix_profile = mp.stump(data, m=window_size)

plt.plot(matrix_profile[:,0])
plt.show()