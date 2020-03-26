import numpy as np
from itertools import count
from matplotlib import pyplot as plt

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# def autocorr(x):
#     result = np.correlate(x, x, mode='full')
#     return result[result.size // 2:]

inputPath = "data/input/"
outputPath = "data/output/"

# PART I --------------------------------------------------------
# ---------------------------------------------------------------

# get first time series
data = np.loadtxt(inputPath + "electricity_normal.txt")[:,0]

# first, just take a look at the time series:
# plt.plot(data)
# plt.show()

# split into train and test
SPLIT = 200
train = data[0 : -SPLIT]

# negative values are not supported, shift
minTrain = np.amin(train)
if minTrain < 0:
    minTrain = abs(minTrain) + 1
    train += minTrain

SNAP = 50
# split train data in list of snapshots. [[0:49], [50,99]....]
snapshots = [train[x : x + SNAP] for x in range(0, len(train), SNAP)]

# PART II -------------------------------------------------------
# ---------------------------------------------------------------

# generate diagonal matrices.
Gmats = list(map(np.diag, snapshots))

# generate indication matrices with Yij = 1 iif Gij != 0
Ymats = [np.zeros(G.shape) for G in Gmats]
for i, G in enumerate(Gmats):
    Ymats[i][G.nonzero()] = 1

# proximity matrix W using k-hop simil. (prox-hop)
prox = 10
Whop = sum([np.diag([1] * (SNAP - x), k=x) for x in range(prox)])
Wprof = None # soon
W = Whop

# PART IV -------------------------------------------------------
# ---------------------------------------------------------------

# constants
Kdim = 30 # latent space dimension
lmbda = 0.001 # regularization param
gamma = 2e-2 # regularization param

# Laplacian smoothing
D = np.zeros((W.shape)) 
np.fill_diagonal(D, np.sum(W,0))

# initialisation of U_t, B and A randomly
Umats = [0.5 * (abs(np.zeros((SNAP, Kdim))) + 1)] * len(Gmats)

A = 0.5 * (abs(np.zeros((Kdim, Kdim))) + 1)
B = 0.5 * (abs(np.zeros((Kdim, Kdim))) + 1)

# ---------------------------------------------------------------
# update U matrices
assert len(Ymats) == len(Gmats) and len(Gmats) == len(Umats)

it = 0
MAXITER = 600
converged = False
while it < MAXITER and not converged:
    
    oldA = A.copy() # for convergence
    
    for t, Y, G, U in zip(count(), Ymats, Gmats, Umats):
        
        # in the pseudo-code U_t-1 and U_t+1 are not
        # defined for 1st & last iteration
        if t == 0 or t == len(Gmats) - 1:
            continue

        # formula [7]
        nomiG = (Y * G).dot(U).dot(B.T) + (Y.T * G.T).dot(U).dot(B) + lmbda * (W + W.T).dot(U) + gamma * (Umats[t - 1].dot(A) + Umats[t + 1].dot(A.T))
        denoG = (Y * (U.dot(B).dot(U.T))).dot(U.dot(B.T) + U.dot(B)) + lmbda * D.dot(U) + gamma * (U + U.dot(A).dot(A.T))      

        Umats[t] = Umats[t] * np.sqrt(np.sqrt(np.nan_to_num(np.divide(nomiG, denoG))))  
    
    # ---------------------------------------------------------------
    # update B
    utgu = lambda Uu, Gg, Yy: ((Uu.T).dot(Yy * Gg)).dot(Uu)
    utuutu = lambda Bb: lambda Uu, Yy: Uu.T.dot(Yy * (Uu.dot(Bb).dot(Uu.T))).dot(Uu) # curry
    utubutu = utuutu(B)

    # formula [8]
    nomiB = sum(map(utgu, Umats, Gmats, Ymats))
    denoB = sum(map(utubutu, Umats, Ymats))
    B = B * np.nan_to_num(np.divide(nomiB, denoB))

    # ---------------------------------------------------------------
    # update A
    # formula [9]
    nomiA = sum([Umats[i - 1].T.dot(Umats[i]) for i in range(1, len(Umats))])
    denoA = sum([Umats[i - 1].T.dot(Umats[i - 1]).dot(A) for i in range(1, len(Umats))])
    A = A * np.nan_to_num(np.divide(nomiA, denoA))

    if it > 50 and it%2==0 and np.linalg.norm(oldA - A) <1e-4 :
        print("converged in {} iterations".format(it))
        converged = True
    it = it + 1

# PART V --------------------------------------------------------
# ---------------------------------------------------------------

# forecast
Ulast = Umats[-2].copy()
Acopy = A.copy()
preds = None
firstIt = True
for i in range(int(SPLIT/SNAP)):
    pred = (Ulast.dot(Acopy)).dot(B).dot((Ulast.dot(Acopy)).T)
    
    if firstIt:
        firstIt = False
        preds = np.diag(pred, 0)
    else:    
        preds = np.concatenate(([preds, np.diag(pred, 0)]))
        Acopy = Acopy.dot(A)

# unshift the values
preds = preds - minTrain
truth = data[-SPLIT:]

#metrics
print("rmse {}".format(rmse(truth,preds)))
print("mape {}".format(mape(truth,preds)))

#plots
plt.plot(truth, 'b', label='truth')
plt.plot(preds, 'r', label="pred")
plt.show()
