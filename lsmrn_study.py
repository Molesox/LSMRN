import numpy as np
from itertools import count
from matplotlib import pyplot as plt

def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

inputPath = "data/input/"
outputPath = "data/output/"

# PART I --------------------------------------------------------
# ---------------------------------------------------------------

# get first time series
data = np.loadtxt(inputPath + "electricity_normal.txt")[:,1]

#first just take a look at the time series:
# plt.plot(data)
# plt.show()

# split into train and test
SPLIT = 100
train = data[0 : -SPLIT]

# negative values are not supported, shift
minTrain = np.amin(train)
if minTrain < 0:
    minTrain = abs(minTrain) + 1
    train = train + minTrain

# split train data in list of snapshots. [[0:99], [100,199]....]
snapshots = [train[x : x + SPLIT] for x in range(0, len(train), SPLIT)]

# PART II -------------------------------------------------------
# ---------------------------------------------------------------

# generate 1-over diagonal matrices.
Gmats = list(map(np.diag, snapshots))

# generate indication matrices with Yij = 1 iif Gij != 0
Ymats = [np.zeros(G.shape) for G in Gmats]
for i, G in enumerate(Gmats):
    Ymats[i][G.nonzero()] = 1
    # Ymats[i][-1][-1] = 1




# PART III ------------------------------------------------------
# ---------------------------------------------------------------

# there is only one proximity matrix. But we'll construct 2 variants.
# the 1st one is k-hop similarity with k = 2. I.e. W_i_j = 1 if the
# shortest path between two vertices i and j is less than 2.
# as our graph is a path, t1 --> t2 --> t3 --> ... ---> tn
# we get W = |0 1 1 0 ... 0|
#            |0 0 1 1 ... 0|
#            |0    . .    0|
#            |0     . .   0|
#            |0 ...   1 1 0|
#            |0 ...     1 1|
#            |0 ...     0 1|
#            |0 ...       0|

# question, should we put 1 in diagonal ? By now, no.
# but something to consider.
# also, laplacian regulariser L = W - D seems useless
# as we'll get (-2) in the diagonal and other proximity
# matrices that I've seen set a min threeshold of 0.

Whop = sum([np.diag([1] * (SPLIT - x), k=(x)) for x in range(6)])
Wprof = None # soon
W = Whop

# PART IV -------------------------------------------------------
# ---------------------------------------------------------------

# constants
Kdim = 40 # latent space dimension
lmbda = 2e3 # regularization param
gamma = 2e-4 # regularization param
D = np.zeros((W.shape)) 
np.fill_diagonal(D, np.sum(W,0))



# D = W # in case I change my mind about laplacian

# initialisation of U_t, B and A randomly
Umats = [abs(np.random.randn(SPLIT, Kdim))] * len(Gmats)
test = Umats[-1].copy()


A = abs(np.random.randn(Kdim, Kdim))
B = abs(np.random.randn(Kdim, Kdim))

# ---------------------------------------------------------------
# update U matrices
assert len(Ymats) == len(Gmats) and len(Gmats) == len(Umats)
it = 0
MAXITER = 400
converged = False
while it < MAXITER and not converged:
    # print("iter = ", it)
    oldA = A.copy() # for convergence
    # oldB = B.copy()
    
    for t, Y, G, U in zip(count(), Ymats, Gmats, Umats):
        # print("t = ", t)
        # in the pseudo-code U_t-1 and U_t+1 are not
        # defined for 1st & last iteration
        if t == 0 or t == len(Gmats) - 1:
            continue
        # formula [7]
        nomiG = (Y * G).dot(U).dot(B.T)
        + (Y.T * G.T).dot(U).dot(B)
        + 0.5 * lmbda * (W + W.T).dot(U)
        + gamma * (Umats[t - 1].dot(A) + Umats[t + 1].dot(A.T))

        denoG = (Y * (U.dot(B).dot(U.T))).dot(U.dot(B.T) + U.dot(B))
        + lmbda * D.dot(U)
        + gamma * (U + U.dot(A).dot(A.T))      

        U *= np.sqrt(np.sqrt(np.nan_to_num(np.divide(nomiG , denoG))))  
    
    
    # ---------------------------------------------------------------
    # update B
    utgu = lambda U, G, Y: ((U.T).dot(Y * G)).dot(U)
    utuutu = lambda B: lambda U, Y: U.T.dot(Y * (U.dot(B).dot(U.T))).dot(U) # curry
    utubutu = utuutu(B)

    # formula [8]
    nomiB = sum(map(utgu, Umats, Gmats, Ymats))
    denoB = sum(map(utubutu, Umats, Ymats))
    B *= np.nan_to_num(np.divide(nomiB,denoB))
    # Bbis = B.copy()
  

    # ---------------------------------------------------------------
    # update A
    # formula [9]
    nomiA = sum([Umats[i - 1].T.dot(Umats[i]) for i in range(1, len(Umats))])
    denoA = sum([Umats[i - 1].T.dot(Umats[i - 1]).dot(A) for i in range(1, len(Umats))])
    A *= np.nan_to_num(np.divide(nomiA,denoA))
    Abis = A.copy()

    if it > 100 and it%5==0 and np.linalg.norm(oldA - Abis) <1e-3 :
        print("converged in {} iterations".format(it))
        converged = True
    it = it + 1

# PART V --------------------------------------------------------
# ---------------------------------------------------------------

# forecast
Ulast = Umats[-1].copy()
pred = (Ulast.dot(A)).dot(B).dot((Ulast.dot(A)).T)

print("all close {}", np.allclose(test, Ulast))
print("norm {}".format(np.linalg.norm(test -Ulast)))
# unshift the values
pred = pred - minTrain

pred = np.diag(pred, 0)
truth = data[-SPLIT:]

print(rmse(truth,pred))

plt.plot(truth, 'b', label='truth')
plt.plot(pred, 'r')
plt.show()
# np.savetxt(outputPath + str(SPLIT) + "_prediction.txt", pred)