import numpy as np
from itertools import count
from matplotlib import pyplot as plt

inputPath = "data/input/"
outputPath = "data/output/"

# PART I --------------------------------------------------------
# ---------------------------------------------------------------

# get first time series
data = np.loadtxt(inputPath + "electricity_normal.txt")[:, 3]

#first just take a look at the time series:
# plt.plot(data)
# plt.show()

# split into train and test
SPLIT = 200
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
Gmats = list(map(np.diag, snapshots, [1] * len(snapshots)))

# generate indication matrices with Yij = 1 iif Gij != 0
Ymats = [np.zeros(G.shape) for G in Gmats]
for i, G in enumerate(Gmats):
    Ymats[i][G.nonzero()] = 1

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

Whop = np.diag([1] * SPLIT, k=1) + np.diag([1] * (SPLIT - 1), k=2)
Wprof = None # soon
W = Whop

# PART IV -------------------------------------------------------
# ---------------------------------------------------------------

# constants
Kdim = 40 # latent space dimension
lmbda = 2e3 # regularization param
gamma = 2e-4 # regularization param
D = W # in case I change my mind about laplacian

# initialisation of U_t, B and A randomly
Umats = [abs(np.random.randn(SPLIT + 1, Kdim))] * len(Gmats)
A = abs(np.random.randn(Kdim, Kdim))
B = abs(np.random.randn(Kdim, Kdim))

# ---------------------------------------------------------------
# update U matrices
assert len(Ymats) == len(Gmats) and len(Gmats) == len(Umats)
it = 0
MAXITER = 400
converged = False
while it < MAXITER and not converged:

    oldA = A # for convergence
    
    for t, Y, G, U in zip(count(), Ymats, Gmats, Umats):
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

        U = U * np.sqrt(np.sqrt(nomiG/denoG))  

    # ---------------------------------------------------------------
    # update B
    utgu = lambda U, G, Y: ((U.T).dot(Y * G)).dot(U)
    utuutu = lambda B: lambda U, Y: U.T.dot(Y * (U.dot(B).dot(U.T))).dot(U) # curry
    utubutu = utuutu(B)

    # formula [8]
    nomiB = sum(map(utgu, Umats, Gmats, Ymats))
    denoB = sum(map(utubutu, Umats, Ymats))
    B = B * (nomiB/denoB)

    # ---------------------------------------------------------------
    # update A
    # formula [9]
    nomiA = sum([Umats[i - 1].T.dot(Umats[i]) for i in range(1, len(Umats))])
    denoA = sum([Umats[i - 1].T.dot(Umats[i - 1]).dot(A) for i in range(1, len(Umats))])
    A = A * (nomiA/denoA)
    
    if it > 100 and np.linalg.norm(A - oldA) < 1e-4:
        print("converged in {} iterations".format(it))
        converged = True
    it = it + 1

# PART V --------------------------------------------------------
# ---------------------------------------------------------------

# forecast
Ulast = Umats[-1]
pred = (Ulast.dot(A)).dot(B).dot((Ulast.dot(A)).T)

# unshift the values
pred = pred - minTrain

np.savetxt(outputPath + str(SPLIT) + "_prediction.txt", pred)