import  numpy as np
from itertools import count

'''
    These are the parameters of the model:
        i) size of the snapshots
        ii) length of the forecast horizon
        iii) proximity matrix 
            a)  k for k-hop similarity
            b) matrix profile
        iv) latent space dimension
        v) lambda regularization param
        vi) gamma regularization param
        vii) max iterations
'''

class LSMRN:

    def __init__(self, inPath:str, outPath:str, params:dict):

        self.inPath = inPath
        self.outPath = outPath
        self.params = self.checkParameters(params)

        self.data = self.loadData()
        self.dataShape = self.nbTS, self.elems = self.data.shape

        self.train = self.split()
        self.trainShape = self.nbTTS, self.Telems = self.train.shape

        self.shiftVals = self.shiftTrain()
        self.snapshots = self.snaps()

        self.Gmats = self.adjMatrices()
        self.Ymats = self.indicationMat()

        self.W = self.proxMat()
        self.D = self.W.copy()

        self.Umats = self.makeU()
        self.Amats = []
        self.Bmats = []
        
    def fit(self):

        Kdim = self.params["Kdim"]
        lmbda = self.params["lambda"]
        gamma = self.params["gamma"]

        for ts, ymats, gmats, umats, W, D in zip(
        count(),
        self.Ymats,
        self.Gmats,
        self.Umats,
        self.W,
        self.D):

            A = abs(np.random.randn(Kdim, Kdim))
            B = abs(np.random.randn(Kdim, Kdim))

            it = 0
            MAXITER = 400
            converged = False
            while it < MAXITER and not converged:

                oldA = A # for convergence

                for t, Y, G, U in zip(count(), ymats, gmats, umats):
                    # in the pseudo-code U_t-1 and U_t+1 are not
                    # defined for 1st & last iteration
                    if t == 0 or t == len(gmats) - 1:
                        continue

                    # formula [7]
                    nomiG = (Y * G).dot(U).dot(B.T)
                    + (Y.T * G.T).dot(U).dot(B)
                    + 0.5 * lmbda * (W + W.T).dot(U)
                    + gamma * (umats[t - 1].dot(A) + umats[t + 1].dot(A.T))

                    denoG = (Y * (U.dot(B).dot(U.T))).dot(U.dot(B.T) + U.dot(B))
                    + lmbda * D.dot(U)
                    + gamma * (U + U.dot(A).dot(A.T))

                    U = U * np.sqrt(np.sqrt(nomiG/denoG))

                # update B
                utgu = lambda U, G, Y: ((U.T).dot(Y * G)).dot(U)
                utuutu = lambda B: lambda U, Y: U.T.dot(Y * (U.dot(B).dot(U.T))).dot(U) # curry
                utubutu = utuutu(B)

                # formula [8]
                nomiB = sum(map(utgu, umats, gmats, ymats))
                denoB = sum(map(utubutu, umats, ymats))
                B = B * (nomiB/denoB)

                # update A
                # formula [9]
                nomiA = sum([umats[i - 1].T.dot(umats[i]) for i in range(1, len(umats))])
                denoA = sum([umats[i - 1].T.dot(umats[i - 1]).dot(A) for i in range(1, len(umats))])
                A = A * (nomiA/denoA)

                if it > 100 and it % 3 == 0 and np.linalg.norm(A - oldA) < 1e-4:
                    print("converged in {} iterations".format(it))
                    converged = True
                it = it + 1

            self.Amats.append(A)
            self.Bmats.append(B)


    def makeU(self):

        Umats = [[] for _ in range(self.nbTTS)]

        NBsnaps = len(self.Gmats[0])
        size = self.params["snap_size"]
        kdim = self.params["Kdim"]

        for i in range(self.nbTTS):
            Umats[i] = [abs(np.random.randn(size + 1, kdim)) for _ in range(NBsnaps)]
        
        return Umats

    def proxMat(self):

        khop = self.params["prox_mat"]
        size = self.params["snap_size"]
        assert type(khop) == int and khop > 0 and khop < size

        W = sum([np.diag([1] * (size - x), k=(x + 1)) for x in range(khop)])
        Whop = [W for _ in range(self.nbTTS)]
        return Whop

    def indicationMat(self):

        Ymats = [[] for _ in range(self.nbTTS)]
        for i, Gmat in enumerate(self.Gmats):
            Ymats[i] = [np.zeros(G.shape) for G in Gmat]
            for j, G in enumerate(Gmat):
                Ymats[i][j][G.nonzero()] = 1

        return Ymats

    def adjMatrices(self):

        Gmats = [[] for _ in range(self.nbTTS)]
        for i, snap in enumerate(self.snapshots):
            Gmats[i] = list(map(np.diag, snap, [1] * len(snap)))

        return Gmats

    def snaps(self):

        size = self.params["snap_size"]
        assert type(size) == int and size > 1

        snaps = [[] for _ in range(self.nbTTS)]
        for i, ts in enumerate(self.train):
            snaps[i] = [ts[x : x + size] for x in range(0, len(ts), size)]

        return snaps

    def shiftTrain(self):

        minVals = [float('inf')] * self.nbTS
        for i, ts in enumerate(self.train):
            minVals[i] = np.amin(ts)
            if minVals[i] < 0:
                ts = ts + abs(minVals[i]) + 1

        return minVals

    def split(self):
        steps = self.params["pred_steps"]
        return self.data[:, 0:-steps]
        
    def loadData(self):
        data = np.loadtxt(self.inPath)
        if data.shape[0] > data.shape[1]:
            data = data.T
        return data

    def checkParameters(self, params:dict):
        for key in params.keys():
            assert key in [
            "snap_size",
            "pred_steps",
            "prox_mat",
            "Kdim",
            "lambda",
            "gamma",
            "maxiter"]
        return params


if __name__ == "__main__":

    param = {
        "snap_size": 100,
        "pred_steps": 100,
        "prox_mat": 3,
        "Kdim": 10,
        "lambda": 2e3,
        "gamma": 2e-4,
        "maxiter": 400
    }

    inputPath = "data/input/electricity_normal.txt"
    outputPath = "data/output/"

    myModel = LSMRN(inputPath, outputPath, param)
    myModel.fit()
