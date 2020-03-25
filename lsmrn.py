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
        self.D = self.laplacian()

        self.Umats = self.makeU()
        self.Amats = self.randomMat()
        self.Bmats = self.randomMat()

        self.preds = None
    
    def laplacian(self):

        D = [np.zeros((W.shape)) for W in self.W]
        for i, W in enumerate(self.W):
            np.fill_diagonal(D[i], np.sum(W, 0))
        return D

    def save(self):

        for i, pred in enumerate(self.preds):
            np.savetxt(self.outPath + str(i) + "pred.txt", *pred)

    def forecast(self):
        
        window = int(self.params["pred_steps"]/self.params["snap_size"])
        preds = [[] for _ in range(self.nbTTS)]
        Umats = self.Umats

        for ts, A, B in zip(count(), self.Amats, self.Bmats):
            for w in range(window):
                preds[ts].append((Umats[ts][-1].dot(A)).dot(B).dot((Umats[ts][-1].dot(A)).T))
        
        self.preds = preds
        self.unshiftPred()
        
    def fit(self):

        Kdim = self.params["Kdim"]
        lmbda = self.params["lambda"]
        gamma = self.params["gamma"]

        for ts, ymats, gmats, W, D in zip(
        count(),
        self.Ymats,
        self.Gmats,
        self.W,
        self.D):

            it = 0
            MAXITER = self.params["maxiter"]
            converged = False
            while it < MAXITER and not converged:

                oldA = self.Amats[ts].copy() # for convergence

                for t, Y, G in zip(count(), ymats, gmats):
                    # in the pseudo-code U_t-1 and U_t+1 are not
                    # defined for 1st & last iteration
                    if t == 0 or t == len(gmats) - 1:
                        continue

                    # formula [7]
                    nomiG = (Y * G).dot(self.Umats[ts][t]).dot(self.Bmats[ts].T)
                    + (Y.T * G.T).dot(self.Umats[ts][t]).dot(self.Bmats[ts])
                    +  lmbda * (W + W.T).dot(self.Umats[ts][t])
                    + gamma * (self.Umats[ts][t - 1].dot(self.Amats[ts]) + self.Umats[ts][t + 1].dot(self.Amats[ts].T))

                    denoG = (Y * (self.Umats[ts][t].dot(self.Bmats[ts]).dot(self.Umats[ts][t].T))).dot(self.Umats[ts][t].dot(self.Bmats[ts].T) + self.Umats[ts][t].dot(self.Bmats[ts]))
                    + lmbda * D.dot(self.Umats[ts][t])
                    + gamma * (self.Umats[ts][t] + self.Umats[ts][t].dot(self.Amats[ts]).dot(self.Amats[ts].T))

                    self.Umats[ts][t] *= np.sqrt(np.sqrt(np.nan_to_num(np.divide(nomiG , denoG))))
    
                # update B
                utgu = lambda U, G, Y: ((U.T).dot(Y * G)).dot(U)
                utuutu = lambda B: lambda U, Y: U.T.dot(Y * (U.dot(B).dot(U.T))).dot(U) # curry
                utubutu = utuutu(self.Bmats[ts])

                # formula [8]
                nomiB = sum(map(utgu, self.Umats[ts], gmats, ymats))
                denoB = sum(map(utubutu, self.Umats[ts], ymats))
                self.Bmats[ts] *= np.nan_to_num(np.divide(nomiB,denoB))
               
                # update A
                # formula [9]
                nomiA = sum([self.Umats[ts][i - 1].T.dot(self.Umats[ts][i]) for i in range(1, len(self.Umats[ts]))])
                denoA = sum([self.Umats[ts][i - 1].T.dot(self.Umats[ts][i - 1]).dot(self.Amats[ts]) for i in range(1, len(self.Umats[ts]))])
                self.Amats[ts] *= np.nan_to_num(np.divide(nomiA,denoA))

                if it > 100 and it % 3 == 0 and np.linalg.norm(self.Amats[ts] - oldA) < 1e-3:
                    print("converged in {} iterations".format(it))
                    converged = True
                it = it + 1
                


    def makeU(self):

        Umats = [[] for _ in range(self.nbTTS)]

        NBsnaps = len(self.Gmats[0])
        size = self.params["snap_size"]
        kdim = self.params["Kdim"]

        for i in range(self.nbTTS):
            Umats[i] = [abs(np.random.randn(size, kdim)) for _ in range(NBsnaps)]
        
        return Umats

    def proxMat(self):

        khop = self.params["prox_mat"]
        size = self.params["snap_size"]
        assert type(khop) == int and khop > 0 and khop < size

        W = sum([np.diag([1] * (size - x), k=(x)) for x in range(khop)])
        Whop = [W for _ in range(self.nbTTS)]
        return Whop

    def randomMat(self):
        Kdim = self.params["Kdim"]
        X = abs(np.random.randn(Kdim, Kdim))
        return [X for _ in range(self.nbTTS)]

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
            Gmats[i] = list(map(np.diag, snap))

        return Gmats

    def snaps(self):

        size = self.params["snap_size"]
        assert type(size) == int and size > 1

        snaps = [[] for _ in range(self.nbTTS)]
        for i, ts in enumerate(self.train):
            snaps[i] = [ts[x : x + size] for x in range(0, len(ts), size)]

        return snaps

    def unshiftPred(self):
        for i, minVal in enumerate(self.shiftVals):
            self.preds[i] -= (minVal + 1)

    def shiftTrain(self):

        minVals = [float('inf')] * self.nbTS
        for i, ts in enumerate(self.train):
            minVals[i] = np.amin(ts)
            if minVals[i] < 0:
                self.train[i] = ts + abs(minVals[i]) + 1

        return minVals

    def split(self):
        steps = self.params["pred_steps"]
        return self.data[:, 0:-steps]
        
    def loadData(self):
        data = np.loadtxt(self.inPath)
        if data.shape[0] > data.shape[1]:
            data = data.T
            print("data {}".format(data.shape))
            
        return data[0:1,:]

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
        "snap_size": 200,
        "pred_steps": 200,
        "prox_mat": 5,
        "Kdim": 40,
        "lambda": 1000,
        "gamma": 0.0001,
        "maxiter": 800
    }

    inputPath = "data/input/electricity_normal.txt"
    outputPath = "data/output/"

    myModel = LSMRN(inputPath, outputPath, param)
    myModel.fit()
    myModel.forecast()
    myModel.save()
