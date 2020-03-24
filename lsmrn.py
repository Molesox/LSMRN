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

    def fit(self):
        for ts, ymats, gmats, umats in zip(
        count(),
        self.Ymats,
        self.Gmats,
        self.Umats):
            print(ts)

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

        Whop = sum([np.diag([1] * (size - x), k=(x + 1)) for x in range(khop)])

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
