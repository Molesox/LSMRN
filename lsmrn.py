import numpy as np
import stumpy as stu
import pickle
from itertools import repeat, count

class Lsmrn:

    def __init__(self, params, toCenter=True, toShift=False):

        self.toCenter = toCenter
        self.toShift = toShift

        self.data = params["time_series"]

        self.split = params["split"]
        self.snap = params["snap"]
        self.mpWindow = params["mp_window"]
        self.kdim = params["kdim"]
        self.diag = params["diag"]
        self.lmbda = params["lambda"]
        self.gamma = params["gamma"]


        self.params = params

        self.minTrain = None
        self.train = self.prepareTrain()
        self.snapshots = self.makeSnaps()

        self.Gmats = self.makeAdjMats()
        self.Ymats = self.makeIdxMats()

        self.Wprof = self.makeProxMat()
        self.Laplace = self.makeLaplacian()

        self.fitdone = False
        self.preds = None
        self.truth = None

    
    def doForecast(self):
        A, B, U = self.fit()
        self.forecast(A, B, U)
        self.postprocess()
        self.truth = self.makeTruth()
        self.save()
    
    def rmse(self, y_true, y_pred):
        # root mean square error
        return np.sqrt(np.mean(np.square(y_true - y_pred)))

    def mape(self, y_true, y_pred):
        # mean absolute percentage error
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def save(self):

        self.params.pop("time_series")
        
        errors = {
            "pred": self.preds,
            "truth": self.truth,
            "rmse": self.rmse(self.truth, self.preds),
            "mape": self.mape(self.truth, self.preds)
            }
        errors.update(self.params)

        with open('results','ab') as f:
            pickle.dump(errors, f)
            
    def proxMat(self, mp, size):
        W = np.zeros((size, size))
        for i, val in enumerate(mp[:,0]):
            W[i][mp[i, 1]] = 1./ val
        return W

    def makeTruth(self):
        split = self.split
        diag = self.diag

        until = diag * int(split/self.snap)
        if until != 0 :
            return self.data[- split: -until]
        else:
            return self.data[- split:]

    def fit(self):

        Kdim = self.kdim
        SNAP = self.snap
        W = self.Wprof
        D = self.Laplace
        gamma = self.gamma
        lmbda = self.lmbda

        Umats = [0.5 *  (abs(np.random.randn(SNAP, Kdim)))] * len(self.Gmats)

        A = 0.5 * abs(np.random.randn(Kdim, Kdim))
        B = 0.5 * abs(np.random.randn(Kdim, Kdim))

        it = 0
        maxiter = 600
        converged = False

        while it < maxiter and not converged:
        
            oldA =  A.copy() # for convergence

            # update U matrices
            for t, Y, G, U in zip(count(), self.Ymats, self.Gmats, Umats):
                
                # in the pseudo-code U_t-1 and U_t+1 are not
                # defined for 1st & last iteration
                if t == 0 or t == len(self.Gmats) - 1:

                    nomiG = ((Y * G).dot(U)).dot(B.T) + ((Y.T * G.T).dot(U)).dot(B) + lmbda * (W + W.T).dot(U)
                    denoG = (Y * (U.dot(B).dot(U.T))).dot(U.dot(B.T) + U.dot(B)) + lmbda * D.dot(U) + gamma * (U + U.dot(A).dot(A.T))      

                    Umats[t] = Umats[t] * np.sqrt(np.sqrt(np.nan_to_num(np.divide(nomiG, denoG))))
                    continue

                # formula [7]
                nomiG = (Y * G).dot(U).dot(B.T) + (Y.T * G.T).dot(U).dot(B) + lmbda * (W + W.T).dot(U) + gamma * (Umats[t - 1].dot(A) + Umats[t + 1].dot(A.T))
                denoG = (Y * (U.dot(B).dot(U.T))).dot(U.dot(B.T) + U.dot(B)) + lmbda * D.dot(U) + gamma * (U + U.dot(A).dot(A.T))      

                Umats[t] = Umats[t] * np.sqrt(np.sqrt(np.nan_to_num(np.divide(nomiG, denoG))))  
            
            # ---------------------------------------------------------------
            # update B
            utgu = lambda U, G, Y: ((U.T).dot(Y * G)).dot(U)
            utuutu = lambda B: lambda U, Y: U.T.dot(Y * (U.dot(B).dot(U.T))).dot(U) # curry
            utubutu = utuutu(B)

            # formula [8]
            nomiB = sum(map(utgu, Umats, self.Gmats, self.Ymats))
            denoB = sum(map(utubutu, Umats, self.Ymats))
            B = B * np.nan_to_num(np.divide(nomiB, denoB))

            # ---------------------------------------------------------------
            # update A
            # formula [9]
            nomiA = sum([Umats[i - 1].T.dot(Umats[i]) for i in range(1, len(Umats))])
            denoA = sum([Umats[i - 1].T.dot(Umats[i - 1]).dot(A) for i in range(1, len(Umats))])
            A = A * np.nan_to_num(np.divide(nomiA, denoA))
            
            # check convergence
            if it > 50 and it % 2 == 0 :
                diff = np.linalg.norm(oldA -  A,-2)

                if diff < 1e-4:
                    print("converged in {} iterations".format(it))
                    converged = True

            it = it + 1

        self.fitdone = True
        return A, B, Umats[-1]

    def forecast(self, A, B, U):
        if not self.fitdone:
            raise Exception("Fit must be done before forecasting")
        Acopy = A.copy()
        preds = None
        firstIt = True
        diag = self.diag
        for i in range(int(self.split/self.snap)):

            pred = (U.dot(Acopy)).dot(B).dot((U.dot(Acopy)).T)
           
            if firstIt:
                firstIt = False
                preds = np.diag(pred, diag)
            else:    
                preds = np.concatenate(([preds, np.diag(pred, diag)]))
                Acopy = Acopy.dot(A)
        
        self.preds = preds

    def postprocess(self):
        if self.toCenter:
            self.preds = self.center(self.preds)
        elif self.toShift:
            self.preds = unshift(self.preds, self.minTrain)

    def makeLaplacian(self):
        D = np.zeros((self.Wprof.shape))
        np.fill_diagonal(D, np.sum(self.Wprof, 0))
        return D

    def scale(self, X, x_min, x_max):
        nomi = (X - X.min(axis=0)) * (x_max - x_min)
        deno = X.max(axis=0) - X.min(axis=0)
        deno[deno==0] = 1
        return x_min + nomi/deno 

    def makeProxMat(self):

        window = self.mpWindow
        Pmats = [stu.stump(snap, window) for snap in self.snapshots]
        
        for i in range(len(Pmats)):
            Pmats[i][:,0] *= (i+1)

        Wprof = sum(map(self.proxMat, Pmats, repeat(self.snap, len(Pmats))))
        Wprof = self.scale(Wprof, 0, 1)
        return Wprof

    def makeIdxMats(self):
        Ymats = [np.zeros(G.shape) for G in self.Gmats]
        for i, G in enumerate(self.Gmats):
            Ymats[i][G.nonzero()] = 1
        return Ymats

    def makeAdjMats(self):
        return list(map(np.diag, self.snapshots))

    def makeSnaps(self):
        snap = self.snap
        return [self.train[x : x + snap] for x in range(0, len(self.train), snap)]

    def prepareTrain(self):
        train = self.data[0 : -self.split]
        train, self.minTrain = shift(train)
        
        return train

    def center(self, dat):
        # center the time series around 0
        mean = np.mean(dat)
        return dat - mean

def unshift(ts, val):
    return ts - val

def shift(ts):
    minVal = np.amin(ts)
    if minVal < 0:
        minVal = abs(minVal) + 1
        ts += minVal
    else:
        minVal = 0
    
    return ts, minVal