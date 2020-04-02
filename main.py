from lsmrn import *
import numpy as np
import matplotlib.pyplot as plt
def main():
    
    TS = np.loadtxt("data/input/electricity_normal.txt")[:,19]
    params = {
        "time_series":TS,
        "split":100,
        "snap":100,
        "mp_window":25,
        "kdim":30,
        "diag":1,
        "gamma":2e-4,
        "lambda":1e-1
    }
    lsmrn = Lsmrn(params)
    lsmrn.doForecast()

    plt.plot(lsmrn.truth,label='truth')
    plt.plot(lsmrn.preds,label='pred')
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()