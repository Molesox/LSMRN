from lsmrn import *
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def modelPlot():

    data = []
    with open("results", 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    # print(data)
    errors = [err["mape"] for err in data]
    windows = [w["gamma"] for w in data]
    
    plt.plot(windows, errors)
    plt.show()

    minIdx = errors.index(min(errors))
    pred = data[minIdx]["pred"]
    truth = data[minIdx]["truth"]

    print("gamma {}".format(data[minIdx]["gamma"]))
    
    plt.plot(pred, label="pred")
    plt.plot(truth, label="truth")
    plt.legend(loc="upper left")
    plt.show()

def main():
    
    TS = np.loadtxt("data/input/electricity_normal.txt")[:,1]
    print("Time series shape {}".format(TS.shape[0]))

    params = {
        "time_series":TS,
        "split":200,
        "snap":100,
        "mp_window":20,
        "kdim":31,
        "diag":1,
        "gamma":0.25,
        "lambda":0.5
    }

    lsmrn = Lsmrn(params, toCenter = True, toShift=False)
    lsmrn.doForecast()

    with open("results", 'rb') as fr:
        data = pickle.load(fr)
    os.remove("results")

    pred = data["pred"]
    truth = data["truth"]

    plt.plot(pred, label="pred")
    plt.plot(truth, label="truth")
    plt.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
    # modelPlot()
