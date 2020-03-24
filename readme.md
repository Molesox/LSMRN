    Here I try to adapt the Latent Space Model for Road Networks
    to work with time series in general.
    I assume the time series being a (planar) path graph. The idea 
    is to use spatial information among the time series learned by another
    data-driven learning method as the matrix profile or other correlation
    distances.
    To achieve that I use first the original proximity matrix and the 
    matrix profile model.
    The code is structured as follow:

 1) split data [0:99], [100,199], ...
    this size is arbitrry

 2) generate adjency matrices G_1, G_2, ... from the splits
    i) see data as path graph, then construct adj
    ii) create corresponding Y_t indication matrices

 3) generate proximity matrix W over all the data using
    i) Kinda AR model, i.e. k-hope in time (path graph)
    ii) Matrix profile (soon)

 4) implement global learning
    i) U_t computation
    ii) B computation
    iii) A computation

 5) implement forecasting
    i) see output adjency matrix as planar graph
    ii) train NN to get a better interpretation (soon)

 6) plot the (split) * (2*W) * (2*forecast) results
