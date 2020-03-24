# Latent Space Model for Road Networks

## Introduction
This is my implementation of the well known algorithm for traffic forecast; the Latent Space Model for Road Networks (**LSMRN**).

## Some considerations
Here I try to adapt **LSMRN** to work with time series in general.

I assume the time series being a weighted (planar) path graph. Where the weights between two vertices is the time series value.

The idea is to use spatial information among the time series learned by another data-driven learning method as the _matrix profile_ or other correlation methods to define distances between the nodes.

To achieve that I use first the original proximity matrix and the 
matrix profile model.

## Code structure
### The (logic of) code is structured as follow:

1. split data [0:99], [100,199], ...
   (this size is arbitrry)

2. generate adjency matrices G_1, G_2, ... from the splits
   - see data as path graph, then construct adj
   - create corresponding Y_t indication matrices

3. generate proximity matrix W over all the data using
   - Kinda AR model, i.e. k-hope in time (path graph)
   - Matrix profile (soon)

4. implement global learning
   - U_t computation
   - B computation
   - A computation

5. implement forecasting
   - see output adjency matrix as planar graph
   - train NN to get a better interpretation (soon)

6. plots