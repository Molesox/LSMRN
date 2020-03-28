# Latent Space Model for Road Networks

## Introduction
This is my implementation of the well known algorithm for traffic forecast; the [Latent Space Model for Road Networks](http://www-scf.usc.edu/~dingxiod/Papers/deng-kdd16.pdf) (**LSMRN**) using _Matrix Profile_ algortihm in with stumpy in Python .

## Some considerations
Here I try to adapt **LSMRN** to work with time series in general.

I assume the time series being a weighted (planar) path graph. Where the weights between two vertices is the time series value.

The idea is to use spatial information among the time series learned by another data-driven learning method as the _Matrix Profile_ or other correlation methods to define some kind of distances between the nodes.

To achieve that I use first the original k-hop similarity matrix and the _Matrix Profile_ algorithm.

## Code structure
The code has five main parts structured as follow:

1. Preprocess
   - load data
   - split in train and test
   - as negative values are not supported, we shift the train dataset.
   - split data in a list of snapshots, eg: [0:99], [100,199], ...
   
2. Adjacency
   - generate adjacency matrices G_1, G_2, ... from the snapshots.
   - create corresponding Y_t indication matrices. 
   

3. Proximity matrix and some constants
   - compute matrix profile for each snapshot and combine them in one general proximity matrix.
   - normalize the proximity matrix.
  
4. Global learning
   - U_t computation
   - B computation
   - A computation

5. Forecasting
   - see output adjacency matrix as planar graph
   - some metrics

6. plots