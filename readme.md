# Internal Pairwise Granger Causality Test for Time Series Data

This is an implementation of the internal pairwise granger causality test 
for time series data.  The purpose of this implementation is to be a fast 
base c++ implementation with minimal dependencies, and an implementation 
of the Granger causality test included.

The code does however have dependencies: 
+ The Eigen Library - For Linear Algebra (invert matrix)
+ The boost Library - For Cummulative Distribution Functions (T and F)

## Generate a sequence using genCorrelatedSequence.R 

I created an R script for generating a file with a correlated time series 
that can then be subjected to the internal pairwise granger causality testing.
