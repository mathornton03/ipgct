# Internal Pairwise Granger Causality Test for Time Series Data

This is an implementation of the internal pairwise granger causality test 
for time series data.  The purpose of this implementation is to be a fast 
base c++ implementation with minimal dependencies, and an implementation 
of the Granger causality test included.

The code does however have dependencies: 
+ The Eigen Library - For Linear Algebra (invert matrix)
+ The boost Library - For Cummulative Distribution Functions (T and F)

## Installation (Tested on Linux Only) 

After ensuring that you have install boost and eigen3 properly for your system, 
you can clone the ipgct repository using: 

```
git clone https://github.com/mathornton03/ipgct/
```

You can then build an executable version of the ipgct procedure using: 

```
g++ -I /usr/include/eigen3 gct.cpp -o ipgct.exe
```

where `/usr/include/eigen3' is replaced with your system path to eigen3.

Finally you can run the internal pairwise granger causality test using: 

```
./ipgct.exe INFILE OUTFILE SSSIZE MXLAG
```

Where, 
+ `INFILE' is the path to a whitespace delimited time-series file
+ `OUTFILE' is the path to a file for writing the granger_F_test_pvalue matrix to.
+ `SSSIZE' is the integral size of subsequences. 
+ `MXLAG' is the maximum number of lags to try in the testing.

## Generate a sequence using genCorrelatedSequence.R 

I created an R script for generating a file with a correlated time series 
that can then be subjected to the internal pairwise granger causality testing.
