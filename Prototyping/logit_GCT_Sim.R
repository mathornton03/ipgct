source("./Prototyping/logit_GCT.R")
library(rlist)
n <- 100 # The size of the time series 
p <- 0.5 # The proportion parameter (for uncorrelated series)

# Some example binary time series sequences. 
Y <- rbinom(n, 1, p)
X <- rbinom(n, 1, p)

# The maximum number of lags to try.
K <- 5
sig_ly <- 0.05 # significance level to keep lags of Y in model.
sig_lxy <- 0.05 # significance level to keep lags of X in model.

M <- 100
pvals <- c()
list.rbind(lapply(c(2,5,10,20), function(K){
list.rbind(lapply(c(0.1, 0.05, 0.01, 0.005), function(sig_lxy){
list.rbind(lapply(c(0.1, 0.05, 0.01, 0.005), function(sig_ly){
list.rbind(lapply(c(100,1000,10000), function(n){
    lapply(1:M, function(x){
        Y <- rbinom(n, 1, p)
        X <- rbinom(n, 1, p)
        gct_pval <- logit_gct(X, Y, K, sig_ly, sig_lxy)
        gct_pval
    }) -> pvals
    rejproj <- sum(pvals <= 0.05)/M
    c(K,sig_ly,sig_lxy,n,rejproj)
}))}))}))})) -> simres

colnames(simres) <- c("K", "sig_ly", "sig_lxy", "n", "prop_rejected")
write.table(simres, file = "simulation_results_1.tsv", sep = "\t")
