series_size <- 100000
correlated_lags <- 7
lag_effect_mean <- 0
lag_effect_var <- 1
series_init_mean <- 0
series_init_var <- 1
ran_mean <- 0
ran_var <- 0.1
outfilename <- "sfile.txt"
efffilename <- "efile.txt"

lag_effects <- rnorm(correlated_lags, lag_effect_mean, lag_effect_var)
lag_effects <- lag_effects/sum(abs(lag_effects))

series_init <- rnorm(correlated_lags, series_init_mean, series_init_var)

series <- series_init
for (i in (1 + correlated_lags):(series_size+1)) {
    nv <- sum(series[(i-correlated_lags):(i-1)]*lag_effects) 
    nv <- nv + rnorm(1, ran_mean, ran_var)
    series <- c(series,nv)
}

sink(file = outfilename)
cat(series)
sink(file = efffilename)
cat(lag_effects)
sink()