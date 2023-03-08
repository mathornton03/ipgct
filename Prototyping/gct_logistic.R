rm(list = ls())

X <- rbinom(100, 1, 0.5)
Y <- rbinom(100, 1, 0.5)
k <- 5

lagged_xy <- Y[1:(length(Y) - (k - 1))]
for (i in (2:k)){
    lagged_xy <- cbind(lagged_xy, Y[i:(length(Y) - (k - i))])
    lagged_xy <- cbind(lagged_xy, X[i:(length(X) - (k - i))])
}
lagged_xy <- data.frame(lagged_xy)

lagged_y <- Y[1:(length(Y) - (k - 1))]
for (i in (2:k)){
    lagged_y <- cbind(lagged_y, Y[i:(length(Y) - (k - i))])
}
lagged_y <- data.frame(lagged_y)

colnames(lagged_xy)[1] <- "Y"
colnames(lagged_y)[1] <- "Y"

full <- glm(Y ~ ., data = lagged_xy, family = binomial(link = "logit"))
null <- glm(Y ~ ., data = lagged_y, family = binomial(link = "logit"))

ll_full <- logLik(full)
ll_null <- logLik(null)

ts <- -2 * (ll_full - ll_null)
pchisq(ts, df = k-1, lower.tail = FALSE)

lagged_x <- X[2:(length(X) - (k - 2))]
for (i in (3:k)){
    lagged_x <- cbind(lagged_x, X[i:(length(X) - (k - i))])
}
rnorm(k-1) -> beta
ps <- apply(lagged_x, 1, function(x) { 1/(1+exp(-sum(x * beta))) })


# Generate binary sequence Y such that X Granger-causes Y
p <- 0.8
Y <- rep(0, 100)
Y[1] <- X[1]
for (i in 2:100) {
  Y[i] <- rbinom(1, 1, ps[i-1])
}


