gct_logistic <- function(X,Y,k = 5){
  data_xy <- Y[1:(length(Y)-(k-1))]
  for (i in 2:k){
    data_xy <- cbind(data_xy,Y[i:(length(Y)-(k-i))])
    data_xy <- cbind(data_xy,X[i:(length(X)-(k-i))])
  }

  data_y <- Y[1:(length(Y)-(k-1))]
  for (i in 2:k){
    data_y <- cbind(data_y, Y[i:(length(Y)-(k-i))])
  }

  colnames(data_xy)[1] <- "Y"
  data_xy <- data.frame(data_xy)

  glm(Y~.,data=data,family=binomial(link="logit")) -> full_fit
  full_llik <- sum(log(1+exp(predict(full_fit))))

  colnames(data_y)[1] <- "Y"
  data_y <- data.frame(data_y)

  glm(Y~., data=data_y,family=binomial(link="logit")) -> null_fit
  null_llik <- sum(log(1+exp(predict(null_fit))))

  lr_stat <- -2 * (null_llik - full_llik)
  pval <- pchisq(lr_stat,k,lower.tail=FALSE)
  return(pval)
}
set.seed(123)
X <- rbinom(1000,1,0.5)
Y <- rbinom(1000,1,0.5)

gct_logistic(X, Y, 10) -> pval_random

set.seed(123)
n <- 1000
p <- 0.8

# Generate binary sequence X
X <- rbinom(n, 1, 0.5)

# Generate binary sequence Y that is Granger caused by X
Y <- rep(0, n)
Y[1] <- X[1]
for (i in 2:n) {
  Y[i] <- rbinom(1, 1, p * X[i-1] + (1-p) * Y[i-1])
}

gct_logistic(X, Y, 10) -> pval_caused