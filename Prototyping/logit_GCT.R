rm(list=ls())
# Set up some parameters

# First produce the proper lagged values of Y 
logit_gct <- function(X, Y, K, sig_ly, sig_lxy){
    stopifnot(K >= 1)

    lagged_y <- Y[1:(length(Y) - K)]
    for (i in 1:K){
        lagged_y <- cbind(lagged_y, Y[(i + 1):(length(Y) - (K - i))])
    }
    lydf <- data.frame(lagged_y)
    colnames(lydf)[1] <- "Y"

    # Now logistically regress Y on it's proper lagged values
    ly_mod <- glm(Y ~ ., data = lydf, family = binomial(link = "logit"))

    # Determine the names of which Y variables to keep in the model
    ly_keep_names <- names(which(coef(summary(ly_mod))[,4] <  sig_ly))
    if ("(Intercept)" %in% ly_keep_names){ ly_keep_names <- ly_keep_names[-which(ly_keep_names == "(Intercept)")]}
    nkeep <- length(ly_keep_names)

    # Form a dataset with only those variables, and the lagged values of X
    lagged_xy <- lydf[,c("Y",ly_keep_names)]
    for (i in 1:K){
        lagged_xy <- cbind(lagged_xy, X[(i + 1):(length(X) - (K - i))])
    }
    colnames(lagged_xy)[(2 + nkeep):ncol(lagged_xy)] <- paste("X", 1:K, sep = "")

    lxydf <- data.frame(lagged_xy)
    colnames(lxydf)[1] <- "Y"
    # logistically regress Y on the kept Y lags, and all lags of X.
    lxy_mod <- glm(Y ~ ., data = lxydf, family = binomial(link = "logit"))
    lxy_keep_names <- names(which(coef(summary(lxy_mod))[,4] < sig_lxy))
    if ("(Intercept)" %in% lxy_keep_names){ lxy_keep_names <- lxy_keep_names[-which(lxy_keep_names == "(Intercept)")]}

    if (length(ly_keep_names) != 0){
        null_form <- paste("Y ~ ", paste(ly_keep_names, collapse = "+"), sep = "")
        null_mod <- glm(null_form, data = lxydf, family = binomial(link = "logit"))
    } else {
        null_mod <- glm("Y ~ 1", data = lxydf, family = binomial(link = "logit"))
    }

    if (length(lxy_keep_names != 0)) {
        full_form <- paste("Y ~ ", paste(union(lxy_keep_names, ly_keep_names), 
            collapse = " + "), sep = "")
        full_mod <- glm(full_form, data = lxydf, family = binomial(link = "logit"))
    } else {
        full_mod <- glm("Y ~ 1", data = lxydf, family = binomial(link = "logit"))
    }
    anova(null_mod, full_mod, test="Chisq") -> lrt
    lrt$`Pr(>Chi)`[2] -> gct_pval
    if(is.na(gct_pval)){ gct_pval <- 1}
    return(gct_pval)
}
