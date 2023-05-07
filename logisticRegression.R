# Read CSV files
data <- read.csv("data.csv", header=FALSE)
responses <- read.csv("responses.csv", header=FALSE)
# Convert data frames to matrix and vector
data_matrix <- as.matrix(data)
responses_vector <- as.vector(t(responses))
# Perform logistic regression
logistic_model <- glm(responses_vector ~ data_matrix, family = binomial(link=logit))
# Print model summary
summary(logistic_model)
