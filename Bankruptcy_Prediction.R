library(ggplot2)
library(knitr)
library(dplyr)
library(tidyr)
library(caret)
library(corrplot)
library(e1071)
library(nortest)


data <- read.csv("taiwanese_bankruptcy_prediction.csv")
summary(data)

#sample size number of predectors and responce
dim(data)

#responce vairable Y
hist(data$Bankrupt)

#our data doesnt have any null values
data %>% is.na() %>% colSums()
total_data <- data
##################
#categorical varieble
histogram(total_data$Liability.Assets.Flag, main = 'Percentage distribution of flags in Liability.Assets.Flag')
histogram(total_data$Net.Income.Flag, main = 'Percentage distribution of flags in Net.Income.Flag')

zeroVar_cat = nearZeroVar(total_data)
zeroVar_cat # remove all the caegorical variables.
##############
y <- total_data$Bankrupt
data <- data[, !colnames(data) %in% c("Bankrupt","Liability.Assets.Flag", "Net.Income.Flag")]

par(mfrow = c(6,6))
for (i in colnames(data[, 1:36])){
  hist(data[[i]], main = i , xlab = i)
}

par(mfrow = c(6,6))
for (i in colnames(data[, 36:71])){
  hist(data[[i]], main = i , xlab = i)
}

par(mfrow = c(6,6))
for (i in colnames(data[, 72:93])){
  hist(data[[i]], main = i , xlab = i)
}

##############

skewValues <- apply(data, 2, skewness)
skew_matrix <- matrix(skewValues, nrow = 93, ncol = 1, byrow = TRUE)
rownames(skew_matrix) <- names(data)
kable(skew_matrix, format = "pipe", col.names = 'skewness')

###################################################################################
library(MASS)    # For generating multivariate data
library(ggplot2)  # For plotting

#applying transformation boxcox -venkat 
transformed_data <- lapply(names(data), function(x) {
  bxcx = BoxCoxTrans(data[,x])
  predict(bxcx, data[,x])
})

transformed_data <- data.frame(sapply(transformed_data,c))
colnames(transformed_data) <- names(data)

skewValues_trans <- apply(transformed_data, 2, skewness)
skew_matrix_trans <- matrix(skewValues_trans, nrow = 93, ncol = 1, byrow = TRUE)
rownames(skew_matrix_trans) <- names(transformed_data)
kable(skew_matrix_trans, format = "pipe", col.names = 'skewness')


#spatial sign transformation
# Perform spatial sign transformation
transformed_data <- as.data.frame.matrix(spatialSign(transformed_data))

skewValues <- apply(transformed_data, 2, skewness)
skew_matrix <- matrix(skewValues, nrow = 93, ncol = 1, byrow = TRUE)
rownames(skew_matrix) <- names(data)
kable(skew_matrix, format = "pipe", col.names = 'skewness')

# Create a data frame for plotting
original_df <- as.data.frame(data)
original_df_name <- names(data)
transformed_df <- as.data.frame(transformed_data)

par(mfrow = c(4,2))
lapply(1:length(original_df_name), function(i) {
  col_name <- original_df_name[[i]]
  boxplot(original_df[, i], main=col_name, xlab=col_name)
  mlabel <- paste('after transformation of',col_name)
  boxplot(transformed_df[, i], main=mlabel, xlab=col_name)
})

#performing PCA after removing correlated data
trans = preProcess(data, method = c("BoxCox", "center", "scale", "pca"))
trans

# Apply the transformations:
transformed <- predict(trans, data)  
dim(data) #before
dim(transformed) #after
str(transformed)


###############
XX <- transformed
XX
sigma=var(XX) ## variance-covariance matrix
sigma
QQ=eigen(sigma) ## calculate eigen values and eigen vectors of sigma
lamda=QQ$values 
AA=QQ$vectors 
##first colume of AA is the first eigen vector corresponding to the first eigen valus.
PCs=as.matrix(XX)%*%as.matrix(AA)

PCs

propotion=cumsum(lamda)/sum(lamda)
propotion
par(mfrow = c(1,1))
plot(propotion)
abline(h = 0.95, col = "red")
abline(v = 45, col = "red")
abline(h = 0.90, col = "blue")
abline(v = 40, col = "blue")
abline(h = 0.99, col = "green")
abline(v = 51, col = "green")

###############

#corelation plot before
par(mfrow = c(1,1))
cor_pred = cor(XX)
corrplot(cor_pred)# order = 'hclust')

#removing highly Correlation data
#highCorr <- findCorrelation(cor_pred, cutoff = .85)
#length(highCorr)
#filteredSegData <- data[, -highCorr]
#length(filteredSegData)
#filtered_cor_pred = cor(filteredSegData)
#corrplot(cor(filtered_cor_pred), order = 'hclust')

#########################
# model building
set.seed(5790)
index <- createDataPartition(y, p = 0.7, list = FALSE)


xtrain = XX[index,]
ytrain = factor(y[index], labels = c("zero", "one"))
xtest = XX[-index,]
ytest = factor(y[-index], labels = c("zero", "one"))

#####################
#models

#venkat
# 1. Logistic Reg.
# 2. Linear Disc. Analysis
# 3. PLS
# 4. pinealized models
# 5. Nearest Shrunken centriod

#Rayal
# 1. Non-Linear Disc.
# 2. Neural Net.
# 3. Flexible Disc
# 4. SVM
# 5. KNN
# 6. Naive Bayes

ctrl <- trainControl(method = "cv", 
                     number = 10,
                     classProbs = TRUE,
                     summaryFunction = defaultSummary,
                     savePredictions = TRUE)

#venkat
# 1. Logistic Reg.
lrFull <- train(xtrain,
                y = ytrain,
                method = "glm",
                metric = "ROC",
                trControl = ctrl)
lrFull


lr_pred <- predict(lrFull, xtest)
confusionMatrix(lr_pred,ytest)

lr_pred <- predict.glm(lrFull$finalModel, xtest, type = "response")
lr_prob <- predict(lrFull, xtest, type = "prob")
lr_FullRoc <- roc(response = ytest,
                  predictor = lr_prob[, 2],
                  levels = rev(levels(ytest)))
plot(lr_FullRoc, legacy.axes = TRUE)
auc(lr_FullRoc)

# 2. Linear Disc. Analysis
LDAFull <- train(xtrain,
                 y = ytrain,
                 method = "lda",
                 metric = "ROC",
                 trControl = ctrl)
LDAFull

plot(LDAFull) #no tuning parameters for this model.

lda_pred <- predict(LDAFull, xtest)
confusionMatrix(data =lda_pred,reference=as.factor(ytest))

ROC_pred <- predict(LDAFull$finalModel, xtest, type = "response")
ROC_prob <- predict(LDAFull, xtest, type = "prob")
ROC_FullRoc <- roc(response = ytest,
                  predictor = ROC_prob[, 2],
                  levels = rev(levels(ytest)))
auc(ROC_FullRoc)

# 3. PLS
plsFit2 <- train(xtrain,
                 y = ytrain,
                 method = "pls",
                 tuneGrid = expand.grid(.ncomp = 1:4),
                 preProc = c("center","scale"),
                 metric = "ROC",
                 trControl = ctrl)

plsFit2

plot(plsFit2) #no tuning parameters for this model.
pls_pred <- predict(plsFit2, xtest)
confusionMatrix(data =pls_pred,reference=as.factor(ytest))

# 4. pinealized models
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(476)
glmnTuned <- train(xtrain,
                   y = ytrain,
                   method = "glmnet",
                   tuneGrid = glmnGrid,
                   preProc = c("center", "scale"),
                   metric = "ROC",
                   trControl = ctrl)
glmnTuned

plot(glmnTuned) #no tuning parameters for this model.
glmn_pred <- predict(glmnTuned, xtest)
confusionMatrix(data =glmn_pred,reference=as.factor(ytest))

# 5. Nearest Shrunken centriod

nscGrid <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(476)
nscTuned <- train(xtrain,
                  y = ytrain,
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = nscGrid,
                  metric = "ROC",
                  trControl = ctrl)

nscTuned
plot(nscTuned)
nsc_pred <- predict(nscTuned, xtest)
confusionMatrix(data =nsc_pred,reference=as.factor(ytest))

##1. Non-Linear Disc.
mdaFit <- train(x = xtrain,
                y = ytrain,
                method = "mda",
                metric = "kappa",
                tuneGrid = expand.grid(.subclasses = 1:10),
                trControl = ctrl)
mdaFit
plot(mdaFit)

mdaPred <- predict(mdaFit, xtest)

postResample(mdaPred, ytest)

confusionMatrix(mdaPred, ytest)

###not working
##2. Neural Network 
library(nnet)
library(caret)
## Using the formula interface to fit a simple model:
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))
maxSize <- max(nnetGrid$.size)
numWts <- (maxSize * (52 + 1) + (maxSize+1)*2) ## 4 is the number of predictors use when 

nnetFit <- train(x = xtrain,
                 y = ytrain,
                 method = "nnet",
                 metric = "kappa",
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit
plot(nnetFit)

nnetPred <- predict(nnetFit, xtest)

postResample(nnetPred, ytest)

confusionMatrix(nnetPred, ytest)

##3. Flexible Disc
marsGrid <- expand.grid(degree = 1:3, nprune = 2:10)

fdaModel <- train(x = xtrain,
                  y = ytrain,
                  method = "fda",
                  metric = "kappa",
                  tuneGrid = marsGrid,
                  trControl = ctrl)

fdaModel
plot(fdaModel)

fdaPred <- predict(fdaModel, xtest)

postResample(fdaPred, ytest)

confusionMatrix(data = fdaPred, reference=ytest)


##4 SVM
library(kernlab)
library(caret)
sigmaRangeReduced <- sigest(as.matrix(xtrain))
svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 6)))
svmRModel <- train(x = xtrain, 
                   y = ytrain,
                   method = "svmRadial",
                   metric = "kappa",
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel
plot(svmRModel)

svmPred <- predict(svmRModel, xtest)

postResample(svmPred, ytest)

confusionMatrix(svmPred, ytest)


##5 KNN
#ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)
#library(caret)

knnFit <- train(x = xtrain, 
                y = ytrain,
                method = "knn",
                metric = "kappa",
                tuneGrid = data.frame(.k = 1:50),
                trControl = ctrl)

knnFit
plot(knnFit)

knnPred <- predict(knnFit, xtest)

postResample(knnPred, ytest)

confusionMatrix(knnPred, ytest)


##6 Naive Bayes
library(klaR)
nbFit <- train( x = xtrain, 
                y = ytrain,
                method = "nb",
                metric = "kappa",
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)

nbFit
#plot(nbFit)

nbPred <- predict(nbFit, xtest)

postResample(nbPred, ytest)

confusionMatrix(nbPred, ytest)
