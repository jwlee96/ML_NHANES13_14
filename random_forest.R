setwd("/Users/juntianwang/Desktop/Machine Learning")
library(randomForest)
library(caret)
library(ROCR)
X_test <- read.csv("X_test.csv")
X_test2 <- X_test[,-1]
X_train <- read.csv("X_train.csv")
X_train2 <- X_train[,-1]
y_test <- read.csv("y_test.csv")
y_test2 <- y_test[,-1]
y_test2 <- as.factor(y_test2)
y_train <- read.csv("y_train.csv")
y_train2 <- y_train[,-1]
y_train2 <- as.factor(y_train2)

y_train$y_val <- as.factor(y_train$y_val)
data.test<-data.frame(y_test2, X_test2)
data.train<-data.frame(y_train2, X_train2)

#null model
model <- randomForest(y_train2 ~., data=data.train )
model
y_pred <- predict(model, newdata = X_test2 ,type="prob")

ROCpred <- prediction(as.numeric(y_pred[,2]), y_test2)
auc_ROCR <- performance(ROCpred, measure = "auc")
# AUC = 0.91
auc_ROCR@y.values[[1]] 
roc_perf <- performance(ROCpred , "tpr" , "fpr")
# ROC plot
plot(roc_perf,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))
cost_perf = performance(ROCpred, "cost") 
# cutoff = 0.532
cutoff <- ROCpred@cutoffs[[1]][which.min(cost_perf@y.values[[1]])]

y_predfin = ifelse(y_pred[,2]>cutoff,1,0)
# accuracy = 0.84
mean(y_predfin==y_test2) 
confusionMatrix(y_test2,as.factor(y_predfin))



control <- trainControl(method="cv", number=10)

# control <- trainControl(method="repeatedcv", number=10, repeats=3)

# Best mtry = 24
bestmtry <- tuneRF(X_test2, y_test2, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)

# Fit random forest model with mtry = 24
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- 24
tunegrid <- expand.grid(.mtry=mtry)
# Running time = 16min
start.time <- Sys.time()
rf_tunning <- train(y_train2~., data=data.train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)
time.taken

y_pred <- predict(rf_tunning, newdata = X_test2 ,type="prob")

ROCpred <- prediction(as.numeric(y_pred[,2]), y_test2)
auc_ROCR <- performance(ROCpred, measure = "auc")
auc_ROCR@y.values[[1]] #best cutoff
roc_perf <- performance(ROCpred , "tpr" , "fpr")
plot(roc_perf,
     colorize = TRUE,
     print.cutoffs.at= seq(0,1,0.05),
     text.adj=c(-0.2,1.7))

cost_perf = performance(ROCpred, "cost") 

cutoff <- ROCpred@cutoffs[[1]][which.min(cost_perf@y.values[[1]])]

y_predfin = ifelse(y_pred[,2]>cutoff,1,0)

mean(y_predfin==y_test2) #accuracy

confusionMatrix(y_test2,as.factor(y_predfin))


install.packages("devtools")
library(devtools)
devtools::install_github('araastat/reprtree')
library(reprtree)
model <- randomForest(y_test2~., data=data.test, importance=TRUE, ntree=500, mtry = 16, do.trace=100)
reprtree:::plot.getTree(model)
