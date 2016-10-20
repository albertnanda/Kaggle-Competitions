#Kaggle Competition
#clean memory
rm(list = ls())
gc()
#Load the libraries
require(caret)
require(corrplot)
require(car)
require(glmnet)
#require(RWeka)
require(earth)
require(h2o)
require(readr)
require(stringr)
require(Matrix)
require(xgboost)
require(data.table)
require(Cubist)

#read the data set
setwd("d:/Kaggle Competition/")
raw_data=fread("train.csv",stringsAsFactors = T)
raw_data=data.table(raw_data)

#remove the outliers
raw_data$flag=ifelse(raw_data$loss>=8000,1,0)
#splitting train and test data for grouped categories
set.seed(1)
index=createDataPartition(raw_data$loss,p=.8,list = F)
#index = sort(sample(nrow(raw_data), nrow(raw_data)*.8))
train=raw_data[index,]
test=raw_data[-index,]

#create sparse matrix
train_matrix <- sparse.model.matrix(id+loss+flag~.-1, data = train)
test_matrix <- sparse.model.matrix(id+loss+flag~.-1,data = test)
#create xgb model

dtrain <- xgb.DMatrix(data = train_matrix, label = train$flag)
dtest <- xgb.DMatrix(data =test_matrix, label = test$flag)
watchlist <- list(train=dtrain, test=dtest)
set.seed(1)
param=list("objective" = "binary:logistic",
           "eval_metric" = "error",
           "eta"=.11,
           "max_depth"= 15,
           "subsample"=.5,
           "colsample_bytree"=.5,
           "min_child_weight"=1,
           "watchlist"=watchlist
           )
xgb=xgboost(param=param,data = dtrain,nrounds = 100)

          
xgb_pred=predict(xgb,test_matrix)
pred=ifelse(xgb_pred>=.4,1,0)
head(pred)
ct=table(pred=pred,obs=test$flag)
ct
(recall=ct[2,2]/sum(ct[,2]))
(prec=ct[2,2])/sum(ct[2,])
(RMSE_xgb=sqrt(mean((xgb_pred-test$loss)^2)))

#regression model
train_0=train[flag==0,]
train_1=train[flag==1,]
train_1=train_1[,-133,with=F]

dim(train_1)
##create test data set
test_0=test[pred==0,-133,with=F]
test_1=test[pred==1,-133,with=F]
dim(test_1)
#train model with Flag==0

#create sparse matrix
train_matrix <- sparse.model.matrix(id+loss~.-1, data = train_0)
test_matrix <- sparse.model.matrix(id+loss~.-1,data = test_0)
#create xgb model

dtrain <- xgb.DMatrix(data = train_matrix, label = train_0$loss)
dtest <- xgb.DMatrix(data =test_matrix, label = test_0$loss)
watchlist <- list(train=dtrain, test=dtest)
set.seed(1)
param=list("objective" = "reg:linear",
           "eval_metric" = "rmse",
           "eta"=.11,
           "max_depth"= 15,
           "subsample"=.5,
           "colsample_bytree"=.5,
           "min_child_weight"=1,
           "watchlist"=watchlist
)
xgb=xgboost(param=param,data = dtrain,nrounds = 100)
xgb_pred=predict(xgb,test_matrix)
(RMSE_xgb=sqrt(mean((xgb_pred-test_0$loss)^2)))

#running 0 data

#train model with Flag==0

#create sparse matrix
train_matrix <- sparse.model.matrix(id+loss~.-1, data = train_1)
test_matrix <- sparse.model.matrix(id+loss~.-1,data = test_1)
#create xgb model

dtrain <- xgb.DMatrix(data = train_matrix, label = train_1$loss)
dtest <- xgb.DMatrix(data =test_matrix, label = test_1$loss)
watchlist <- list(train=dtrain, test=dtest)
set.seed(1)
param=list("objective" = "reg:linear",
           "eval_metric" = "rmse",
           "eta"=.11,
           "max_depth"= 15,
           "subsample"=.5,
           "colsample_bytree"=.5,
           "min_child_weight"=1,
           "watchlist"=watchlist
)
xgb=xgboost(param=param,data = dtrain,nrounds = 100)
xgb_pred=predict(xgb,test_matrix)
(RMSE_xgb=sqrt(mean((xgb_pred-test_1$loss)^2)))

names=train_matrix@Dimnames[[2]]

#weighted average error
w_rmse=(5131*nrow(test_1)+1740*nrow(test_0))/(nrow(test_0)+nrow(test_1))

w_rmse
#plot the feature importance matrix
names=train_matrix@Dimnames[[2]]
importance_matrix = xgb.importance(names, model=xgb)
head(importance_matrix,n=50) 

pp=mean(train_1$loss)
(sqrt(mean((pp-test_1$loss)^2)))
#parameter Tuning

xgb.grid <- expand.grid(nrounds = 100,
                        eta = .01,
                        max_depth = 10,
                        gamma=1,
                        colsample_bytree=.5,
                        min_child_weight=1)
set.seed(1)
trCtrl=trainControl(method = "cv")
xgbTune=train(loss~.-id,data=raw_data_extreme,method="xgbTree",
              trControl=trCtrl,
              subsample=.5,
              tuneGrid=xgb.grid,
              booster="gbtree",
              objective = "reg:linear",
              eval_metric="rmse"
    )
xgbTune


#graphics in r

