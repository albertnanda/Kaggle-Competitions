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

#create sparse matrix
data_matrix=sparse.model.matrix(id+loss~.-1,data=raw_data)

#create xgb model

ddata <- xgb.DMatrix(data = data_matrix, label = raw_data$loss)
set.seed(1)
param=list("objective" = "reg:linear",
           "eval_metric" = "rmse",
           "eta"=.11,
           "max_depth"= 15,
           "subsample"=.5,
           "colsample_bytree"=.5,
           "min_child_weight"=1
           )

xgb=xgb.cv(params = param,data = ddata,nrounds = 100,nfold = 10)
  
  
xgb_pred=predict(xgb,test_matrix)
(RMSE_xgb=sqrt(mean((xgb_pred-raw_data$loss)^2)))





















