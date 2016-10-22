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

#save the original variable
save_loss=raw_data$loss


#log transform the variable
raw_data$loss=log(raw_data$loss+mean(raw_data$loss))



#create sparse matrix
data_matrix=sparse.model.matrix(id+loss~.-1,data=raw_data)

#create xgb model

ddata <- xgb.DMatrix(data = data_matrix, label = raw_data$loss)
set.seed(1)
param=list("objective" = "reg:linear",
           "eval_metric" = "rmse",
           "eta"=.2,
           "max_depth"= 8,
           "subsample"=.4,
           "colsample_bytree"=.3,
           "min_child_weight"=2
           )

xgb=xgb.cv(params = param,data = ddata,nrounds = 30,nfold = 10)



xgb_final=xgb.train(params = param,data = ddata,nrounds = 30)



test=fread("test.csv",stringsAsFactors = T)
test=data.table(test)
#create sparse matrix
test_matrix=sparse.model.matrix(id~.-1,data=test)

test_xgb <- xgb.DMatrix(data = test_matrix)




xgb_pred=predict(xgb_final,test_xgb)
head(xgb_pred)
exp(max(xgb_pred))
xgb_pred=exp(xgb_pred)

test$loss=xgb_pred

test=test[,c(1,ncol(test)),with=F]
write.csv(test,"logtest2.csv")
max(test$loss)











