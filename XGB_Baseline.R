#Kaggle Competition
#clean memory
rm(list = ls())
gc()
#Load the libraries
require(caret)
require(Matrix)
require(xgboost)
require(data.table)
require(Metrics)
require(Cubist)
library(earth)

#read the data set and convert the factors to numbers
setwd("d:/Kaggle Competition/")
train=fread("train.csv",stringsAsFactors = T)
train=data.frame(lapply(train,as.numeric))
train=data.table(train)

#save the original variable
save_loss=train$loss
save=train
train=save
head(train$loss)

#create sparse matrix
#train_matrix=sparse.model.matrix(loss~.-1,data = train)

train=train[,c("cont12"):=NULL]
indx=grep("cont",names(train))
for(j in indx)
  set(train,j=j,value=log1p(train[[j]]*10^6))


#simple log
train=save
train$loss=log(200+train$loss)
names(train)

#create xgb model
dtrain <- xgb.DMatrix(data = as.matrix(train[,-c(grep("id",names(train)),
                                                grep("loss",names(train))),
                                                with=F]), 
                                                label = train$loss)

set.seed(0)

logregobj <- function(preds, dtrain){
  labels = getinfo(dtrain, "label")
  con = 2
  x = preds-labels
  grad =con*x / (abs(x)+con)
  hess =con^2 / (abs(x)+con)^2
  return (list(grad = grad, hess = hess))
}

param=list(objective = logregobj,
           eta=.005, 
           max_depth= 12,
           subsample=.8,
           colsample_bytree=.5,
           min_child_weight=1,
           base_score=7,
           alpha=1,
           gamma=1
)

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y),exp(yhat))
  return (list(metric = "error", value = err))
}


xgb= xgb.cv(params=param,
             dtrain,
             nrounds=2000,
             nfold=5,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1,
             feval=xg_eval_mae,
             maximize=FALSE
            )


xgb_final=xgb.train(params = param,data = dtrain,nrounds = 5000)



#read the test data set
test=fread("test.csv",stringsAsFactors = T)
test=data.frame(lapply(test,as.numeric))
test=data.table(test)

test=test[,c("cont12"):=NULL]
indx=grep("cont",names(test))
for(j in indx)
  set(test,j=j,value=log1p(test[[j]]*10^6))

#create  matrix
dtest <- xgb.DMatrix(data = as.matrix(test[,-1,with=F])) 
                      

#predict
xgb_pred=predict(xgb_final,dtest)
#rescale the variable and anti-log
xgb_pred=exp(xgb_pred)
xgb_pred=xgb_pred-200
head(xgb_pred)
max(xgb_pred)
min(xgb_pred)
mean(xgb_pred)
sd(xgb_pred)


test$loss=xgb_pred

ggplot(test,aes(loss))+geom_histogram(binwidth = 0.5)


#write file to disk
test=test[,c(1,ncol(test)),with=F]
test$id=as.integer(test$id)
head(test)
write.csv(test,"new3.csv",row.names = F)
max(test$loss)
sum(test$loss)
