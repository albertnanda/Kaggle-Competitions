#Kaggle Competition
#clean memory
rm(list = ls())
gc(reset = T)
#Load the libraries
require(caret)
require(Matrix)
require(xgboost)
require(data.table)
require(caret)
require(snow) 
require(coop)
require(scales)
require(Metrics)

#read the data set and convert the factors to numbers
setwd("d:/Kaggle Competition/")
train=fread("train_final.csv")
test=fread("test_final.csv")
train_Y=fread("./mcmc/trn_id.csv")
oob_id=train_Y$id
train_Y=train_Y$loss


indx=grep("cont",names(train_test),value = T)
for(i in indx)
{
  train_test[[i]]=log(train_test[[i]]*10^6+200)
  
}
gc(verbose = F)

#head(train_test$fecat)
#simple log
shift=200
train_Y=log(train_Y+shift)

#create xgb model
dtest <- xgb.DMatrix(data = as.matrix(test))
remove(test)
gc()
logregobj <- function(preds, dtrain){
  labels = getinfo(dtrain, "label")
  con = .7
  x = preds-labels
  grad =con*x / (abs(x)+con)
  hess =con^2 / (abs(x)+con)^2
  return (list(grad = grad, hess = hess))
}


param=list(objective = logregobj,
           eta=.05, 
           max_depth= 15,
           subsample=.6, #.7
           colsample_bytree=.9, #.8
           min_child_weight=100, #100,90
           base_score=7.76
)


xg_eval_mae <- function (yhat, dtrain) {
  y = as.numeric(getinfo(dtrain, "label"))
  err= as.numeric(mae(exp(y),exp(yhat)))
  return (list(metric = "mae", value = round(err,4)))
}
gc(verbose = F)
set.seed(0)
#create fold
nfolds=5
folds=createFolds(train_Y,k=nfolds,list = T,returnTrain = T)
prediction=numeric(nrow(dtest))



oob=data.frame(id=NULL,real=NULL,pred=NULL)
gc()
for(i in 1:length(folds))
{
  i=1
  cat("starting Fold",i,"\n")
  X_train=train[folds[[i]],]
  Y_train=train_Y[folds[[i]]]
  X_val=train[-folds[[i]],]
  Y_val=train_Y[-folds[[i]]]
  id_val=oob_id[-folds[[i]]]
  dtrain=xgb.DMatrix(data = as.matrix(X_train),label=Y_train)
  dtrain2=xgb.DMatrix(data = as.matrix(X_val),label=Y_val)
  watchlist=list(train=dtrain,test=dtrain2)
  model=xgb.train(params = param,data = dtrain,watchlist=watchlist,
                  early_stopping_rounds = 300,
                  feval=xg_eval_mae,print_every_n = 50,nrounds = 6000,maximize=FALSE)
  
  pred=predict(model,dtest)
  prediction=prediction+exp(pred)-shift
  dval=xgb.DMatrix(as.matrix(X_val))
  pred=exp(predict(model,dval))-shift
  oob=rbind(oob,cbind(id=id_val,real=exp(Y_val)-shift,pred=pred))
  gc(verbose = F)
}

#final mae
print(mae(oob$real,oob$pred))
prediction=prediction/nfolds
final$loss=prediction

#write file to disk
write.csv(final,"xgb_fe_2.csv",row.names = F)

#write oob to disk if you want
write.csv(oob,"oob_xgb_2.csv",row.names=F)
