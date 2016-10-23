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
require(DMwR)
require(scales)
require(sigmoid)
require(Metrics)

#read the data set and convert the factors to numbers
setwd("d:/Kaggle Competition/")
raw_data=fread("train.csv",stringsAsFactors = T)
raw_data=data.frame(lapply(raw_data,as.numeric))
raw_data=data.table(raw_data)
raw_data[,id:=NULL]

#save the original variable
save_loss=raw_data$loss
save=raw_data

#Scale loss
raw_data$loss=log(raw_data$loss+1)

#try log+scale
raw_data=save
raw_data$loss=log(raw_data$loss+1)
max=max(raw_data$loss)
min=min(raw_data$loss)
raw_data$loss=rescale(raw_data$loss)

#create xgb model

dtrain <- xgb.DMatrix(data = as.matrix(raw_data[,-ncol(raw_data),with=F]), label = raw_data$loss)
set.seed(1)

param=list("objective" = "reg:logistic",
           "eta"=.075,
           "max_depth"= 6,
           "subsample"=.7,
           "colsample_bytree"=.7,
           "min_child_weight"=1
)

xg_eval_mae <- function (yhat, dtrain) {
  y = getinfo(dtrain, "label")
  err= mae(exp(y*(max-min)+min),exp(yhat*(max-min)+min))
  return (list(metric = "error", value = err))
}

xgb= xgb.cv(params=param,
             dtrain,
             nrounds=750,
             nfold=4,
             early_stopping_rounds=15,
             print_every_n = 10,
             verbose= 1,
             feval=xg_eval_mae,
             stratified = F,
             maximize=FALSE)

xgb=xgb.cv(params = param,data = ddata,nrounds = 100,nfold = 10,stratified = F)

xgb_final=xgb.train(params = param,data = ddata,nrounds = 750)
















#raw_data=save
mean=mean(save_loss)
sd=sd(save_loss)
lambda=2
raw_data$loss=save_loss
head(raw_data$loss)
raw_data=save



#remove outliers and scale
raw_data=raw_data[loss>=100,]
dim(raw_data)

p=max(raw_data$loss)-min(raw_data$loss)
q=min(raw_data$loss)
raw_data$loss=rescale(raw_data$loss)
max(raw_data$loss)
#try softmax transformation
raw_data$loss=sigmoid(raw_data$loss,method = "logistic")
raw_data$loss=sigmoid(raw_data$loss,method = "logistic",inverse = T)

lambda=1
raw_data$loss=SoftMax(raw_data$loss,lambda = 2,avg=mean,std=sd)

min(raw_data$loss)
max(raw_data$loss)
#try log + rescaling
raw_data$loss=log(raw_data$loss)
(q=min(raw_data$loss))
(p=max(raw_data$loss))
raw_data$loss=rescale(raw_data$loss)
#try log+softmax
raw_data$loss=log(raw_data$loss)
mean=mean(raw_data$loss)
sd=sd(raw_data$loss)
lambda=2
raw_data$loss=SoftMax(raw_data$loss,lambda = lambda,avg=mean,std=sd)
max(raw_data$loss)
#try log
raw_data$loss=log(raw_data$loss+1)
#create sparse matrix
data_matrix=as.matrix(raw_data[,-c(1,ncol(raw_data)),with=F])

#create xgb model

ddata <- xgb.DMatrix(data = data_matrix, label = raw_data$loss)
set.seed(1)

param=list("objective" = "reg:logistic",
           "eval_metric" = "rmse",
           "eta"=.075,
           "max_depth"= 6,
           "subsample"=.7,
           "colsample_bytree"=.7,
           "min_child_weight"=1
           )



xgb=xgb.cv(params = param,data = ddata,nrounds = 100,nfold = 10,stratified = F)

xgb_final=xgb.train(params = param,data = ddata,nrounds = 750)
test=fread("test.csv",stringsAsFactors = T)
test=as.data.frame(sapply(test, as.numeric))
test=data.table(test)
#create sparse matrix
test_matrix=as.matrix(test[,-1,with=F])


test_xgb <- xgb.DMatrix(data = test_matrix)




xgb_pred=predict(xgb_final,test_xgb)

#scale the variables back to the original scale, p&q should be pre-computed
xgb_pred=(xgb_pred*p)+q
#convert the variables back to the original scale inverse softmax, make sure mean and sd variables are loaded
xgb_pred=inverse_softmax(xgb_pred)
xgb_pred=exp(xgb_pred)
#scale it back and do the exp transformation
xgb_pred=(xgb_pred*(p-q))+q
xgb_pred=exp(xgb_pred)

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
write.csv(test,"atmp5.csv",row.names = F)
max(test$loss)











#Softmax Inverse Function
inverse_softmax=function(x)
{
  result=mean-((log((1-x)/x)*lambda*sd)/(2*pi))
}

packageVersion("xgboost")

