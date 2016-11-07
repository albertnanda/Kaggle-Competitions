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

train_B=train[cat57==2,]
plot(train_B)
slm=lm(log(loss+100)~.-id,data=train_B)
par(mfrow=c(2,2))
y=knn.reg
plot(slm)


lmTune <- train(loss~.-id,data=train_B,
                 method = "lm",
                 trControl = trainControl(method = "cv",n=4))

lmTune
knnTune <- train(loss~.-id,data=train_B,
                 method = "knn",
                 tuneGrid = data.frame(.k = 1:2),
                 trControl = trainControl(method = "cv",n=4))


knnTune
plot(sqrt(train_B$loss))
#create sparse matrix
#train_matrix=sparse.model.matrix(loss~.-1,data = train)

train=train[,c("cont12"):=NULL]

#simple log
train=save
train$loss=log(train$loss)
names(train)
train=train_B
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
           eta=.01, 
           max_depth= 10,
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
