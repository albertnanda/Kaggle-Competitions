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
train=fread("train.csv")
test=fread("test.csv")
# save id & loss
final=data.table(id=test$id)
train_Y=train$loss

#remove id and loss
train[,c("id","loss"):=NULL]
test[,c("id"):=NULL]

cat_vindx=c("cat101","cat105","cat109","cat110","cat111","cat113","cat114","cat116")


data_comb=rbind(train,test)


vec=c(26,1)
convert2code=function(x)
{
  temp=utf8ToInt(x)-64
  if(length(temp)==1) {
    length(temp)=2 
    temp=rev(temp)
    }
  return(sum(temp*vec,na.rm = T))
}

for(i in cat_vindx)
{
  data_comb[[i]]=sapply(data_comb[[i]],convert2code)
  data_comb[[i]]=log(200+data_comb[[i]])
}
  


indx=names(data_comb)
indx=setdiff(indx,cat_vindx)
for(i in indx)
{
  if(length(grep("cat",i))>0)
  {
    data_comb[[i]]=as.integer(as.factor(data_comb[[i]]))
    data_comb[[i]]=log(200+data_comb[[i]])
  } else
  {
    data_comb[[i]]=log(10^3*data_comb[[i]]+200)
  }
}

#simple log
train_Y=log(200+train_Y)

#create xgb model
dtrain <- xgb.DMatrix(data = as.matrix(data_comb[1:length(train_Y),]),label = train_Y)

dtest <- xgb.DMatrix(data = as.matrix(data_comb[(length(train_Y)+1):nrow(data_comb),]))
  
watchlist=list(train=dtrain)


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
            #watchlist=watchlist,
            maximize=FALSE
)

set.seed(0)
xgb_final=xgb.train(params = param,data = dtrain,watchlist=watchlist,
                    feval=xg_eval_mae,nrounds = 5000)



#predict
xgb_pred=predict(xgb_final,dtest)
#rescale the variable and anti-log
xgb_pred=exp(xgb_pred)-200
head(xgb_pred)
max(xgb_pred)
min(xgb_pred)
mean(xgb_pred)
sd(xgb_pred)


final$loss=xgb_pred
head(final)

#write file to disk
write.csv(final,"14.csv",row.names = F)


