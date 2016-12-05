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
train=fread("train.csv")
test=fread("test.csv")
final=data.frame(id=test$id)
train=train[(loss<=250 | loss>=20000) ,id] #250 & 19000 first fold 1112.14 best is 20000
xx=data.frame(id=train,loss=NA)
head(xx)
write.csv(xx,"natrain.csv",row.names = F)
head(train)
nrow(train)
train_bac=copy(train)

188318-nrow(train)
oob_id=train$id
train_Y=train$loss

train[,c("id","loss"):=NULL]
test[,c("id"):=NULL]

train_test=rbind(train,test)

indx=grep("cat",names(train_test),value=T)
for(i in indx)
{
  temp=train_bac[,list(mean=mean(loss)),by=i]
  temp=temp[order(mean)]
  temp$rank=as.integer(factor(temp$mean,levels = unique(temp$mean)))
  train_test[[i]]=temp$rank[match(train_test[[i]],temp[[i]])]
  train_test[[i]][is.na(train_test[[i]])]<-10
}

remove(train_bac,train,test,temp)
#head(train_test$fecat)
gc(verbose = F)
var=c("cat80","cat87","cat57","cat12","cat79","cat10","cat7","cat89","cat2","cat72",
      "cat81","cat11","cat1","cat13","cat9","cat3","cat16","cat90","cat23","cat36",
      "cat73","cat103","cat40","cat28","cat111","cat6","cat76","cat50","cat5",
      "cat4","cat14","cat38","cat24","cat82","cat25")

for(i in 1:(length(var)-1))
{
  for(j in (i+1):length(var))
  {
    train_test[[paste(var[i],var[j],sep = "_")]]=train_test[[var[i]]]*
      train_test[[var[j]]]
  }
  
}  
ncol(train_test)
gc(verbose = F)


corr_mat=pcor(as.matrix(train_test))

#corrplot(corr_mat, order = "hclust", tl.cex = .35)
(CorVariables <- findCorrelation(corr_mat,.9,names = T))


#removing highly correled variables//this removes a lot of interaction variables
train_test[,c(CorVariables) := NULL]
#names(train_test)
remove(corr_mat,CorVariables)
gc(verbose = F)
ncol(train_test)

indx=grep("cat",names(train_test))
mat=as.matrix(train_test[,indx,with=F])
clus=makeCluster(3) #100/x
train_test$fecat=parRapply(clus,mat,function(x) sum(table(x)^2))
stopCluster(clus)
train_test$fecat=train_test$fecat/length(indx)^2

#clean-up
remove(mat,clus)
gc(verbose = F)




#simple log
(min=min(train_Y))
(max=max(train_Y))
train_Y=log(train_Y) #rescale(train_Y)

train_X=train_test[1:length(train_Y),]
test_X=train_test[(length(train_Y)+1):nrow(train_test),]




#create xgb model
dtest <- xgb.DMatrix(data = as.matrix(test_X))

remove(train_test,test_X)
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
           eta=.01, 
           max_depth= 16, #16
           subsample=.7, #.7
           colsample_bytree=.7, #.7
           min_child_weight=100, #100 
           base_score=median(train_Y)
)

rev_scale=function(x)
{
  x=x*(max-min)+min
}

xg_eval_mae <- function (yhat, dtrain) {
  y = as.numeric(getinfo(dtrain, "label"))
  err= as.numeric(mae(exp(y),exp(yhat))) #as.numeric(mae(rev_scale(y),rev_scale(yhat)))
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
for(i in 1:nfolds)
{
  cat("starting Fold",i,"\n")
  X_train=train_X[folds[[i]],]
  Y_train=train_Y[folds[[i]]]
  X_val=train_X[-folds[[i]],]
  Y_val=train_Y[-folds[[i]]]
  id_val=oob_id[-folds[[i]]]
  dtrain=xgb.DMatrix(data = as.matrix(X_train),label=Y_train)
  dtrain2=xgb.DMatrix(data = as.matrix(X_val),label=Y_val)
  watchlist=list(train=dtrain,test=dtrain2)
  model=xgb.train(params = param,data = dtrain,watchlist=watchlist,
                  early_stopping_rounds = 50,
                  feval=xg_eval_mae,
                  print_every_n = 50,nrounds = 6000,maximize=FALSE)
  
  pred=predict(model,dtest)
  prediction=prediction+exp(pred)#prediction+rev_scale(pred)
  dval=xgb.DMatrix(as.matrix(X_val))
  pred=exp(predict(model,dval))#rev_scale(predict(model,dval))
  oob=rbind(oob,cbind(id=id_val,real=exp(Y_val),pred=pred))
  gc(verbose = F)
}

#final mae
print(mae(oob$real,oob$pred))
prediction=prediction/nfolds

final$loss=prediction

#write file to disk
write.csv(final,"xgblog_test_outl.csv",row.names = F)

#write oob to disk if you want
oob=as.data.table(oob)
oob=oob[order(id)]
oob[,real:=NULL]


#write oob to disk if you want
write.csv(oob,"xgblog_train_outl.csv",row.names=F)
