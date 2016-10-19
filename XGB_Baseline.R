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

#read the data set
setwd("d:/Kaggle Competition/")
raw_data=fread("train.csv",stringsAsFactors = T)
raw_data=data.table(raw_data)

#splitting train and test data for grouped categories
set.seed(1)
index=createDataPartition(raw_data$cat116,p=.8,list = F)
index = sort(sample(nrow(raw_data), nrow(raw_data)*.8))
train=raw_data[index,]
test=raw_data[-index,]

#create sparse matrix
train_matrix <- sparse.model.matrix(id+loss~.-1, data = train)
test_matrix <- sparse.model.matrix(id+loss~.-1,data = test)


#create xgb model

dtrain <- xgb.DMatrix(data = train_matrix, label = train$loss)
dtest <- xgb.DMatrix(data =test_matrix, label = test$loss)
watchlist <- list(train=dtrain, test=dtest)
set.seed(1)
xgb=xgboost(data = dtrain,
            nrounds = 200,
            eta=.1,
            max_depth= 15,
            subsample=.5,
            booster="gbtree",
            objective = "reg:linear",
            eval_metric="rmse",
            colsample_bytree=.5,
            min_child_weight=1,
            watchlist=watchlist
)
xgb_pred=predict(xgb,test_matrix)
(RMSE_xgb=sqrt(mean((xgb_pred-test$loss)^2)))

#Data Processing

##Cut the sample into two parts
quantile(raw_data$loss,prob=seq(.9,1,length=11),type = 5)

raw_data_extreme=raw_data[raw_data$loss>=15000,]
dim(raw_data)
nrow(raw_data_extreme)
raw_data_normal=raw_data[raw_data$loss<6400,]
quantile(raw_data_extreme$loss,prob=seq(0,1,length=11),type=5)

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
