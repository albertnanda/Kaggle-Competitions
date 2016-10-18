#Kaggle Competition #Fresh Attempt
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
library(xgboost)

#read the data set
setwd("d:/Kaggle Competition/")
raw_data=read.csv("train.csv",header=T)


#splitting train and test data for grouped categories
set.seed(1)
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
            nrounds = 2000,
            eta=.01,
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









#parameter Tuning

xgb.grid <- expand.grid(nrounds = 100,
                        eta = c(0.01,0.1),
                        max_depth = 15,
                        gamma=1,
                        colsample_bytree=.5,
                        min_child_weight=1)
set.seed(1)
trCtrl=trainControl(method = "cv")
xgbTune=train(loss~.-id,data=train,method="xgbTree",
              trControl=trCtrl,
              subsample=.5,
              tuneGrid=xgb.grid,
              booster="gbtree",
              objective = "reg:linear",
              eval_metric="rmse",
              watchlist=watchlist
)
