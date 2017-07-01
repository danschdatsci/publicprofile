# load libraries
library(caret)
library(nnet)
library(tidyverse)
library(corrplot)
library(magrittr)
library(zoo)
library(MASS)
library(xgboost)

set.seed(1000)

train_features = read.csv('.../dengue_features_train.csv')
train_labels   = read.csv('.../dengue_labels_train.csv')

#get our dependent variable into the training set
train_features$total_cases <- train_labels$total_cases
train_labels <- NULL

summary(train_features)

#we are missing a lot of the ndvi measures, so lets fill the missing with "0" and sum them into one new variable
veg_index_fill <- as.data.frame(cbind(train_features$ndvi_ne, train_features$ndvi_nw, train_features$ndvi_se, train_features$ndvi_sw))
veg_index_fill[is.na(veg_index_fill)] <- 0
veg_index_fill$total <- veg_index_fill$V1 + veg_index_fill$V2 + veg_index_fill$V3 + veg_index_fill$V4
train_features$total_ndvi <- veg_index_fill$total

#split the data by city
sj_train_features = train_features %>% filter(city == 'sj')
iq_train_features = train_features %>% filter(city == 'iq')



#Basic plots
plot(sj_train_features$total_cases, type = "l")
plot(iq_train_features$total_cases, type = "l")
plot(hist(train_features$total_cases, breaks = "fd"))

train_features %>% 
  dplyr::select(-city, -year, -weekofyear, -week_start_date) %>%
  cor(use = 'pairwise.complete.obs') -> correlation_matrix

png(height=800, width=1000, pointsize=15, file='C:/Users/Dan/Desktop/Northwestern/Predict 413 - Spring 2017/Final/dengue_features_train.png')
corrplot(correlation_matrix, type="lower", method="color",diag=FALSE)
dev.off()
#not a whole lot helping us here...

#lets try some lag
shift <- function(x, n){
  c(x[-(seq(n))], rep(NA, n))
}

lag_weeks <- seq(1,24,1)

for (week in lag_weeks) {
  sj_train_features[paste("lagged_total_cases_", week)] <- shift(sj_train_features$total_cases, week)
}

sj_train_features %>% 
  dplyr::select(-city, -year, -weekofyear, -week_start_date) %>%
  cor(use = 'pairwise.complete.obs') -> correlation_matrix

png(height=800, width=1000, pointsize=12, file='C:/Users/Dan/Desktop/Northwestern/Predict 413 - Spring 2017/Final/dengue_sj_features_train_explore.png')
corrplot(correlation_matrix, type="lower", method="color",diag=FALSE)
dev.off()

#we can clearly see that our temperature predictor variables are most effective with 8 weeks of lag
#lets try the same with iq

for (week in lag_weeks) {
  iq_train_features[paste("lagged_total_cases_",week)] <- shift(iq_train_features$total_cases, week)
}

iq_train_features %>% 
  dplyr::select(-city, -year, -weekofyear, -week_start_date) %>%
  cor(use = 'pairwise.complete.obs') -> correlation_matrix

png(height=800, width=1000, pointsize=12, file='C:/Users/Dan/Desktop/Northwestern/Predict 413 - Spring 2017/Final/dengue_iq_features_train_explore.png')
corrplot(correlation_matrix, type="lower", method="color",diag=FALSE)
dev.off()
#it looks like something is different about the environment of iq that our predictors react 
#in a completely distinctive way


eightwklag <- c(NA,NA,NA,NA,NA,NA,NA,NA)
eightwklag <- append(eightwklag, sj_train_features$reanalysis_air_temp_k)
eightwklag <- head(eightwklag, (length(eightwklag) - 8))

sj_train_features$eight_week_lag_temp <- eightwklag

eightwklag <- c(NA,NA,NA,NA,NA,NA,NA,NA)
eightwklag <- append(eightwklag, iq_train_features$reanalysis_air_temp_k)
eightwklag <- head(eightwklag, (length(eightwklag) - 8))

iq_train_features$eight_week_lag_temp <- eightwklag

# split up the data into training and validation subsets
sj_train_subtrain <- head(sj_train_features, 800)
sj_train_subtest  <- tail(sj_train_features, nrow(sj_train_features) - 800)

iq_train_subtrain <- head(iq_train_features, 400)
iq_train_subtest  <- tail(iq_train_features, nrow(iq_train_features) - 400)

features = c("reanalysis_dew_point_temp_k", 
               "reanalysis_tdtr_k",
               "reanalysis_specific_humidity_g_per_kg",
               "eight_week_lag_temp")

sj_train_subtrain[features] %<>% na.locf(fromLast = TRUE)
iq_train_subtrain[features] %<>% na.locf(fromLast = TRUE) 

sj_train_subtest[features] %<>% na.locf(fromLast = TRUE)
iq_train_subtest[features] %<>% na.locf(fromLast = TRUE) 






###########NEGATIVE BINOMIAL############################################
mae <- function(error) return(mean(abs(error)))

neg_bin_model <- function(train, test)
{
  form <- "total_cases ~ 1 +
  weekofyear +
  eight_week_lag_temp +
  reanalysis_dew_point_temp_k + 
  reanalysis_tdtr_k +
  reanalysis_specific_humidity_g_per_kg"
  

  combined <- rbind(train, test)
  combined_model = glm.nb(formula=form,
                          data = combined)
  
  return (combined_model)
}

sj_negbin_model <- neg_bin_model(sj_train_subtrain, sj_train_subtest)
sj_negbin_results <-  predict(sj_negbin_model, sj_train_subtest)
sj_negbin_score   <-  mae(sj_train_subtest$total_cases - sj_negbin_results)

iq_negbin_model <- neg_bin_model(iq_train_subtrain, iq_train_subtest)
iq_negbin_results <-  predict(iq_negbin_model, iq_train_subtest)
iq_negbin_score   <-  mae(iq_train_subtest$total_cases - iq_negbin_results)

summary(sj_negbin_model)

plot(sj_train_features$total_cases, type = "l")
lines(sj_negbin_model$fitted, col="blue")

plot(iq_train_features$total_cases, type = "l")
lines(iq_negbin_model$fitted, col="blue")


#########NEURAL NET MODEL################################################

avnnet_model <-  function(train, test) {
  
combined <- rbind(train, test)
combined_model <- avNNet(total_cases ~ 1 +
                           eight_week_lag_temp +
                           weekofyear +
                           reanalysis_dew_point_temp_k + 
                           reanalysis_tdtr_k +
                           reanalysis_specific_humidity_g_per_kg, combined, repeats=25, size=20, decay=0.1, linout = TRUE)

  return (combined_model)
}

sj_avnnet_model <- avnnet_model(sj_train_subtrain, sj_train_subtest)
sj_avnnet_results <-  predict(sj_avnnet_model, sj_train_subtest)
sj_avnnet_score   <-  mae(sj_train_subtest$total_cases - sj_avnnet_results)

iq_avnnet_model <- avnnet_model(iq_train_subtrain, iq_train_subtest)
iq_avnnet_results <-  predict(iq_avnnet_model, iq_train_subtest)
iq_avnnet_score   <-  mae(iq_train_subtest$total_cases - iq_avnnet_results)

varImp(sj_avnnet_model)

fitted <- predict(sj_avnnet_model, sj_train_subtrain)
plot(sj_train_features$total_cases, type = "l")
lines(fitted, col="blue")

fitted <- predict(iq_avnnet_model, iq_train_subtrain)
plot(iq_train_features$total_cases, type = "l")
lines(fitted, col="blue")

##########BOOSTED TREE###########################################

train <- sj_train_subtrain
test <- sj_train_subtest
y <- train$total_cases

lag_weeks <- seq(1,24,1)

for (week in lag_weeks) {
  train[paste("lagged_total_cases_", week)] <- NULL
}

train$city <- NULL
train$total_cases <- NULL
train$week_start_date <- NULL
train$year <- NULL

for (week in lag_weeks) {
  test[paste("lagged_total_cases_", week)] <- NULL
}

test$city <- NULL
test$total_cases <- NULL
test$week_start_date <- NULL
test$year <- NULL

seed = 1000

xgb_params = list(
  booster="gbtree",
  colsample_bytree= 0.7,
  subsample = 0.7,
  nthread=13,#
  eta = 0.1,
  objective= 'count:poisson',
  max_depth= 4,
  min_child_weight= 1,
  eval_metric= "mae",
  seed = seed
)


#convert test data to an xgbmatrix
dtest <- xgb.DMatrix(data.matrix(test))

#create cross validation folds
kfolds<- 10
folds<-createFolds(y, k = kfolds, list = TRUE, returnTrain = FALSE)
fold <- as.numeric(unlist(folds[1]))

x_train<-train[-fold,]
x_val<-train[fold,]

y_train<-y[-fold]
y_val<-y[fold]


#convert training data to xgbmatrix
dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dval = xgb.DMatrix(as.matrix(x_val), label=y_val)

#perform training
sj_boostedtree_model = xgb.train(params = xgb_params,
                             data = dtrain,
                             nrounds = 2000,
                             watchlist = list(train = dtrain, val=dval),
                             print_every_n = 25,
                             early_stopping_rounds=50)

importance_matrix <- xgb.importance(feature_names = colnames(train), model = sj_boostedtree_model)
xgb.plot.importance(importance_matrix = importance_matrix, top_n = 10)


#create xgboost model and get performance stats
dtrain <- xgb.DMatrix(data.matrix(train))
trainpredictions =  predict(sj_boostedtree_model, newdata = dtrain)

plot(sj_train_features$total_cases, type = "l")
lines(trainpredictions, col="blue")

xofy <- seq(801,936,1)

testpredictions =  predict(sj_boostedtree_model, newdata = dtest)
lines(xofy, testpredictions, col="red")

sj_boostedtree_results <-  predict(sj_boostedtree_model, newdata = dtest)
sj_boostedtree_score   <-  mae(sj_train_subtest$total_cases - sj_boostedtree_results)




train <- iq_train_subtrain
test <- iq_train_subtest
y <- train$total_cases

lag_weeks <- seq(1,24,1)

for (week in lag_weeks) {
  train[paste("lagged_total_cases_", week)] <- NULL
}

train$city <- NULL
train$total_cases <- NULL
train$week_start_date <- NULL
train$year <- NULL

for (week in lag_weeks) {
  test[paste("lagged_total_cases_", week)] <- NULL
}

test$city <- NULL
test$total_cases <- NULL
test$week_start_date <- NULL
test$year <- NULL

seed = 1000

xgb_params = list(
  booster="gbtree",
  colsample_bytree= 0.7,
  subsample = 0.7,
  nthread=13,#
  eta = 0.1,
  objective= 'count:poisson',
  max_depth= 4,
  min_child_weight= 1,
  eval_metric= "mae",
  seed = seed
)


#convert test data to an xgbmatrix
dtest <- xgb.DMatrix(data.matrix(test))

#create cross validation folds
kfolds<- 10
folds<-createFolds(y, k = kfolds, list = TRUE, returnTrain = FALSE)
fold <- as.numeric(unlist(folds[1]))

x_train<-train[-fold,]
x_val<-train[fold,]

y_train<-y[-fold]
y_val<-y[fold]


#convert training data to xgbmatrix
dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dval = xgb.DMatrix(as.matrix(x_val), label=y_val)

#perform training
iq_boostedtree_model = xgb.train(params = xgb_params,
                             data = dtrain,
                             nrounds = 2000,
                             watchlist = list(train = dtrain, val=dval),
                             print_every_n = 25,
                             early_stopping_rounds=50)

importance_matrix <- xgb.importance(feature_names = colnames(train), model = iq_boostedtree_model)
xgb.plot.importance(importance_matrix = importance_matrix, top_n = 10)


#create xgboost model and get performance stats
dtrain <- xgb.DMatrix(data.matrix(train))
trainpredictions =  predict(iq_boostedtree_model, newdata = dtrain)

plot(iq_train_features$total_cases, type = "l")
lines(trainpredictions, col="blue")

xofy <- seq(401,520,1)

testpredictions =  predict(iq_boostedtree_model, newdata = dtest)
lines(xofy, testpredictions, col="red")

iq_boostedtree_results <-  predict(iq_boostedtree_model, newdata = dtest)
iq_boostedtree_score   <-  mae(iq_train_subtest$total_cases - iq_boostedtree_results)



#Make our csv files for submission

test_features = read.csv('.../dengue_features_test.csv')

#repeat our data treatments we did ont he training set
veg_index_fill <- as.data.frame(cbind(test_features$ndvi_ne, test_features$ndvi_nw, test_features$ndvi_se, test_features$ndvi_sw))
veg_index_fill[is.na(veg_index_fill)] <- 0
veg_index_fill$total <- veg_index_fill$V1 + veg_index_fill$V2 + veg_index_fill$V3 + veg_index_fill$V4
test_features$total_ndvi <- veg_index_fill$total

#split the data by city
sj_test_features = test_features %>% filter(city == 'sj')
iq_test_features = test_features %>% filter(city == 'iq')


#add 8 wk lagged temp variable
eightwklag <- c(NA,NA,NA,NA,NA,NA,NA,NA)
eightwklag <- append(eightwklag, sj_test_features$reanalysis_air_temp_k)
eightwklag <- head(eightwklag, (length(eightwklag) - 8))

sj_test_features$eight_week_lag_temp <- eightwklag

eightwklag <- c(NA,NA,NA,NA,NA,NA,NA,NA)
eightwklag <- append(eightwklag, iq_test_features$reanalysis_air_temp_k)
eightwklag <- head(eightwklag, (length(eightwklag) - 8))

iq_test_features$eight_week_lag_temp <- eightwklag

#fill NAs
features = c("reanalysis_dew_point_temp_k", 
             "reanalysis_tdtr_k",
             "reanalysis_specific_humidity_g_per_kg",
             "eight_week_lag_temp")

sj_test_features[features] %<>% na.locf(fromLast = TRUE)
iq_test_features[features] %<>% na.locf(fromLast = TRUE) 

sj_test_features$negbin_predicted <- predict(sj_negbin_model,sj_test_features)
iq_test_features$negbin_predicted <- predict(iq_negbin_model,iq_test_features)

submissions = read.csv('.../submission_format.csv')
inner_join(submissions, rbind(sj_test_features,iq_test_features)) %>%
  dplyr::select(city, year, weekofyear, total_cases = negbin_predicted) ->
  negbin_predictions

negbin_predictions$total_cases %<>% round()
write.csv(negbin_predictions, '.../negbin_predictions.csv', row.names = FALSE)


sj_test_features$avnnet_predicted <- predict(sj_avnnet_model,sj_test_features)
iq_test_features$avnnet_predicted <- predict(iq_avnnet_model,iq_test_features)

inner_join(submissions, rbind(sj_test_features,iq_test_features)) %>%
  dplyr::select(city, year, weekofyear, total_cases = avnnet_predicted) ->
  avnnet_predictions

avnnet_predictions$total_cases %<>% round()
write.csv(avnnet_predictions, '.../avnnet_predictions.csv', row.names = FALSE)


sj_test_features$boostedtree_predicted <- predict(sj_boostedtree_model, newdata = xgb.DMatrix(data.matrix(sj_test_features)))
iq_test_features$boostedtree_predicted <- predict(iq_boostedtree_model, newdata = xgb.DMatrix(data.matrix(iq_test_features)))

inner_join(submissions, rbind(sj_test_features,iq_test_features)) %>%
  dplyr::select(city, year, weekofyear, total_cases = boostedtree_predicted) ->
  boostedtree_predictions

boostedtree_predictions$total_cases %<>% round()
write.csv(boostedtree_predictions, '.../boostedtree_predictions.csv', row.names = FALSE)



