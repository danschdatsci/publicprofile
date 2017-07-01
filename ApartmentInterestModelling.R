# Load packages and data
library(lubridate)
library(dplyr)
library(jsonlite)
library(caret)
library(purrr)
library(xgboost)
library(MLmetrics)
library(tidytext)
library(reshape2)
library(knitr)
library(stringr)
library(syuzhet)
library(DT)
library(data.table)
library(randomForest)
library(ggplot2)
library(e1071)
library(AER)


seed = 1842
set.seed(seed)


train <- fromJSON(".../train.json")
test <- fromJSON(".../test.json")

#########Unnest our json data
#Train
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)
train_id <-train$listing_id

#Test
vars <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
test_id <-test$listing_id

#Fill listings that have blank features
train[unlist(map(train$features,is_empty)),]$features = 'None'
test[unlist(map(test$features,is_empty)),]$features = 'None'

#add dummy interest level for test
test$interest_level <- 'test_set'


#combine train and test data for more feature extraction
all_data <- rbind(train,test)

str(all_data)
max(all_data$price)


ggplot(all_data, aes(x=price)) +
  geom_histogram(binwidth=1000, colour="black", fill="white") +
  coord_cartesian(xlim = c(0, 15000))

ggplot(all_data, aes(x=bathrooms)) +
  geom_histogram(binwidth=1, colour="black", fill="white") +
  coord_cartesian(xlim = c(0, 5))

ggplot(all_data, aes(x=bedrooms)) +
  geom_histogram(binwidth=1, colour="black", fill="white") +
  coord_cartesian(xlim = c(0, 6))

##############################Process Word features

#Variable extraction from features
all_features <- all_data[,names(all_data) %in% c("features","listing_id")] %>%
  filter(map(features, is_empty) != TRUE) %>%
  tidyr::unnest(features)
all_features$features <- tolower(all_features$features)

#get most common features with above 100 total occurences
feature_df = data.frame(feature = tolower(unlist(all_features$features))) %>% # convert all features to lower case
  group_by(feature) %>%
  summarise(feature_count = n()) %>%
  arrange(desc(feature_count)) %>%
  filter(feature_count >= 100)

#binning similar features according to our earlier "feature_df" result
#places_with_doormen <-all_features %>% filter(str_detect(all_features$features, paste(c('doorman','doormen'),collapse = "|")))
all_features$features[grep("doorman", all_features$features)] <- "doorman"
all_features$features[grep("doormen", all_features$features)] <- "doorman"

#places_with_laundry <- all_features[intersect(grep('laundry|dryer|washer', all_features$features),grep('dish', all_features$features,invert=TRUE)),]
all_features$features[intersect(grep('laundry|dryer|washer', all_features$features),grep('dish', all_features$features,invert=TRUE))] <- "laundry"

#places_with_roof <- all_features[grep('roof', all_features$features),]
all_features$features[grep('roof', all_features$features)] <- "roof deck"

#places_with_outdoor <- all_features[grep('outdoor', all_features$features),]
all_features$features[grep('outdoor', all_features$features)] <- "outdoor space"

#places_with_hardwood <- all_features[grep('hardwood', all_features$features),]
all_features$features[grep('hardwood', all_features$features)] <- "hardwood floors"

#places_with_pools <- all_features %>% filter(str_detect(all_features$features, 'pool'))
all_features$features[grep("pool", all_features$features)] <- "pool"
  
#places_with_fitness <- all_features[grep('fitness|gym', all_features$features),]
all_features$features[grep("fitness|gym", all_features$features)] <- "fitness"

#places_with_parking <- all_features[intersect(grep('parking', all_features$features),grep('valet', all_features$features,invert=TRUE)),]
all_features$features[intersect(grep('parking', all_features$features),grep('valet', all_features$features,invert=TRUE))] <- "parking"

#places_with_garage <- all_features %>% filter(str_detect(all_features$features, 'garage'))
all_features$features[grep("garage", all_features$features)] <- "garage"

#places_with_prewar <- all_features[grep('pre-war|prewar', all_features$features),]
all_features$features[grep('pre-war|prewar', all_features$features)] <- "pre-war"

#places_with_high_ceilings <- all_features[grep('high ceiling|high ceilings', all_features$features),]
all_features$features[grep('high ceiling|high ceilings', all_features$features)] <- "high ceilings"

#places_with_playroom <- all_features[grep('playroom', all_features$features),]
all_features$features[grep('playroom', all_features$features)] <- "children's playroom"

#places_with_super <- all_features[grep('super', all_features$features),]
all_features$features[grep('super', all_features$features)] <- "super"

#line below used for spot checking original listings
#unlist(all_data[which(all_data$listing_id == "7227432"),"features"])

#if feature length greater than 50, slice it out

all_features <- all_features[!nchar(as.character(all_features$features)) > 50,]


feature_df = data.frame(feature = tolower(unlist(all_features$features))) %>% # convert all features to lower case
  group_by(feature) %>%
  summarise(feature_count = n()) %>%
  arrange(desc(feature_count)) %>%
  filter(feature_count >= 100)


all_features_df <- as.data.table(all_features)
all_data_feat_df <- dcast.data.table(all_features_df[features %in% feature_df$feature], listing_id ~ features, fun.aggregate = function(x) as.integer(length(x) > 0), value.var = "features")


#merge word features back into main data frame
all_data<-merge(all_data, all_data_feat_df, by = "listing_id", sort = FALSE,all.x=TRUE)







#################Non-word features

#convert building and manager id to integer
all_data$building_id<-as.integer(factor(all_data$building_id))
all_data$manager_id<-as.integer(factor(all_data$manager_id))

#convert street and display address to integer
all_data$display_address<-as.integer(factor(all_data$display_address))
all_data$street_address<-as.integer(factor(all_data$street_address))


#convert date
all_data$created<-ymd_hms(all_data$created)
all_data$month<- month(all_data$created)
all_data$day<- day(all_data$created)
all_data$hour<- hour(all_data$created)
all_data$created = NULL


##Length of description in words
all_data$description_len<-sapply(strsplit(all_data$description, "\\s+"), length)


#price to bedroom ratio
all_data$bed_price <- all_data$price/all_data$bedrooms
all_data[which(is.infinite(all_data$bed_price)),]$bed_price = all_data[which(is.infinite(all_data$bed_price)),]$price

#add sum of rooms and price per room
all_data$room_sum <- all_data$bedrooms + all_data$bathrooms
all_data$room_diff <- all_data$bedrooms - all_data$bathrooms
all_data$room_price <- all_data$price/all_data$room_sum
all_data$bed_ratio <- all_data$bedrooms/all_data$room_sum
all_data[which(is.infinite(all_data$room_price)),]$room_price = all_data[which(is.infinite(all_data$room_price)),]$price

#log transforms
all_data$photo_count <- log(all_data$photo_count + 1)
all_data$feature_count <- log(all_data$feature_count + 1)
all_data$price <- log(all_data$price + 1)
all_data$room_price <- log(all_data$room_price + 1)
all_data$bed_price <- log(all_data$bed_price + 1)

#other minor counts
all_data$feature_count <- lengths(all_data$features)
all_data$features = NULL
all_data$photo_count <- lengths(all_data$photos)

#fill in listings that we couldn't extract features from
feature_list <- colnames(all_data_feat_df)
feature_list <- feature_list[2:66]

all_data[feature_list][is.na(all_data[feature_list])] <- 0
all_data[bed_ratio][is.na(all_data[bed_ratio])] <- 0



###############sentiment analysis
sentiment <- get_nrc_sentiment(all_data$description)
#datatable(head(sentiment))
all_data$negative <- sentiment$negative
all_data$positive <- sentiment$positive

all_data$description = NULL



##############################split train and test sets before building models
train <- all_data[all_data$listing_id %in%train_id,]
test <- all_data[all_data$listing_id %in%test_id,]

#Convert labels to integers
train$interest_level[train$interest_level == "low"] = 0
train$interest_level[train$interest_level == "medium"] = 1
train$interest_level[train$interest_level == "high"] = 2
y <- as.numeric(train$interest_level)
train$interest_level = NULL
train$photos = NULL
test$interest_level = NULL
test$photos = NULL





###################Build out Boosted tree model

xgb_params = list(
  booster="gbtree",#
  colsample_bytree= 0.7,
  subsample = 0.7,
  nthread=13,#
  eta = 0.1,
  objective= 'multi:softprob',
  max_depth= 4,
  min_child_weight= 1,
  eval_metric= "mlogloss",
  num_class = 3,
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
boostedtreemodel = xgb.train(params = xgb_params,
                 data = dtrain,
                 nrounds = 2000,
                 watchlist = list(train = dtrain, val=dval),
                 print_every_n = 25,
                 early_stopping_rounds=50)

importance_matrix <- xgb.importance(feature_names = colnames(train), model = boostedtreemodel)
xgb.plot.importance(importance_matrix = importance_matrix, top_n = 10)


#create xgboost model and get performance stats
trainpredictions =  as.data.frame(matrix(predict(boostedtreemodel, newdata = dtrain, type = "prob"), nrow=dim(train), byrow=TRUE))

trainpredictions$predresponse[trainpredictions$V1 > trainpredictions$V2 & trainpredictions$V1 > trainpredictions$V3] = 0
trainpredictions$predresponse[trainpredictions$V2 > trainpredictions$V1 & trainpredictions$V2 > trainpredictions$V3] = 1
trainpredictions$predresponse[trainpredictions$V3 > trainpredictions$V1 & trainpredictions$V3 > trainpredictions$V2] = 2

confusionMatrix(trainpredictions$predresponse,y)

#Make our csv file for submission
allpredictions =  as.data.frame(matrix(predict(boostedtreemodel,dtest), nrow=dim(test), byrow=TRUE))
allpredictions = cbind (allpredictions, test$listing_id)
names(allpredictions)<-c("low","medium","high","listing_id")
write.csv(allpredictions,paste0(Sys.Date(),"boostedtree",seed,".csv"),row.names = FALSE)








#############Build our random forest model
rftrain <- all_data[all_data$listing_id %in%train_id,]
rftest <- all_data[all_data$listing_id %in%test_id,]
rftrain$interest_level[rftrain$interest_level == "low"] = 0
rftrain$interest_level[rftrain$interest_level == "medium"] = 1
rftrain$interest_level[rftrain$interest_level == "high"] = 2

#Convert labels to integers
dependent_train <- as.integer(factor(rftrain$interest_level))
dependent_train <- dependent_train - 1
rftrain$interest_level = NULL
rftrain$photos = NULL
rftrain$bed_ratio = NULL
rftest$interest_level = NULL
rftest$photos = NULL
rftest$bed_ratio = NULL

fac_dependent_train <- as.factor(dependent_train)

formula <- as.formula(paste("fac_dependent_train", " ~ ."))

names(rftrain) <- make.names(names(rftrain), unique=TRUE)
names(rftest) <- names(rftrain)

rfmodel <- randomForest(formula, rftrain, ntree=10,keep.forest=TRUE)
summary(model)
plot(model)
varImpPlot(model, n.var = 10)

rftrainpredictions =  as.data.frame(predict(rfmodel, newdata = rftrain, type = "prob"))
rftrainpredictions$predresponse = 0
rftrainpredictions$predresponse[rftrainpredictions$"0" > rftrainpredictions$"1" & rftrainpredictions$"0" > rftrainpredictions$"2"] = 0
rftrainpredictions$predresponse[rftrainpredictions$"1" > rftrainpredictions$"0" & rftrainpredictions$"1" > rftrainpredictions$"2"] = 1
rftrainpredictions$predresponse[rftrainpredictions$"2" > rftrainpredictions$"0" & rftrainpredictions$"2" > rftrainpredictions$"1"] = 2

confusionMatrix(rftrainpredictions$predresponse,y)


rfpredict <- as.data.frame(predict(rfmodel, newdata = rftest, type = "prob"))
rfpredict$listing_id <- rftest$listing_id
rfpredict$low <- rfpredict$'0'
rfpredict$medium <- rfpredict$'1'
rfpredict$high <- rfpredict$'2'
rfpredict=rfpredict[,c(4,5,6,7)]
write.csv(rfpredict,paste0(Sys.Date(),"randomforest.csv"),row.names = FALSE)














#################multinomial regression
library(nnet)
multimodel <- multinom(formula, rftrain)
summary(multimodel)
varImpPlot(multimodel, n.var = 10)

rftest[is.na(rftest)] <- 0

mntrainpredictions =  as.data.frame(predict(multimodel, newdata = rftrain, type = "prob"))
mntrainpredictions$predresponse = 0
mntrainpredictions$predresponse[mntrainpredictions$"0" > mntrainpredictions$"1" & mntrainpredictions$"0" > mntrainpredictions$"2"] = 0
mntrainpredictions$predresponse[mntrainpredictions$"1" > mntrainpredictions$"0" & mntrainpredictions$"1" > mntrainpredictions$"2"] = 1
mntrainpredictions$predresponse[mntrainpredictions$"2" > mntrainpredictions$"0" & mntrainpredictions$"2" > mntrainpredictions$"1"] = 2

confusionMatrix(mntrainpredictions$predresponse,y)

variable_signif <- as.data.frame.matrix(coeftest(multimodel))
variable_signif <- variable_signif[0:88,]
variable_signif <- variable_signif[order(abs(variable_signif$Estimate), decreasing = TRUE),]

mnpredict <- as.data.frame(predict(multimodel, newdata = rftest, type = "prob"))
mnpredict$listing_id <- rftest$listing_id
mnpredict$low <- mnpredict$'0'
mnpredict$medium <- mnpredict$'1'
mnpredict$high <- mnpredict$'2'
mnpredict=mnpredict[,c(4,5,6,7)]
write.csv(mnpredict,paste0(Sys.Date(),"multinom.csv"),row.names = FALSE)


