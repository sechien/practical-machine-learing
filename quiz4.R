library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(gbm)
library(lubridate)
library(forecast)
library(e1071)

# load q1 data
data(vowel.train)
data(vowel.test)
set.seed(33833)
# set y as factor
vowel.train$y <- as.factor(vowel.train$y)
vowel.test$y <- as.factor(vowel.test$y)
# fit with random forest and gbm
rfFit <- train(y ~., data =vowel.train, method = "rf")
gbmFit <- train(y ~., data =vowel.train, method = "gbm")
# predict
rfResult <- predict(rfFit, newdata = vowel.test)
gbmPredict <- predict(gbmFit, newdata = vowel.test)
# predict with combining predictors

# create dataframe
preDF <- data.frame(rfResult, gbmPredict, y = vowel.test$y)
combined_fit <- train(y ~ ., data = preDF, method = "gam")
combinedPredict <- predict(combined_fit, newdata = preDF)

# get confusion matrix
confusionMatrix(rfResult, vowel.test$y)$overall['Accuracy']
confusionMatrix(gbmPredict, vowel.test$y)$overall['Accuracy']
confusionMatrix(combinedPredict, vowel.test$y)$overall['Accuracy'] ## this is wrong

## q2

set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

set.seed(62433)
rf.model <- train(diagnosis~., data=training, method = "rf")
gbm.model <- train(diagnosis~., data=training, method = "gbm")
lda.model <- train(diagnosis~., data=training, method = "lda")

rfResult <- predict(rf.model, testing)
gbmResult <- predict(gbm.model, testing)
ldaResult <- predict(lda.model, testing)

# stack
combined.data <- data.frame(rfResult, gbmResult, ldaResult, diagnosis = testing$diagnosis)
combined.model <- train(diagnosis ~., data = combined.data, method = "rf")

# predict by combined model
combinedResult  <- predict(combined.model, testing)

# get accuracy
confusionMatrix(testing$diagnosis, rfResult)$overall['Accuracy']
confusionMatrix(testing$diagnosis, gbmResult)$overall['Accuracy']
confusionMatrix(testing$diagnosis, ldaResult)$overall['Accuracy']
confusionMatrix(testing$diagnosis, combinedResult)$overall['Accuracy']

# q3
set.seed(3523)

library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(233)
lassoFit <- train(CompressiveStrength ~ ., data = training, method = "lasso")
plot.enet(lassoFit$finalModel, xvar="penalty", use.color=TRUE)

## q4
library(forecast)
library(quantmod)
library(lubridate)  
dat = read.csv("~/R/practical machine learing/gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

ts.model <- bats(tstrain)
fcast <- forecast(ts.model, level = 95, h = dim(testing)[1])
sum(fcast$lower < testing$visitsTumblr & testing$visitsTumblr < fcast$upper) / 
    dim(testing)[1]

#  q5

set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

set.seed(325)
library(e1071)
mod_svm <- svm(CompressiveStrength ~ ., data = training)
pred_svm <- predict(mod_svm, testing)
accuracy(pred_svm, testing$CompressiveStrength)