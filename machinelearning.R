# load packages
library(caret)
library(mlbench)
library(randomForest)
library(doMC)

library(corrplot)
# attach the iris dataset to the environment
data(iris)
# rename the dataset
dataset <- iris
# create a list of 80% of the rows in the original dataset we can use for training
validationIndex <- createDataPartition(dataset$Species, p=0.80, list=FALSE) # select 20% of the data for validation
validation <- dataset[-validationIndex,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validationIndex,]

# Run algorithms using 10-fold cross validation
trainControl <- trainControl(method="cv", number=10)
metric <- "Accuracy"



# LDA
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric,
                 trControl=trainControl)
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric,
                  trControl=trainControl)
# KNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric,
                 trControl=trainControl)
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric,
                 trControl=trainControl)
#RF
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=trainControl)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summarize Best Model
print(fit.lda)



predictions <- predict(fit.lda, validation) 
confusionMatrix(predictions, validation$Species)