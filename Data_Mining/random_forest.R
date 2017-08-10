install.packages("randomForest")
library("randomForest")

## Seperate out independent and dependent variables in the complete dataset
y <- data[,1]
x <- data[,2:17]

## Separate out independent and dependent variables in the training dataset
y.train <- data.train[,1]
x.train <- data.train[,2:17]
y.test <- data.test[,1]
x.test <- data.test[,2:17]

## Create the random forest
rf.train <- randomForest(x.train, y.train)
print(rf.train)

## Plot
##"Prediction error as a function of number of trees"
plot(rf.train)
plot(predict(rf.train, data=x.train), y.train)
abline(c(0,1),col=2)

## Determine the optimal value of mtry (Number of variables randomly sampled as candidates at each split)
## Default mtry=4, try values 2:10 step 2
for (mtry in seq(2, 10, 2)) {
  rf.train.mtry <- randomForest(x.train, y.train, mtry=mtry)
  print(paste("OOB estimate for num_trees=500 mtry=", mtry, "is = ", rf.train.mtry$err.rate[500]))
}
plot(seq(2, 10, 2), c(0.0344375*100, 0.0356875*100, 0.0375625*100, 0.039625*100, 0.0433125*100), type="b", pch=21, col="red",
     ylim=range(seq(3.4, 4.4, 0.2)),
     main="Prediction Error as a function of number of variables sampled", xlab="number of variables randomly sampled",
     ylab="OOB Error Estimate (%)")


## Which variables are important for prediction ?
rf <- randomForest(x, y, importance=TRUE)
varImpPlot(rf, type = 1, main = paste("Random Forest - Variable Importance Chart (entire dataset)"))

## Optimized Random Tree for In-sample and Out-of-Sample prediction
rf.train.opt <- randomForest(x.train, y.train, mtry=2)
y.train.pred <- predict(rf.train.opt, x.train, type = "class")
table(y.train, y.train.pred, dnn = c("Truth", "Predicted"))
acc <-  sum(y.train == y.train.pred)/nrow(x.train)
paste("Prediction Accuracy for In-Sample Prediction = ", acc)


##Optimized Random Tree Out-of-Sample Prediction Accuracy
y.test.pred <- predict(rf.train.opt, x.test, type = "class")
table(y.test, y.test.pred, dnn = c("Truth", "Predicted"))
acc <-  sum(y.test == y.test.pred)/nrow(x.test)
paste("Prediction Accuracy for In-Sample Prediction = ", acc)