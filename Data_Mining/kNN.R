#k-Nearest Neighbors Algorithm (kNN)
#Library class comes by default with R. Note the kNN function in class can only do the following:
# - only performs "Classification" predictions with "Numeric" Independent (x) variables 
library(class)

## Convert X's into numeric and Y into Factor
x.train <- sapply(data.train[,2:17], as.numeric) 
x.test  <- sapply(data.test[,2:17], as.numeric)
y.train <- sapply(data.train[,1], as.factor)
y.test  <- sapply(data.test[,1], as.factor)

## Structure and Dimensions of the test and training dataset
str(x.train); str(y.train) 
str(x.test) ; str(y.test)
dim(x.train); dim(y.train) 
dim(x.test) ; dim(y.test)

## Function to Normalize Independent X variables. 
## Note, scale of Independent variables has an impact on prediction, so using normalized Z-scores
normalize_fn <- function(numbers) {
  (numbers - mean(numbers))/sd(numbers)
}

## Normalize X's in the test and train Datasets
## Apply() can apply function to each row or column in a matrix or data frame
## Use "2" to do it by column, use "1" to do it by row
x.train.normalized <- apply(x.train, 2, normalize_fn)
x.test.normalized <- apply(x.test, 2, normalize_fn)

## Verify normalization results - Mean should be close to zero and SD = 1 (Z score)
apply(x.train.normalized, 2, mean)
apply(x.train.normalized, 2, sd)
apply(x.test.normalized, 2, mean)
apply(x.test.normalized, 2, sd)


## Determine the optimal value of 'k' with cross-validation on the Normalized Training Dataset
## Loop through odd values of k = 1, 3, ..., 19
k.cvv.opt = 0
k.cvv.opt.accuracy = 0.0
accuracy_vector <- c()
for (k in seq(1, 20, 2)) {
  y.cvv.predicted <- knn.cv(x.train.normalized, y.train, k)
  k.cvv.accuracy <- sum(y.train == y.cvv.predicted)/nrow(x.train.normalized)
  accuracy_vector <- c(accuracy_vector, k.cvv.accuracy)
  if(k.cvv.opt.accuracy < k.cvv.accuracy){
    k.cvv.opt.accuracy <- k.cvv.accuracy
    k.cvv.opt <- k
  }
  print(paste("With", k, "neighbors the accuracy is", k.cvv.accuracy))
}
accuracy_vector <- accuracy_vector * 100;
plot(seq(1, 20, 2), accuracy_vector, type="b", pch=21, col="red", xaxt="n", 
     main = "Accuracy as a function of number of neighbors(k)",
     xlab="number of neighbors (k)", ylab="Accuracy rate (%)")
axis(seq(1, 20, 2), side=1,labels=seq(1, 20, 2))
print(paste("Optimal value of k between 1-20 = ", k.cvv.opt))
## We are seeing that as we increase the value of 'k', the accuracy decreases.

## Making Predictions on the Training Dataset with optimum 'k' computed with cross-validation
y.train.predicted <- knn(x.train.normalized, x.train.normalized, y.train, k.cvv.opt)
k.train.accuracy <-  sum(y.train == y.train.predicted)/nrow(x.train.normalized)
print(paste("Training Dataset Prediction Accuracy (with opt k) =", k.train.accuracy))

## Making Predictions on the Test Dataset with optimum 'k' computed with cross-validation
y.test.predicted <- knn(x.train.normalized, x.test.normalized, y.train, k.cvv.opt)
k.test.accuracy <-  sum(y.test == y.test.predicted)/nrow(x.test.normalized)
print(paste("Test Dataset Prediction Accuracy (with opt k) =", k.test.accuracy))

## Confusion Matrix for In-Sample and Out-of-Sample Prediction
table(y.train, y.train.predicted, dnn = c("Truth", "Predicted"))
table(y.test, y.test.predicted, dnn = c("Truth", "Predicted"))

## Generate the Accuracy values for all values of 'k', similar to what was done before for kNN-CVV
train_acc <- c()
test_acc <- c()
for (k in seq(1, 20, 2)) {
  y.train.predicted <- knn(x.train.normalized, x.train.normalized, y.train, k)
  train_acc <- c(train_acc, sum(y.train == y.train.predicted)/nrow(x.train.normalized))
  print(paste("Train (k = ", k, ") Accuracy = ", 
              sum(y.train == y.train.predicted)/nrow(x.train.normalized) ))
  
  y.test.predicted <- knn(x.train.normalized, x.test.normalized, y.train, k)
  test_acc <- c(test_acc,  sum(y.test == y.test.predicted)/nrow(x.test.normalized))
  print(paste("Test (k = ", k, ") Accuracy = ", 
              sum(y.test == y.test.predicted)/nrow(x.test.normalized) ))
}
train_acc <- train_acc * 100
test_acc <- test_acc * 100
plot(seq(1, 20, 2), train_acc, type="b", pch=21, col="green", xaxt="n", 
     main = "Train/Test Accuracy as a function of number of neighbors(k)",
     xlab="number of neighbors (k)", ylab="Accuracy rate (%)",
     xlim=range(seq(1, 20, 2)), ylim=range(c(90:100)))
par(new=T)
plot(seq(1, 20, 2), test_acc, type="b",axes=F,col="red", xaxt="n", yaxt="n", xlab=" ", ylab=" ",
     xlim=range(seq(1, 20, 2)), ylim=range(c(90:100)))
axis(seq(1, 20, 2), side=1,labels=seq(1, 20, 2))
