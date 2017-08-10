# Below packages are for ROC curve and AUC calculation
install.packages("verification")
install.packages("ROCR") 
library("verification")
library("ROCR")             #ROC Curve, AUC
library(rpart)              #Classification Tree

## Read Project data
data <- read.csv("./letter-recognition.data", header=FALSE)
colnames(data) <- c("letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar",
                    "y2bar", "xybar", "x2ybr", "xy2br", "x-edge", "xegvy", "y-ege", "yegvx")

## Creating Test and Training Dataset with 80% of original dataset going for Training Dataset
subset <- sample(nrow(data), nrow(data) *0.8)
data.train <- data[subset, ]
data.test <- data[-subset, ]

#---------------------------------------------------------------------------------------------------------
## Summary statistics
##--------------------
dim(data)         #Dimensions of the Entire Dataset
dim(data.train)   #Dimensions of the Training Dataset
dim(data.test)    #Dimensions of the Test Dataset
summary(data)     #5-fold summary statistic of the entire dataset

## Histogram of all independent variables
for(colname in colnames(data))
{
  if(colname == "letter") next
  hist(data[, colname], main = paste("Histogram of", colname))
}

##Scatter Plots (check for relationship between independent variables)
pairs(data[,c(2:6,1)], main = "Scatterplot for variables x-box, y-box, width, high, onpix vs. letter")
pairs(data[,c(7:11,1)], main = "Scatterplot for variables x-bar, y-bar, x2bar, y2bar, xybar vs. letter")
pairs(data[,c(12:17,1)], main = "Scatterplot for variables x2ybr, xy2br, x-edge, xegvy, y-ege, yegvx vs. letter")

## Box Plots -- Outlier Detection
boxplot(data.train[,2:17], main = "Boxplot for Predictor Variables in the Training Dataset")
#---------------------------------------------------------------------------------------------------------
## Classification Tree
##--------------------
pcut <- 0.3
## Symmetric cost function -- needs to be re-written for classification problem with more than two outcomes. 
symmetric_cost_fn <- function(r, pi) {
  mean(((r == 0) & (pi > pcut)) | ((r == 1) & (pi < pcut)))
}

## Create a Classification Tree with symmetric cost function for Mis-Classification Rate (MR)
data.class_tree <- rpart(formula = letter ~ ., data = data.train, method = "class", cp = 0.0056)
data.class_tree
plot(data.class_tree)                   #Plot the Classification Tree Model
text(data.class_tree, pretty = TRUE)

## Find the relationship between 10-fold cross validation error in the training set and the size of tree
plotcp(data.class_tree)
summary(data.class_tree)

#In-sample Prediction & Misclassification Rate, ROC curve with training set
data.train.pred.tree = predict(data.class_tree, data.train, type = "class")
data.train.pred.tree.prob = predict(data.class_tree, data.train, type = "prob")
table(data.train$letter, data.train.pred.tree, dnn = c("Truth", "Predicted"))
data.train.accuracy <- sum(data.train$letter == data.train.pred.tree)/nrow(data.train)
data.train.accuracy   #0.5445625

data.test.pred.tree = predict(data.class_tree, data.test, type = "class")
data.test.pred.tree.prob = predict(data.class_tree, data.test, type = "prob")
table(data.test$letter, data.test.pred.tree, dnn = c("Truth", "Predicted"))
data.test.accuracy <- sum(data.test$letter == data.test.pred.tree)/nrow(data.test)
data.test.accuracy    #0.5385