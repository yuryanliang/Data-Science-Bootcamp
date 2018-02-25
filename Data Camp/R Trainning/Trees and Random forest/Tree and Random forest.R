# INFORMS @USF DATA BOOTCAMP 2017
# Read Data 

Train<-read.csv("TrainCleaned.csv")
Test<-read.csv("TestCleaned.csv")

# Call rpart and rpart.plot libraries, install them if required

install.packages('rpart')
install.packages('rpart.plot')
library(rpart)
library(rpart.plot)

# Grow the tree
fit<-rpart(Survived~ Age+Sex+SibSp+Pclass+Parch+Fare+Embarked,data=Train,method="class")

# Plot the tree
prp(fit)

#Predictions

predictCART<- predict(fit,newdata = Test,type="class")

# Evaluation

table(Test$Survived,predictCART)

# ****** Randon Forest**********

#required packages
install.packages("randomForest")
library(randomForest)

#Decision variable must be a factor because this is a classification problem.
Train$Survived=as.factor(Train$Survived)
Test$survived=as.factor( Test$Survived)

#Train forest
fit2<-randomForest(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked,data=Train,ntree=2000)

#Evuate results using the confusion matrix
predictForest<-predict(fit2,newdata=Test)
table(Test$Survived,predictForest)



