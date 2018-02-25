rm(list=ls())
getwd()
setwd("C:/Users/nsakib/Desktop/game1/rawtitanic")

training.data.raw=as.data.frame(read.csv(file="train.csv",header=TRUE,na.strings = c("")))

dim(training.data.raw)

pairs(training.data.raw)

## seeing levels
sapply(training.data.raw, function(x) length(unique(x)))


### cleaning ###
#check missing values

sapply(training.data.raw,function(x) sum(is.na(x)))

#alternately,
install.packages("Amelia")
library(Amelia)
missmap(training.data.raw, main = "Missing values vs observed")

### dealing with NA

### Cabin has too many missing valus - should be dropped
### so should name and id

data <- subset(training.data.raw,select=c(2,3,5,6,7,8,10,12))
missmap(data, main = "Missing values vs observed")

### getting rid of embarked missing values
data <- data[!is.na(data$Embarked),]
missmap(data, main = "Missing values vs observed")

#way 1 - remove all rows with NA
dataA=na.omit(data)
missmap(dataA, main = "Missing values vs observed")
dim(dataA)

#way 2 - imputation with mean, median or mode, Let's use mean to replace NA's
data$Age[is.na(data$Age)] <- mean(data$Age,na.rm=T)
missmap(data, main = "Missing values vs observed")

#### Better understand basline of categorical variables!
colnames(data)
is.factor(data$Survived)
is.factor(data$Pclass)
is.factor(data$Sex)
is.factor(data$Age)
is.factor(data$SibSp)
is.factor(data$Parch)
is.factor(data$Fare)
is.factor(data$Embarked)

### if read.csv did not recognize categories already, use as.factor() function. eg. data$Sex <- as.factor(data$Sex)

# Check which one is baseline according to contrast
# illustrate in model formula

contrasts(data$Sex)
contrasts(data$Embarked)

# finally, reset row names - VERY IMPORTANT otherwise loops won't work
rownames(data) <- NULL

#########################################

### fitting the model ###
#split the data for training and testing

#train <- data[1:800,]
#test <- data[801:889,]

set.seed(1)
train=data[sample(nrow(data),800),]
test=data[-as.numeric(rownames(train)),]


model <- glm(Survived ~.,family=binomial(link='logit'),data=train)
summary(model)
anova(model, test="Chisq")
### Sibsp is not t-significant but reduces residuals - could be included in final model

### R-squared
install.packages("pscl")
library(pscl)
pR2(model)

#### how good is the model in predicting from test data?

fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != test$Survived)
print(paste('Accuracy',1-misClasificError))

### Confusion matrix  ##

table(fitted.results,test$Survived)


