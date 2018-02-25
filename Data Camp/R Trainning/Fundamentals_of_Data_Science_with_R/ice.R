rm(list=ls())
getwd()
setwd("C:/Users/nsakib/Desktop/game1")

data=as.data.frame(read.table(file="ice.txt",header=FALSE))
data=data[,-1]
data=data[,c(1,3,4)]
colnames(data)=c("cons","inc","temp")

attach(data)

#have a look at the data
plot(cons,col="blue",xlab="Observation",ylab="Consumption (pints per person)",pch=20,cex=2)
plot(inc,col="red",xlab="Observation",ylab="Weekly Family Income ($)",pch=20,cex=2)
plot(temp,col="darkgreen",xlab="Observation",ylab="Temperature (F)",pch=20,cex=2)

hist(cons,col="skyblue",xlab="Consumption (pints per person)")
hist(inc,col="red",xlab="Weekly Family Income ($)")
hist(temp,col="darkgreen",xlab="Temperature (F)")


#scatterplot
cor(data)
pairs(~cons+inc+temp,data=data,main="Scatterplot",pch=20,cex=3.5,col="deepskyblue4")

#simple linear regression
reg1<-lm(cons~temp,data = data)
summary(reg1)
anova(reg1)
plot(temp,cons,col="blue",pch=19)
abline(reg1,col="red")



#plot in 3d
install.packages("rgl")
library(rgl)
plot3d(inc,temp,cons, type="s", col="red", site=1)

#multivariate linear regression
reg2<-lm(cons~inc+temp)
summary(reg2)
anova(reg2)


#comparison
anova(reg1,reg2)

#post analysis
plot(cons,reg2$residuals,col="red",pch=19)
abline(h=0)


### regression plane
install.packages("rockchalk")
library(rockchalk)
plotPlane(reg2, plotx1 = "inc", plotx2 = "temp",pcol="red",pch=20,pcex=2)

detach(data)
