#####################################################################
# Introduction to R Programming                                     
# Hanisha Tatapudi (tatapudi@mail.usf.edu)
#####################################################################


## 1. ABOUT R & R STUDIO

  # R is a simple intuitive, interpreted language used for statistical computing. 

  # R code is usually stored in text files with the file ending in '.R' 

  # homework.R

  # Similar to Python and other programming languages, any line written in a program file is 
  #assumed to be a R statement, or part thereof.

  # The only exception is comment lines, which start with the character `#` (optionally preceded 
  #by an arbitrary number of white-space characters, i.e., tabs or spaces). 

  # Comment lines are used to define or explain the code and variables within for a better 
  #undertanding from the coders perspective.


  # Comment lines are usually ignored by the R interpreter.  It is not necessary to include them in 
  #your code.


  # You cannot make multi-line comments in R but you can stack multiple comments by using a  '#' 
  #symbol for each line. 


  # RStudio is an open source IDE for R which helps us interact with the program with great ease. It 
  #has all the functions and libraries that R supports and is free to download. 


  # To run an R program, select the lines of code you want to execute and press CTRL+ENTER or 
  #CMD+ENTER on your WINDOWS or MAC computers respectively.


  # You could also click on the Green Arrow or the Run Button on the upper right hand side of your 
  #R Script Editor.


## 2. LET'S GET STARTED WITH BASICS

  # In this session we will show off some off some of the cool stuff you can do in  without 
  #understanding anything about programming. Do not worry bout understanding everything the code does. Just enjoy!

  ## a) Basics 1

    #The following lines of code we will understand how to
    
    # 1. Load datasets
    # 2. Play wit data summary and stasistics
    # 3. Plot Stem and leaf plot and histogram

data()	        # browse pre-loaded data sets

data(rivers)	# get this one: "Lengths of Major North American Rivers"

ls()	        # notice that "rivers" now appears in the workspace

head(rivers)	# peek at the data set (gives first 6 elements of the dataset)
View(rivers) 

length(rivers)	  # how many rivers were measured?

summary(rivers)   # what are some summary statistics?
?summary
# make a stem-and-leaf plot (a histogram-like data visualization)

stem(rivers)      # make a stem-and-leaf plot (a histogram-like data visualization)

stem(log(rivers)) # Notice that the data are neither normal nor log-normal! Take that, Bell curve fundamentalists.

# make a histogram:

hist(rivers, col="#333333", border="white", breaks=25) # play around with these parameters

hist(log(rivers), col="#333333", border="white", breaks=25) # you'll do more plotting later

## b) Basics 2

# Let us toy with another dataset. Here's another neat data set that comes pre-loaded. 
#R has tons of these. 
# This section helps us revise the same basics and starts off with basic plot() function 
#and its alternatives. 

data(discoveries)  # loading data (Yearly numbers of important discoveries)

str(discoveries)   # displays the structure of the dataset

class(discoveries) # class of the dataset, here it is 'ts'

plot(discoveries, col="#333333", lwd=2, xlab="Year",  
     main="Number of important discoveries per year") # try different values for lwd

plot(discoveries, col="#333333", lwd=3, type = "h", xlab="Year",
     main="Number of important discoveries per year") # try type 'h', 'p' , 'o', 'n'

# Rather than leaving the default ordering (by year),
# we could also sort to see what's typical:

sort(discoveries) # arranges the vector in ascending order

stem(discoveries, scale=2) 

max(discoveries) # returns the maximum value from the vector

summary(discoveries)

### c) Basics 3

# In this section we will learn how to generate random numbers and why set.seed() is important.
# Roll a die a few times

round(runif(7, min=.5, max=6.5))

# Roll a die a few time (same results)
# Your numbers will differ from mine unless we set the same seed(31337)

set.seed(31337)

round(runif(7, min=.5, max=6.5))

# Roll a die a few time (same results)
# Try different seeds

set.seed(353)
round(runif(7, min=.5, max=6.5))

# Draw from a standard Gaussian 9 times
?rnorm()
rnorm(9) # random generation for the normal distribution, try it with set seed

##################################################
# Data types and basic arithmetic
##################################################

# Now for the programming-oriented part of the tutorial.
# In this section you will meet the important data types of R:
# integers, numerics, characters, logicals, and factors.
# There are others, but these are the bare minimum you need to
# get started.

# a) INTEGERS
# Long-storage integers are written with L
# Example

5L 
class(5L) 
?class


# (Try ?class for more information on the class() function.)
# In R, every single value, like 5L, is considered a vector of length 1
length(5L) # returns the length of any vector, try it for previous datasets

# You can have an integer vector with length > 1 too:

c(4L, 5L, 8L, 3L) 

length(c(4L, 5L, 8, 3L)) 

class(c(4L, 5L, 8L, 3L)) 
class(8)

# b) NUMERICS
# A "numeric" is a double-precision floating-point number

5 

class(5) 

# Again, everything in R is a vector;
# you can make a numeric vector with more than one element

c(3,3,3,2,2,1) # You can use scientific notation too
5e4 
6.02e23 # Avogadro's number
1.6e-35 # Planck length

# You can also have infinitely large or small numbers
class(Inf)  
class(-Inf) 

# You might use "Inf", for example, in integrate(dnorm, 3, Inf);

# c) BASIC ARITHMETIC

# You can do arithmetic with numbers
# Doing arithmetic on a mix of integers and numerics gives you another numeric

10L + 66L       # integer plus integer gives integer
53.2 - 4        # numeric minus numeric gives numeric
2.0 * 2L        # numeric times integer gives numeric
3L / 4          # integer over numeric gives numeric
3 %% 2          # the remainder of two numerics is another numeric


# Illegal arithmetic yeilds you a "not-a-number":
0 / 0 
class(NaN) 

# You can do arithmetic on two vectors with length greater than 1,
# so long as the larger vector's length is an integer multiple of the smaller
v=c(1:5)
v
v[2:4] #indexing subsetting

c(1,2,3) + c(1,2,3) 

# Since a single number is a vector of length one, scalars are applied 
# elementwise to vectors

(4 * c(1,2,3) - 2) / 2 

# Except for scalars, use caution when performing arithmetic on vectors with 
# different lengths. Although it can be done, 

c(1,2,3,1,2,3) * c(1,2) 

# Matching lengths is better practice and easier to read
c(1,2,3,1,2,3) * c(1,2,1,2,1,2) 


# d) CHARACTERS
# There's no difference between strings and characters in R

"Horatio" 
class("Horatio") 
class('H') 

# Those were both character vectors of length 1
# Here is a longer one:

c('alef', 'bet', 'gimmel', 'dalet', 'he')

length(c("Call","me","Ishmael")) 

# You can do regex operations on character vectors:

substr("Fortuna multis dat nimis, nulli satis.", 9, 15) #subsetting

gsub('u', 'ø', "Fortuna multis dat nimis, nulli satis.") #substitute

# R has several built-in character vectors:

letters

month.abb 

# e) LOGICALS

# In R, a "logical" is a boolean

class(TRUE) 
class(FALSE)  

# Their behavior is normal
f$name =='Ryan'

TRUE == TRUE  
TRUE == FALSE 
FALSE != FALSE
FALSE != TRUE 

# Missing data (NA) is logical, too

class(NA)   

# Use | and & for logic operations.

# OR

TRUE | FALSE    

# AND

TRUE & FALSE    

# Applying | and & to vectors returns elementwise logic operations

c(TRUE,FALSE,FALSE) | c(FALSE,TRUE,FALSE) 

c(TRUE,FALSE,TRUE) & c(FALSE,TRUE,TRUE) 

# You can test if x is TRUE

isTRUE(TRUE)    
?is.na() #test if there is a na in the col

# Here we get a logical vector with many elements:

c('Z', 'o', 'r', 'r', 'o') == "Zorro" 

c('Z', 'o', 'r', 'r', 'o') == "Z" 

# f) FACTORS

# The factor class is for categorical data
# Factors can be ordered (like childrens' grade levels) or unordered (like gender)

factor(c("female", "female", "male", NA, "female"))

# The "levels" are the values the categorical data can take
# Note that missing data does not enter the levels

levels(factor(c("male", "male", "female", NA, "female"))) 

# If a factor vector has length 1, its levels will have length 1, too

length(factor("male"))
length(levels(factor("male"))) 

# Factors are commonly seen in data frames, a data structure we will cover later

data(infert) # "Infertility after Spontaneous and Induced Abortion"
levels(infert$education) 

# g) NULL

# "NULL" is a weird one; use it to "blank out" a vector

class(NULL) 
parakeet = c("beak", "feathers", "wings", "eyes")
parakeet
parakeet <- NULL
parakeet



# h) TYPE COERCION

# Type-coercion is when you force a value to take on a different type
# Type-coercion is when you force a value to take on a different type. 
#There are two types of coercion in R (and in most programming languages)
# 1. Implicit coercion: when you combine two different data sets, 
#R has inbuilt rules to change the datatype
# 2. Explicit coercion: the user specifically mentions the change of a certain variable.

as.character(c(6, 8)) 
as.logical(c(1,0,1,1)) 

# If you put elements of different types into a vector, weird coercions happen:

c(TRUE, 4)
c("dog", TRUE, 4) 
as.numeric("Bilbo")


# Also note: those were just the basic data types
# There are many more data types, such as for dates, time series, etc.



##################################################
# Variables, loops, if/else
##################################################

# A variable is like a box you store a value in for later use.
# We call this "assigning" the value to the variable.
# Having variables lets us write loops, functions, and if/else statements


# VARIABLES

# Lots of way to assign stuff:

x = 5 # this is possible
y <- "1" # this is preferred
TRUE -> z # this works but is weird

# LOOPS

# We've got for loops
for (i in 1:4) {
  print(i)
}

# We've got while loops
a <- 10
while (a > 4) {
  cat(a, "...", sep = "")
  a <- a - 1
}

?cat() #Outputs the objects, concatenating the representations. cat performs much less conversion than print.

# Keep in mind that for and while loops run slowly in R
# Operations on entire vectors (i.e. a whole row, a whole column)
# or apply()-type functions (we'll discuss later) are preferred

# IF/ELSE

# Again, pretty standard
if (4 > 3) {
  print("4 is greater than 3")
} else {
  print("4 is not greater than 3")
}


# FUNCTIONS

# Defined like so:
jiggle <- function(x) {
  x = x + rnorm(1, sd=.1) #add in a bit of (controlled) noise
  return(x)
}

# Called like any other R function:
jiggle(5) 



###########################################################################
# Data structures: Vectors, matrices, data frames, and arrays
###########################################################################

# a) ONE-DIMENSIONAL

# Let's start from the very beginning, and with something you already know: vectors.

vec <- c(8, 9, 10, 11)
vec 

# We ask for specific elements by subsetting with square brackets
# (Note that R starts counting from 1)

vec[1]     
letters[18]
LETTERS[13]
month.name[9]   
c(6, 8, 7, 5, 3, 0, 9)[3]  

# We can also search for the indices of specific components,

which(vec %% 2 == 0)    

# grab just the first or last few entries in the vector,

head(vec, 1)    
tail(vec, 2)

# or figure out if a certain value is in the vector

any(vec == 10) 

# If an index "goes over" you'll get NA:

vec[6]  

# You can find the length of your vector with length()

length(vec) 

# You can perform operations on entire vectors or subsets of vectors

vec * 4 
vec[2:3] * 5    
any(vec[2:3] == 8) 

# and R has many built-in functions to summarize vectors

mean(vec)   
var(vec)    
sd(vec)     
max(vec)    
min(vec)    
sum(vec)    

# Some more nice built-ins:
5:15    # 5  6  7  8  9 10 11 12 13 14 15
seq(from=0, to=31337, by=1337)

# b) TWO-DIMENSIONAL (ALL ONE CLASS)

# You can make a matrix out of entries all of the same type like so:

mat <- matrix(nrow = 3, ncol = 2, c(1,2,3,4,5,6))
mat

# Unlike a vector, the class of a matrix is "matrix", no matter what's in it

class(mat) 

# Ask for the first row

mat[1,] 

# Perform operation on the first column

3 * mat[,1] 

# Ask for a specific cell

mat[3,2]    

# Transpose the whole matrix

t(mat)


# Matrix multiplication

mat %*% t(mat)


# cbind() sticks vectors together column-wise to make a matrix

mat2 <- cbind(1:4, c("dog", "cat", "bird", "dog"))
mat2

class(mat2)

# Again, note what happened!
# Because matrices must contain entries all of the same class,
# everything got converted to the character class

c(class(mat2[,1]), class(mat2[,2]))

# rbind() sticks vectors together row-wise to make a matrix

mat3 <- rbind(c(1,2,4,5), c(6,7,0,4))
mat3

# Ah, everything of the same class. No coercions. Much better.

# c) TWO-DIMENSIONAL (DIFFERENT CLASSES)

# For columns of different types, use a data frame
# This data structure is so useful for statistical programming,
# a version of it was added to Python in the package "pandas".

students <- data.frame(c("Cedric","Fred","George","Cho","Draco","Ginny"),
                       c(3,2,2,1,0,-1),
                       c("H", "G", "G", "R", "S", "G"))

names(students) <- c("name", "year", "house") # name the columns

class(students)

students

class(students$year) 
class(students[,3]) 

# find the dimensions

nrow(students)  

ncol(students)  

dim(students)   

# The data.frame() function converts character vectors to factor vectors
# by default; turn this off by setting stringsAsFactors = FALSE when
# you create the data.frame

?data.frame

# There are many twisty ways to subset data frames, all subtly unalike

data(USArrests)
df = USArrests

View(df)

class(df)

# To drop a column from a data.frame or data.table,
# assign it the NULL value

df[,2] = NULL


# Drop a row by subsetting
# Using data.table:

df[,1:3]

df[df$UrbanPop >= 58]
newdf <- df[ which(df$UrbanPop >=58),] 
newdf
nrow(newdf)

# general subsetting
                         
newdf1 <- df[ which(df$UrbanPop >= 48 & df$Assault > 100), ]
newdf1

length(newdf1)
nrow(newdf1)
View(newdf1)

#adding a column

newvec = c(1:38)
newdf1["S.no"] <- newvec
View(newdf1)

# subsetting random samples
mysample <- df[sample(1:nrow(df), 10, replace=FALSE),]

mysample

#subsetting using subset function
newdata <- subset(df, UrbanPop >= 40 & Murder >8, select=Murder:Rape)
View(newdata)

# d) MULTI-DIMENSIONAL (ALL ELEMENTS OF ONE TYPE)

# Arrays creates n-dimensional tables
# All elements must be of the same type
# You can make a two-dimensional table (sort of like a matrix)

array(c(c(1,2,4,5),c(8,9,3,6)), dim=c(2,4))

# You can use array to make three-dimensional matrices too
array(c(c(c(2,300,4),c(8,9,0)),c(c(5,60,0),c(66,7,847))), dim=c(3,2,2))


# e) LISTS (MULTI-DIMENSIONAL, POSSIBLY RAGGED, OF DIFFERENT TYPES)

# Finally, R has lists (of vectors)

list1 <- list(time = 1:40)
list1$price = c(rnorm(40,.5*list1$time,4)) # random
list1

# You can get items in the list like so

list1$time # one way
list1[["time"]] # another way
list1[[1]] # yet another way

# You can subset list items like any other vector

list1$price[4]

# Lists are not the most efficient data structure to work with in R;
# unless you have a very good reason, you should stick to data.frames
# Lists are often returned by functions that perform linear regressions

##################################################
# The apply() family of functions
##################################################

# Remember mat?
mat

# Use apply(X, MARGIN, FUN) to apply function FUN to a matrix X
# over rows (MAR = 1) or columns (MAR = 2)
# That is, R does FUN to each row (or column) of X, much faster than a
# for or while loop would do

apply(mat, MAR = 2, jiggle)

# Other functions: ?lapply, ?sapply

# Don't feel too intimidated; everyone agrees they are rather confusing

# The plyr package aims to replace (and improve upon!) the *apply() family.

install.packages("plyr")
require(plyr)
?plyr



#########################
# Loading data
#########################

# "pets.csv" is a file on the internet
# (but it could just as easily be be a file on your own computer)

pets <- read.csv("http://learnxinyminutes.com/docs/pets.csv")
pets
head(pets, 2) # first two rows
tail(pets, 1) # last row

# To save a data frame or matrix as a .csv file
write.csv(pets, "pets2.csv") # to make a new .csv file
# set working directory with setwd(), look it up with getwd()

# Try ?read.csv and ?write.csv for more information


#########################
# Plots
#########################

# a)BUILT-IN PLOTTING FUNCTIONS

# Scatterplots!
plot(list1$time, list1$price, main = "fake data")

# Histograms!
hist(rpois(n = 10000, lambda = 5), col = "thistle")

# Barplots!
barplot(c(1,4,5,1,2), names.arg = c("red","blue","purple","green","yellow"))

# GGPLOT2

# But these are not even the prettiest of R's plots
# Try the ggplot2 package for more and better graphics

install.packages("ggplot2")
library(ggplot2)
require(ggplot2)

pp <- ggplot(students, aes(x=house))
pp + geom_histogram()
ll <- as.data.table(list1)
pp <- ggplot(ll, aes(x=time,price))
pp + geom_point()

# ggplot2 has excellent documentation (available http://docs.ggplot2.org/current/)