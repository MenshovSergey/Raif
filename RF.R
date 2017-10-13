setwd("/home/sergej/Programming/CS/R/Raif")
credit.01 <- read.table("union.csv", header=T, sep=",")

set.seed(123)
library(rpart)
credit.01.res <- rpart(bad ~ .,
                             data = credit.01, method="class",
                             control=rpart.control(minsplit=10, minbucket=5, maxdepth=20) )
credit.01.res
library(rpart.plot)
rpart.plot(credit.01.res, type=2, extra = 1)
