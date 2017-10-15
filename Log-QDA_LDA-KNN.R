###: KNN


?iris
names(iris)
dim(iris) # 150 5
rm(iris)

summary(iris) # note: regularize
attach(iris)
cor(iris[,-5])

pairs(iris[,-5], main="Iris Data", 
      pch=21, bg=c("Red","Blue","Green")[unclass(iris$Species)])
plot(iris$Sepal.Length,iris$Sepal.Width, pch=21, bg=c("Red","Blue","Green"))
# linearity between species

###: All predictors
standardized.X<-scale(iris[,-5])
set.seed(123)
library(caret)
train<-createDataPartition(iris$Species, list=FALSE)
train.X<-standardized.X[train,-5]
train.y<-iris[train,5]
test.X<-standardized.X[-train,-5]
test.y<-iris[-train,5]

###: K=1
knn.pred1<-knn(train.X,test.X,train.y,k=1)
table(knn.pred,test.y)
mean(knn.pred!=test.y) # test error of 6.67%
###: K=3
knn.pred3<-knn(train.X,test.X,train.y,k=3)
mean(knn.pred!=test.y) # test error of 6.67% again
###: K=5
knn.pred5<-knn(train.X,test.X,train.y,k=5)
mean(knn.pred!=test.y) # test error of 5.33%

#: k-nn with k=5: ~95.7% accuracy

versicolor<- 1 - (25 / (25 + 0 + 4)) # 13.8% error rate (all errors from versicolor)

###: Petal.Length and Petal.Width
plot(Petal.Length,Petal.Width,main="Petal Length vs. Petal Width",
     pch=21,bg=c("Red","Blue","Green")[unclass(Species)])
knn.pred<-knn(train.X[,c(3,4)],test.X[,c(3,4)],train.y,k=1)
table(knn.pred,test.y)
mean(knn.pred==test.y) # k=1: ~97.3% accuracy
knn.pred<-knn(train.X[,c(3,4)],test.X[,c(3,4)],train.y,k=3)
table(knn.pred,test.y)
mean(knn.pred==test.y) # k=3: ~94.67% accuracy, less flexible model performed worse

###: Sepal.Length and Sepal.Width (least-linear predictors)
plot(Sepal.Length,Sepal.Width,main="Sepal Length vs. Sepal Width",
     pch=21,bg=c("Red","Blue","Green")[unclass(Species)])
knn.pred<-knn(train.X[,c(1,2)],test.X[,c(1,2)],train.y,k=1)
table(knn.pred,test.y)
mean(knn.pred==test.y) # k=1: ~70.7% accuracy
knn.pred<-knn(train.X[,c(1,2)],test.X[,c(1,2)],train.y,k=5)
mean(knn.pred==test.y) # k=5 had best accuracy ~ 77.3% -> less flexible model outperforms
# k>5 increase in error rate


###: LDA-QDA


library(MASS)
attach(iris)
pairs(iris[,-5], main="Iris Data", 
      pch=21, bg=c("Red","Blue","Green")[unclass(iris$Species)])
cor(iris[,-5]) # S.L-P.L, S.L-P.W, P.L-P.W
set.seed(123)
library(caret)
train<-createDataPartition(iris$Species, list=FALSE)
train.X<-iris[train,-5]
train.y<-iris[train,5]
test.X<-iris[-train,-5]
test.y<-iris[-train,5]

# density testing
par(mfrow=c(2,2))
plot(density(Sepal.Length)) 
plot(density(Sepal.Width))
plot(density(Petal.Length)) 
plot(density(Petal.Width))
# S.L and S.W approx. gaussian, P.L and P.W not normal

###: LDA

#: All preds
contrasts(Species)
lda.fit <- lda(Species ~ .,data=iris, subset=train)
lda.fit
par(mfrow=c(1,1))
plot(lda.fit)
lda.pred<-predict(lda.fit,test.X)
lda.pred
lda.class<-lda.pred$class
table(lda.class,test.y)
mean(lda.class==test.y) # 96% accuracy

#: Sepal.Length and Sepal.Width
lda.fit<-lda(Species~Sepal.Length+Sepal.Width,
             data=iris, subset=train)
plot(lda.fit)
lda.predict<-predict(lda.fit,test.X)
lda.class<-lda.predict$class
table(lda.class,test.y)
mean(lda.class==test.y) 
# 81.33% accuracy - expected non-parametic KNN to outperform as data is non-linear but error-rate of lda<knn
# approx. gaussian may have strengthened model

#: Petal.Length and Petal.Width
lda.fit<-lda(Species~Petal.Length+Petal.Width,
             data=iris, subset=train)
plot(lda.fit)
lda.predict<-predict(lda.fit,test.X)
lda.class<-lda.predict$class
mean(lda.class==test.y) # 93.3% accuracy
# highly linear data could have offset non-normality of distributions

###: QDA

#: All Preds
qda.fit<-qda(Species~.,data=iris,subset=train)
qda.class<-predict(qda.fit,test.X)$class
mean(qda.class==test.y) # 94.67% accuracy

#: Sepal.Length and Sepal.Width
qda.fit<-qda(Species~Sepal.Length+Sepal.Width,
             data=iris, subset=train)
qda.class<-predict(qda.fit,test.X)$class
mean(qda.class==test.y) # 81.33% accuracy
# same as lda on S.L and S.W - interesting as QDA performs better with few obs
# also, trade-off for potentially quadratic boundary if offset by non-linearity
table(qda.class,test.y)
# we still see Setosa is classified perfectly because of boundary properties!

#: Petal.Length and Petal.Width
qda.fit<-qda(Species~Petal.Length+Petal.Width,
             data=iris, subset=train)
qda.class<-predict(qda.fit,test.X)$class
mean(qda.class==test.y) # 97.33% accuracy
# higher accuracy than lda, potentially better performance due to not many obs

###: QDA has highest accuracy prediction on Petal.Length vs. Petal.Width to classify
###: iris species


###: Log Reg



library(MASS)
attach(iris)
pairs(iris[,-5], main="Iris Data", 
      pch=21, bg=c("Red","Blue","Green")[unclass(iris$Species)])
cor(iris[,-5]) # S.L-P.L, S.L-P.W, P.L-P.W
set.seed(123)
library(caret)
train<-createDataPartition(iris$Species, list=FALSE)
train.X<-iris[train,-5]
train.y<-iris[train,5]
test.X<-iris[-train,-5]
test.y<-iris[-train,5]

# density testing
par(mfrow=c(2,2))
plot(density(Sepal.Length)) 
plot(density(Sepal.Width))
plot(density(Petal.Length)) 
plot(density(Petal.Width))
# S.L and S.W approx. gaussian, P.L and P.W not normal

###: LDA

#: All preds
contrasts(Species)
lda.fit <- lda(Species ~ .,data=iris, subset=train)
lda.fit
par(mfrow=c(1,1))
plot(lda.fit)
lda.pred<-predict(lda.fit,test.X)
lda.pred
lda.class<-lda.pred$class
table(lda.class,test.y)
mean(lda.class==test.y) # 96% accuracy

#: Sepal.Length and Sepal.Width
lda.fit<-lda(Species~Sepal.Length+Sepal.Width,
             data=iris, subset=train)
plot(lda.fit)
lda.predict<-predict(lda.fit,test.X)
lda.class<-lda.predict$class
table(lda.class,test.y)
mean(lda.class==test.y) 
# 81.33% accuracy - expected non-parametic KNN to outperform as data is non-linear but error-rate of lda<knn
# approx. gaussian may have strengthened model

#: Petal.Length and Petal.Width
lda.fit<-lda(Species~Petal.Length+Petal.Width,
             data=iris, subset=train)
plot(lda.fit)
lda.predict<-predict(lda.fit,test.X)
lda.class<-lda.predict$class
mean(lda.class==test.y) # 93.3% accuracy
# highly linear data could have offset non-normality of distributions

###: QDA

#: All Preds
qda.fit<-qda(Species~.,data=iris,subset=train)
qda.class<-predict(qda.fit,test.X)$class
mean(qda.class==test.y) # 94.67% accuracy

#: Sepal.Length and Sepal.Width
qda.fit<-qda(Species~Sepal.Length+Sepal.Width,
             data=iris, subset=train)
qda.class<-predict(qda.fit,test.X)$class
mean(qda.class==test.y) # 81.33% accuracy
# same as lda on S.L and S.W - interesting as QDA performs better with few obs
# also, trade-off for potentially quadratic boundary if offset by non-linearity
table(qda.class,test.y)
# we still see Setosa is classified perfectly because of boundary properties!

#: Petal.Length and Petal.Width
qda.fit<-qda(Species~Petal.Length+Petal.Width,
             data=iris, subset=train)
qda.class<-predict(qda.fit,test.X)$class
mean(qda.class==test.y) # 97.33% accuracy
# higher accuracy than lda, potentially better performance due to not many obs

###: QDA has highest accuracy prediction on Petal.Length vs. Petal.Width to classify
###: iris species

