library(caret)
library(readr)
library(rpart)
library(randomForest)
library(rpart.plot)
library(rattle)
library(ggplot2)
library(parallel)
library(doParallel)
library(foreach)
library(doSNOW)
library(gt)
X <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/HW-1/MADELON/madelon_train.data", 
                            col_names = FALSE)
Y <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/HW-1/MADELON/madelon_train.labels", 
                                 col_names = FALSE)
x <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/HW-1/MADELON/madelon_valid.data", 
                col_names = FALSE)
y <- read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/HW-1/MADELON/madelon_valid.labels", 
                col_names = FALSE)

X=X[,-501]
x=x[,-501]
mad_train = data.frame(Y,X)
attach(mad_train)
mad_valid = data.frame(y,x)
attach(mad_valid)
###################
#decision tree


## parallel computing Using Apply
# Number of cores to use
n.cores <- detectCores() - 1
#create the cluster
my.cluster <- makeCluster(n.cores)
#register it to be used by %dopar%
registerDoParallel(my.cluster)
clusterEvalQ(my.cluster, 
             {library(caret)
               library(readr)
               library(rpart)
               library(randomForest)
               library(rpart.plot)
               library(rattle)
               library(parallel)})

DT <- function(n,train,test){
  fit <- rpart(X1 ~., data = train,  maxdepth = n ,  method = 'class' )
  g=data.frame(fit$cptable)
  pred <- predict(fit,test, type = 'class' )
  confMat <- table(test[,1],pred)
  accuracy = round(sum(diag(confMat))/sum(confMat),3)
  miss=1-accuracy
  output=list(n,tail(g$nsplit,1),tail(g$xerror,1),accuracy,miss)
  return(output)
}

num_tree <- seq(1,12)

## train Data
Result_train = t(parSapply(my.cluster,num_tree,DT,train=mad_train,test=mad_train))
colnames(Result_train)=c("Tree Size","Tree Depth","Minimum test error","Accuracy","Missclassification")
Result_train
miss_train_DT<- unlist(Result_train[,5])

## valid data
Result_valid = t(parSapply(my.cluster,num_tree,DT,train=mad_train,test=mad_valid))
colnames(Result_valid)=c("Tree Size","Tree Depth","Minimum test error","Accuracy","Missclassification")
Result_valid
miss_valid_DT<- unlist(Result_valid[,5])

plot(num_tree,miss_train_DT,type = "b",col="red",ylim = c(0.1,0.4),xlab = "Number of Trees", ylab = "Misclassifaction error",main = "Decesion Tree")
lines(miss_valid_DT,col="blue",type = "b")
abline(a=0.22,b=0,col="green")
legend(8,0.3,legend=c("training","validation","cut-off"),lty=c(1,1,1),pch=c("o","o"," ") ,col=c("red","blue","green"))
################################

#2
#####

#Train data
sat_train <- data.frame(read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/HW-1/satimage/sat.trn",                       
                      col_names = FALSE))
attach(sat_train)
#Test Data
sat_test <-  data.frame(read_table("D:/OneDrive - Florida State University/MyFSU_OneDrive/FSU Course Work/5635/HW-1/satimage/sat.tst", 
                       col_names = FALSE))
attach(sat_test)
str(sat_train)
#Function
DT <- function(n,train,test){
  fit <- rpart(X37 ~., data = train,  maxdepth = n ,  method = 'class' )
  g=data.frame(fit$cptable)
  pred <- predict(fit,test, type = 'class' )
  confMat <- table(test[,37],pred)
  accuracy = round(sum(diag(confMat))/sum(confMat),3)
  miss=1-accuracy
  output=list(n,tail(g$nsplit,1),tail(g$xerror,1),accuracy,miss)
  return(output)
}

## train Data
Result_train = t(parSapply(my.cluster,num_tree,DT,train=sat_train,test=sat_train))
colnames(Result_train)=c("Tree Size","Tree Depth","Minimum test error","Accuracy","Missclassification")
Result_train
miss_train_DT<- unlist(Result_train[,5])

## valid data
Result_valid = t(parSapply(my.cluster,num_tree,DT,train=sat_train,test=sat_test))
colnames(Result_valid)=c("Tree Size","Tree Depth","Minimum test error","Accuracy","Missclassification")
Result_valid
miss_valid_DT<- unlist(Result_valid[,5])



plot(num_tree,miss_train_DT,type = "b",col="red",ylim = c(0.1,0.6),xlab = "Number of Trees", ylab = "Misclassifaction error",main = "Decesion Tree")
lines(miss_valid_DT,col="blue",type = "b")
abline(a=0.25,b=0,col="green")
legend(8,0.5,legend=c("training","validation","cut-off"),lty=c(1,1,1),pch=c("o","o"," ") ,col=c("red","blue","green"))

#Random Forest

registerDoParallel(n.cores <- detectCores() - 1)
clusterEvalQ(my.cluster, 
             {library(caret)
               library(readr)
               library(rpart)
               library(randomForest)
               library(rpart.plot)
               library(rattle)
               library(parallel)
               library(doParallel)
               library(foreach)
               library(doSNOW)
               library(gt)})



#RF(10,30,mad_train,mad_valid)

n_feature = c(sqrt(500),log(500),500)

RF_out <- function(n_feature,train,test){
k = c(3,10,30,100,300) 
RF= function(n_tree,n_subset,train,test){
  rf = randomForest(as.factor(X1) ~., data = train , mtry=n_subset,  ntree = n_tree )
  pred  = predict(rf,test,type = "class")
  confMat <- table(test[,1],pred)
  accuracy = round(sum(diag(confMat))/sum(confMat),3)
  miss=1-accuracy
  return(miss)
}
my.cluster1 <- makeCluster(detectCores() - 1)
#register it to be used by %dopar%
registerDoParallel(my.cluster1)
clusterEvalQ(my.cluster1, 
             {library(caret)
               library(readr)
               library(rpart)
               library(randomForest)
               library(rpart.plot)
               library(rattle)
               library(parallel)
               library(doParallel)
               library(ggplot2)
               })

miss_train_RF <- parSapply(my.cluster1,k,RF,n_feature,train,train)
miss_valid_RF <- parSapply(my.cluster1,k,RF,n_feature,train,test)
output=data.frame(k,miss_train_RF,miss_valid_RF)
names(output) = c("Num_Tree","Train_Miss","Test_Miss")
color = c("Train Data" = "red","Test Data" = "blue")
p=ggplot(output,aes(x=k))+
  geom_line(aes(y=miss_train_RF,color = "Training"))+
  geom_line(aes(y=miss_valid_RF,color = "Validation"))+
  xlab('Number of Trees')+ylab('Misclassifaction error')
return(list(p,output))
}

V=parLapply(my.cluster,n_feature,RF_out,mad_train,mad_valid)
V[[1]]


## train Data
Result_train = sapply(num_tree,DT,train=d_train,test=d_train)
avg_train_mse = mean(unlist(Result_train['train_mse',]))
avg_test_mse = mean(unlist(Result_train['test_mse',]))
avg_train_R.sq = mean(unlist(Result_train['train_R.sq',]))
avg_train_R.sq = mean(unlist(Result_train['train_R.sq',]))












