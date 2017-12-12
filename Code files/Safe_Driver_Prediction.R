getwd() #current working directory
setwd("C:/Users/janne/Desktop/Edwisor/Project 2") #setting the custom wordking directory
list.files() #listing the files in current working directory
rm(list = ls()) #clearing the environment

#Loading the required libraries

library(dplyr) #data manipulation
library(readr) #input/output
library(data.table) #data manipulation
library(stringr) #string manipulation
library(caret)  #model evaluation (confusion matrix)
library(tibble) #data wrangling
library("ROSE") #over/under sampling
library("randomForest") #random forest model building
library(pROC) #ROC plots
library("MLmetrics") #Normalized Gini

#Loading the given data

train = as.tibble(fread("train.csv",na.strings = c("-1","-1.0"))) #given train data
test = as.tibble(fread("test.csv",na.strings = c("-1","-1.0"))) #given test data
#str(train);str(test)
dim(train);dim(test) #dimensions of train and test
table(train$target) #examining the target variable

#Combining the test and train data for pre-processing

test$target = 0 #creating a target variable in test data
test$data = "test" #creating another variable to identify the test data rows
test = test[, c(1, 60, 59, 2:58)] #reforming with newly created variables

train$data = "train" #creating another variable to identify the train data rows
train = train[, c(1, 60, 2:59)] #reforming with newly created variables

combined_data = as.data.frame(rbind(train,test)) #combining test and train data
dim(combined_data) #dimensions of combined data
rm(train,test) #removing train and test to save RAM
#names = names(train)

#Pre-processing and Feature Engineering

#structure of given data says, the variables are not as decribed as in data descripstion.
#we have to change the data types of the variables according to the data description
#I have made a .csv file with the variable names as one column and their respective data
#type according to the data description. Have a look at it after loading

var_groups = fread("var_groups.csv") #loading the self made csv file
head(var_groups) #head of the var_groups

#finding the intersecting names between combined_data and var_groups
names = intersect(colnames(combined_data), var_groups[["names"]])
#Changing the data types according to the data description
for(var_name in names){
        var_type = subset(var_groups, names %in% var_name, select=type)
        if(var_type == "numeric")
                combined_data[,var_name] = as.numeric(combined_data[,var_name])
        else if(var_type == "binary" || var_type == "categorical")
                combined_data[,var_name] <- as.factor(combined_data[,var_name])
        else if(var_type == "ordinal")
                combined_data[,var_name] <- as.ordered(combined_data[,var_name])
}
rm(var_groups,names,var_type,var_name) #removing un-neccesary things to save RAM

#lets look at the missing values of the data

#dataframe of missing values column wise
missing_values = as.data.frame(colSums(is.na(combined_data))) 
missing_values;rm(missing_values) #viewing and removing missing values

#there are columns with missing values over one lakhs. We will remove the columns with 
#more than 5% of missing values and impute the others

#vectordrop is a vector with columns having more than 5% of missing values
vectordrop <- combined_data[, lapply( combined_data, 
                                      function(m) sum(is.na(m)) / length(m) ) >= .05 ]
#removing the columns in vectordrop from the main data
combined_data = combined_data[,!(colnames(combined_data) %in% colnames(vectordrop))]
dim(combined_data);rm(vectordrop) 

#Now lets impute remaining columns with missing data

miss_pct <- sapply(combined_data, function(x) { sum(is.na(x)) / length(x) })
miss_pct <- miss_pct[miss_pct > 0]
names(miss_pct) #columns with missing data
rm(miss_pct)

#Designing a function to impute factor columns with mode.
mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}

#imputing NAs in factor columns with mode and in numerical columns with Mean.

df = combined_data #just making typing easy :P
rm(combined_data) #saving RAM
df$ps_ind_02_cat[is.na(df$ps_ind_02_cat)]<-mode(df$ps_ind_02_cat) #imputing with mode
df$ps_ind_04_cat[is.na(df$ps_ind_04_cat)]<-mode(df$ps_ind_04_cat)
df$ps_ind_05_cat[is.na(df$ps_ind_05_cat)]<-mode(df$ps_ind_05_cat)
df$ps_car_01_cat[is.na(df$ps_car_01_cat)]<-mode(df$ps_car_01_cat)
df$ps_car_02_cat[is.na(df$ps_car_02_cat)]<-mode(df$ps_car_02_cat)
df$ps_car_07_cat[is.na(df$ps_car_07_cat)]<-mode(df$ps_car_07_cat)
df$ps_car_09_cat[is.na(df$ps_car_09_cat)]<-mode(df$ps_car_09_cat)
df$ps_car_11[is.na(df$ps_car_11)]<-mode(df$ps_car_11)
df$ps_car_12[is.na(df$ps_car_12)]<-mean(df$ps_car_12,na.rm=T) #imputing with mean

#checking for missing values
sum(is.na(df))

#col_levels <- lapply(df, function(x) nlevels(x))
#col_levels

#Drop unused levels
#all_data <- droplevels.data.frame

#Forming train and test sets as before after the pre-processing

TRAIN <- df[1:595212,-2] #train data set after pre-processing
dim(TRAIN)
TEST <- df[595213:1488028,-c(2,3)] #test data set after pre-processing
dim(TEST)
rm(df) #removing df(combined data)

#Analyzing the target variables

table(TRAIN$target)

table(TRAIN$target)/nrow(TRAIN) #Here we observe that target variable is inbalanced, 
#class 0 have 0.963% observations & 1 have 0.037% observations.

#So,for better model to be build target variable should be balanced. 
#we using over and undersampling from ROSE package for this
balanced_train <- ovun.sample(target~.,data=TRAIN,method = "both",N = 90000,p=.5,seed=1)$data
head(balanced_train) #head of the dataset with balanced target
dim(balanced_train) #dimensions of the dataset with balanced target.

#Above we have sampled out only 90000 rows because, using over/undersampling might effect 
#the contributing variables. SO, it is better to take out a sample.It saves computational power too

sum(is.na(balanced_train)) #checking for any Missing values
balanced_train <-as.data.frame(balanced_train)

table(balanced_train$target) #analyzing the target variable in this dataset
table(balanced_train$target)/nrow(balanced_train) # Now data is balanced
str(balanced_train);rm(TRAIN) 

#Designing the model

#to calculate the accuracy we are taking out a train and test sets out of TRAIN dataset

#Now we split above sample data into train and test data ##
s=sample(nrow(balanced_train),round(nrow(balanced_train)*0.7),replace=FALSE)
train = balanced_train[s,]
test = balanced_train[-s,]
dim(train);dim(test)
rm(balanced_train)

#I have experimented with Naive Bayes, Gradient Boosting before Random Forest.
#They are giving me an accuracy between 55-58. RF gave 85%. So my base modek is Random Forest.

#Random Forests predicts numerical data very well, let us try this out

#Random forest will not take a variable with more than 53 categories, 
#so we remove variable "ps_car_11_cat" 

train_rf = subset(train,select=-ps_car_11_cat);rm(train) #train without ps_car_11_cat
test_rf = subset(test,select=-ps_car_11_cat);rm(test) #test without ps_car_11_cat

library("randomForest")
model_rf = randomForest(as.factor(target) ~. , data = train_rf) # Fit Random forest
summary(model_rf)

#png("Model_RF.png")
plot(model_rf, main = "Model_RF") #Plot for number of trees and error
#dev.off()

#png("Var_imp.png")
varImpPlot(model_rf) #Plot for Important variable
#dev.off()

pred_rf <- predict(model_rf,test_rf) #predictin using the model
summary(pred_rf)

#accuracy of the model
confusionMatrix(pred_rf,test_rf$target)

#library(pROC)
#ROC plot for the model
aucrf <- roc(as.numeric(test_rf$target), as.numeric(pred_rf),  ci=TRUE)
#png("ROC.png")
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')
#dev.off()

#Lets try to tune the model to increase the accuracy by including only important variables
var_imp = data.frame(importance(model_rf,type=2)) #variable imp as data frame
var_imp$Variables = row.names(var_imp)
var_imp[order(var_imp$MeanDecreaseGini,decreasing = T),]

d = importance(model_rf)

#creating dataset with imp variable
train_with_imp = subset(train_rf,select=c(1,(which(d>200))+1))
test_with_imp = subset(test_rf,select=c(1,(which(d>200))+1))
rm(test_rf,train_rf,var_imp,aucrf,d,s,mode,model_rf,pred_rf)

#model with important variables
model_rf2 <- randomForest(as.factor(target) ~. , data = train_with_imp,keep.forest=TRUE)

#png("Model_RF2.png")
plot(model_rf2, main = "Model_RF with imp variables ")
#dev.off()

importance(model_rf2) #variable importance of model

#png("Var_imp_rf2.png")
varImpPlot(model_rf2,sort = T,main="Variable Importance")
#dev.off()

#prediction using model 2
pred_rf2 <- predict(model_rf2,test_with_imp)
summary(pred_rf2)

#accuracy of the model
confusionMatrix(pred_rf2,test_with_imp$target)

#ROC plot for the model
aucrf <- roc(as.numeric(test_with_imp$target), as.numeric(pred_rf2),  ci=TRUE)
#png("ROC2.png")
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')
#dev.off()
rm(aucrf) #saving RAM

#library("MLmetrics")
# NormalizedGini(y_pred, y_true)

#Normalized Gini is 
NormalizedGini(as.numeric(pred_rf2),as.numeric(test_with_imp$target))

# Prediction for  Original TEST data 
#TEST_rf = TEST[,-ps_car_11_cat]
pred_rf_TEST <- predict(model_rf2,TEST,type="prob")

head(pred_rf_TEST)

summary(pred_rf_TEST)

final_data = data.frame(TEST$id ,pred_rf_TEST[,2]) #Prob of driver will claim insurance
sum(is.na(final_data)) #checking for missing values
colnames(final_data) <- c("id", "target") 
rownames(final_data) = NULL #just changing the indec
head(final_data);tail(final_data)

#Final submission file in mentioned format.

write.csv(final_data,file="Final_submission.csv")

#++++++++++++++++++++++++++Experimenting with models++++++++++++++++++++++++++++++

#Below are the experiments done by me with different models. If you want to execute please 
#do not remove TRAIN or else "train" and "test" subsetted from TRAIN. I removed them to free
#RAM. 

#taking out the target variable, just to avoid clumpsyness
x_train = train
x_test = test
y_train = x_train$target
y_test = x_test$target
x_train$target = NULL
x_test$target = NULL
dim(x_train);dim(x_test)
length(y_train);length(y_test)

#NAIVE BAYES
library(e1071)
library(caret)
x_cat = cbind(x_train,y_train)
fit = naiveBayes(y_train ~., data = x_cat)
pred = predict(fit,x_test)
summary(pred)
confusionMatrix(pred,y_test)

#Gradient Boosting & Adaboosting
fitControl = trainControl(method = "repeatedcv",number = 4,repeats = 4)
fit_gbm = train(y_train ~., data = x_cat,method = "gbm",trControl = fitControl)
pred_gbm = predict(fit_gbm,x_test)
summary(pred_gbm)
confusionMatrix(pred_gbm,y_test)

#Random Forests
##Random forest not take variable more than 53 categories, so we remove variable "_car_11_cat" 
train_rf = subset(train,select=-ps_car_11_cat);rm(train)
test_rf = subset(test,select=-ps_car_11_cat);rm(test)

library("randomForest")
model_rf = randomForest(as.factor(target) ~. , data = train_rf) # Fit Random forest
summary(model_rf)

### predict
pred_rf <- predict(model_rf,test_rf)
summary(pred_rf)
confusionMatrix(pred_rf,test_rf$target)

library(pROC)
aucrf <- roc(as.numeric(test_rf$target), as.numeric(pred_rf),  ci=TRUE)
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')


plot(model_rf, main = "Model") #Plot for number of trees and error

varImpPlot(model_rf) #Plot for Important variable

var_imp = data.frame(importance(model_rf,type=2))
var_imp$Variables = row.names(var_imp)
var_imp[order(var_imp$MeanDecreaseGini,decreasing = T),]

d = importance(model_rf)
# The importance routine in r for random forest models gives us the mean decrease gini value. Higher the Mean Decrease Gini value, more important the variable.
importance(model_rf)

##create dataset with IMp variable ####
train_with_imp = subset(train_rf,select=c(1,(which(d>200))+1))
test_with_imp = subset(test_rf,select=c(1,(which(d>200))+1))
rm(test_rf,train_rf,var_imp,aucrf,d)


model_rf2 <- randomForest(as.factor(target) ~. , data = train_with_imp,keep.forest=TRUE)
plot(model_rf2, main = "Model ")
importance(model_rf2)
varImpPlot(model_rf2)
varImpPlot(model_rf2,sort = T,main="Variable Importance")

pred_rf2 <- predict(model_rf2,test_with_imp)
summary(pred_rf2)

### score prediction using AUC
confusionMatrix(pred_rf2,test_with_imp$target)


library(pROC)
# AUC value for Model is
aucrf <- roc(as.numeric(test_with_imp$target), as.numeric(pred_rf2),  ci=TRUE)
plot(aucrf, ylim=c(0,1), print.thres=TRUE, main=paste('Random Forest AUC:',round(aucrf$auc[[1]],3)),col = 'blue')
rm(aucrf,model_rf,pred_rf)














