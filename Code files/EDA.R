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
library(ggplot2) #data visualization
library(reshape2)


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
#rm(train,test) #removing train and test to save RAM

library("cowplot")

a = combined_data %>%
        ggplot(aes(ps_ind_06_bin, fill = ps_ind_06_bin)) +
        geom_bar() + 
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_ind_07_bin, fill = ps_ind_07_bin)) +
        geom_bar() +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_ind_08_bin, fill = ps_ind_08_bin)) +
        geom_bar() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_ind_09_bin, fill = ps_ind_09_bin)) +
        geom_bar() +
        theme(legend.position = "none")

#png("Binary_Variables_1.png")
plot_grid(a,b,c,d + rremove("x.text"),
          ncol = 2, nrow = 2)
#dev.off()
rm(a,b,c,d)

a = combined_data %>%
        ggplot(aes(ps_ind_10_bin, fill = ps_ind_10_bin)) +
        geom_bar() +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_ind_11_bin, fill = ps_ind_11_bin)) +
        geom_bar() +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_ind_12_bin, fill = ps_ind_12_bin)) +
        geom_bar() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_ind_13_bin, fill = ps_ind_13_bin)) +
        geom_bar() +
        theme(legend.position = "none")

#png("Binary_Variables_2.png")
plot_grid(a,b,c,d + rremove("x.text"),
          ncol = 2, nrow = 2)
#dev.off()
rm(a,b,c,d)

a = combined_data %>%
        ggplot(aes(ps_ind_16_bin, fill = ps_ind_16_bin)) +
        geom_bar() +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_ind_17_bin, fill = ps_ind_17_bin)) +
        geom_bar() +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_ind_18_bin, fill = ps_ind_18_bin)) +
        geom_bar() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_calc_15_bin, fill = ps_calc_15_bin)) +
        geom_bar() +
        theme(legend.position = "none")

#png("Binary_Variables_3.png")
plot_grid(a,b,c,d + rremove("x.text"),
          ncol = 2, nrow = 2)
#dev.off()
rm(a,b,c,d)

a = combined_data %>%
        ggplot(aes(ps_calc_16_bin, fill = ps_calc_16_bin)) +
        geom_bar() +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_calc_17_bin, fill = ps_calc_17_bin)) +
        geom_bar() +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_calc_18_bin, fill = ps_calc_18_bin)) +
        geom_bar() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_calc_19_bin, fill = ps_calc_19_bin)) +
        geom_bar() +
        theme(legend.position = "none")

e = combined_data %>%
        ggplot(aes(ps_calc_20_bin, fill = ps_calc_20_bin)) +
        geom_bar() +
        theme(legend.position = "none")

#png("Binary_Variables_4.png")
plot_grid(a,b,c,d,e + rremove("x.text"),
          ncol = 2, nrow = 3)
#dev.off()
rm(a,b,c,d,e)

a = combined_data %>%
        ggplot(aes(ps_ind_02_cat, fill = ps_ind_02_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_ind_04_cat, fill = ps_ind_04_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_ind_05_cat, fill = ps_ind_05_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_car_01_cat, fill = ps_car_01_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

#png("Category_Variables_1.png")
plot_grid(a,b,c,d + rremove("x.text"),
          ncol = 2, nrow = 2)
#dev.off()
rm(a,b,c,d)

a = combined_data %>%
        ggplot(aes(ps_car_02_cat, fill = ps_car_02_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_car_03_cat, fill = ps_car_03_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_car_04_cat, fill = ps_car_04_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_car_05_cat, fill = ps_car_05_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

#png("Category_Variables_2.png")
plot_grid(a,b,c,d + rremove("x.text"),
          ncol = 2, nrow = 2)
#dev.off()
rm(a,b,c,d)

a = combined_data %>%
        ggplot(aes(ps_car_06_cat, fill = ps_car_06_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_car_07_cat, fill = ps_car_07_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_car_08_cat, fill = ps_car_08_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_car_09_cat, fill = ps_car_09_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

e = combined_data %>%
        ggplot(aes(ps_car_10_cat, fill = ps_car_10_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

f = combined_data %>%
        ggplot(aes(ps_car_11_cat, fill = ps_car_11_cat)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

#png("Category_Variables_3.png")
plot_grid(a,b,c,d,e,f + rremove("x.text"),
          ncol = 2, nrow = 3)
#dev.off()
rm(a,b,c,d,e,f)

a = combined_data %>%
        mutate(ps_ind_01 = as.factor(ps_ind_01)) %>%
        ggplot(aes(ps_ind_01, fill = ps_ind_01)) +
        geom_bar() +
        theme(legend.position = "none")

b = combined_data %>%
        mutate(ps_ind_03 = as.factor(ps_ind_03)) %>%
        ggplot(aes(ps_ind_03, fill = ps_ind_03)) +
        geom_bar() +
        theme(legend.position = "none")

c = combined_data %>%
        mutate(ps_ind_14 = as.factor(ps_ind_14)) %>%
        ggplot(aes(ps_ind_14, fill = ps_ind_14)) +
        geom_bar() +
        scale_y_log10() +
        theme(legend.position = "none")

d = combined_data %>%
        mutate(ps_ind_15 = as.factor(ps_ind_15)) %>%
        ggplot(aes(ps_ind_15, fill = ps_ind_15)) +
        geom_bar() +
        theme(legend.position = "none")

e = combined_data %>%
        mutate(ps_car_11 = as.factor(ps_car_11)) %>%
        ggplot(aes(ps_car_11, fill = ps_car_11)) +
        geom_bar() +
        theme(legend.position = "none")

f = combined_data %>%
        mutate(ps_calc_04 = as.factor(ps_calc_04)) %>%
        ggplot(aes(ps_calc_04, fill = ps_calc_04)) +
        geom_bar() +
        theme(legend.position = "none")

#png("Integer_Variables_1.png")
plot_grid(a,b,c,d,e,f + rremove("x.text"),
          ncol = 2, nrow = 3)
#dev.off()
rm(a,b,c,d,e,f)

a = combined_data %>%
        mutate(ps_calc_05 = as.factor(ps_calc_05)) %>%
        ggplot(aes(ps_calc_05, fill = ps_calc_05)) +
        geom_bar() +
        theme(legend.position = "none")

b = combined_data %>%
        mutate(ps_calc_06 = as.factor(ps_calc_06)) %>%
        ggplot(aes(ps_calc_06, fill = ps_calc_06)) +
        geom_bar() +
        theme(legend.position = "none")

c = combined_data %>%
        mutate(ps_calc_07 = as.factor(ps_calc_07)) %>%
        ggplot(aes(ps_calc_07, fill = ps_calc_07)) +
        geom_bar() +
        theme(legend.position = "none")

d = combined_data %>%
        mutate(ps_calc_08 = as.factor(ps_calc_08)) %>%
        ggplot(aes(ps_calc_08, fill = ps_calc_08)) +
        geom_bar() +
        theme(legend.position = "none")

e = combined_data %>%
        mutate(ps_calc_09 = as.factor(ps_calc_09)) %>%
        ggplot(aes(ps_calc_09, fill = ps_calc_09)) +
        geom_bar() +
        theme(legend.position = "none")

f = combined_data %>%
        ggplot(aes(ps_calc_10, fill = ps_calc_10)) +
        geom_histogram(binwidth = 1) +
        theme(legend.position = "none")

#png("Integer_Variables_2.png")
plot_grid(a,b,c,d,e,f + rremove("x.text"),
          ncol = 2, nrow = 3)
#dev.off()
rm(a,b,c,d,e,f)

a = combined_data %>%
        ggplot(aes(ps_calc_11, fill = ps_calc_11)) +
        geom_histogram(binwidth = 1) +
        theme(legend.position = "none")

b = combined_data %>%
        mutate(ps_calc_12 = as.factor(ps_calc_12)) %>%
        ggplot(aes(ps_calc_12, fill = ps_calc_12)) +
        geom_bar() +
        theme(legend.position = "none")

c = combined_data %>%
        mutate(ps_calc_13 = as.factor(ps_calc_13)) %>%
        ggplot(aes(ps_calc_13, fill = ps_calc_13)) +
        geom_bar() +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_calc_14, fill = ps_calc_14)) +
        geom_histogram(binwidth = 1) +
        theme(legend.position = "none")

#png("Integer_Variables_3.png")
plot_grid(a,b,c,d + rremove("x.text"),
          ncol = 2, nrow = 2)
#dev.off()
rm(a,b,c,d)

a = combined_data %>%
        ggplot(aes(ps_reg_01, fill = ps_reg_01)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_reg_02, fill = ps_reg_02)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_reg_03, fill = ps_reg_03)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_calc_01, fill = ps_calc_01)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

e = combined_data %>%
        ggplot(aes(ps_calc_02, fill = ps_calc_02)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

f = combined_data %>%
        ggplot(aes(ps_calc_03, fill = ps_calc_03)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

#png("Numeric_Variables_1.png")
plot_grid(a,b,c,d,e,f + rremove("x.text"),
          ncol = 2, nrow = 3)
#dev.off()
rm(a,b,c,d,e,f)

a = combined_data %>%
        ggplot(aes(ps_car_12, fill = ps_car_12)) +
        geom_histogram( binwidth = 0.05) +
        theme(legend.position = "none")

b = combined_data %>%
        ggplot(aes(ps_car_13, fill = ps_car_13)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

c = combined_data %>%
        ggplot(aes(ps_car_14, fill = ps_car_14)) +
        geom_histogram(binwidth = 0.01) +
        theme(legend.position = "none")

d = combined_data %>%
        ggplot(aes(ps_car_15, fill = ps_car_15)) +
        geom_histogram(binwidth = 0.1) +
        theme(legend.position = "none")

#png("Numeric_Variables_2.png")
plot_grid(a,b,c,d + rremove("x.text"),
          ncol = 2, nrow = 2)
#dev.off()
rm(a,b,c,d)

#png("Target_Variable.png")
train %>%
        ggplot(aes(target, fill = target)) +
        geom_bar() + ggtitle("Target Variable") +
        theme(legend.position = "none")
#dev.off()

missing_values = as.data.frame(colSums(is.na(combined_data)))

missing_values[ "var" ] = rownames(missing_values)
missing_values =  melt( missing_values, id.vars="var", value.name="no_of_NA")
missing_values = missing_values[missing_values$no_of_NA >0,]


#png("Columns_with_missing_values.png")

ggplot(missing_values, aes( x = var, y = no_of_NA) ) + 
        geom_bar( position = "identity", stat = "identity" ) + 
        geom_text(aes(label=no_of_NA), vjust=0) + ggtitle("Columns with missing values") +
        theme(axis.text.x = element_text(angle = 90, hjust = 1))
#dev.off()








