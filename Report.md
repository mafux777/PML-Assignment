---
title       : Assignment "Practical Machine Learning"
subtitle    : Analysing Sensor Data
author      : Matthias Funke
---


```
## Loading required package: data.table
## Loading required package: caret
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: doParallel
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```
## Synopsis
The goal of this assignment is to demonstrate how machine learning, in this case a random forest algorithm, can be used to predict whether a particular exercise was done "correctly". The way that the experiment was set up is that the correct way was called "Classe A" and there were 4 ways to do the job incorrectly, "Classe B..E". We trained the RF model with 75% of the data and checked the out-of-sample error by letting the model predict on the remaining 25% test case to obtain an accuracy of 98%. This was done purely on data from accelerometers, gyros, and magnetos. Data for time, window, user, sequence number, were all suppressed. 
Apparently other students were able to obtain higher accuracy by including user name as a predictor, but I chose not to use it to keep the model more "honest".

## Procedure
Read data from CSV file, count NA values and in general explore the data. Eliminate columns with NA data and show how many predictors are left, using nearZeroVar utility function to show variability. Note how all the remaining sensor data predictors have enough variability, therefore the output is informational only. 

```r
setwd("/Users/mfunke/Downloads")
data=read.csv("pml-training.csv", dec='.',na.strings=c("#DIV/0!", "NA"), stringsAsFactors=F)
data=data.table(data)
# identify columns without NA values --- sumna is a utility function defined as sum(is.na(x))
good_vars=data[,lapply(data, sumna)==0]
# select only the columns where we have data
data2=data[, good_vars, with=F]
```

```r
data2[, classe:=as.factor(classe)] # crucial otherwise the model crashes
```

```r
## Pre-process: create training partitions and test partition
inTrain = as.vector(createDataPartition(y=data2$classe, p=.75, list=F))
training=data2[inTrain]
testing=data2[-inTrain]

nearZeroVar(training, saveMetrics=T)
```

```
##                      freqRatio percentUnique zeroVar   nzv
## X                     1.000000  100.00000000   FALSE FALSE
## user_name             1.105913    0.04076641   FALSE FALSE
## raw_timestamp_part_1  1.100000    5.68691398   FALSE FALSE
## raw_timestamp_part_2  1.000000   88.60578883   FALSE FALSE
## cvtd_timestamp        1.027397    0.13588803   FALSE FALSE
## new_window           46.324759    0.01358880   FALSE  TRUE
## num_window            1.100000    5.82959641   FALSE FALSE
## roll_belt             1.078907    7.80676722   FALSE FALSE
## pitch_belt            1.000000   11.69995923   FALSE FALSE
## yaw_belt              1.072193   12.44734339   FALSE FALSE
## total_accel_belt      1.040116    0.19703764   FALSE FALSE
## gyros_belt_x          1.074522    0.87647778   FALSE FALSE
## gyros_belt_y          1.124356    0.45522489   FALSE FALSE
## gyros_belt_z          1.060514    1.11428183   FALSE FALSE
## accel_belt_x          1.041308    1.08030982   FALSE FALSE
## accel_belt_y          1.117698    0.93762740   FALSE FALSE
## accel_belt_z          1.083210    1.96358201   FALSE FALSE
## magnet_belt_x         1.143969    2.09267564   FALSE FALSE
## magnet_belt_y         1.087755    1.96358201   FALSE FALSE
## magnet_belt_z         1.011561    2.96235902   FALSE FALSE
## roll_arm             56.200000   16.61910586   FALSE FALSE
## pitch_arm            90.357143   19.44557684   FALSE FALSE
## yaw_arm              30.107143   18.18861258   FALSE FALSE
## total_accel_arm       1.025564    0.44163609   FALSE FALSE
## gyros_arm_x           1.010610    4.28047289   FALSE FALSE
## gyros_arm_y           1.536585    2.50033972   FALSE FALSE
## gyros_arm_z           1.113065    1.60347873   FALSE FALSE
## accel_arm_x           1.065041    5.23848349   FALSE FALSE
## accel_arm_y           1.138365    3.58064954   FALSE FALSE
## accel_arm_z           1.193548    5.23168909   FALSE FALSE
## magnet_arm_x          1.063492    9.00258187   FALSE FALSE
## magnet_arm_y          1.138462    5.81600761   FALSE FALSE
## magnet_arm_z          1.047059    8.52017937   FALSE FALSE
## roll_dumbbell         1.104167   86.08506591   FALSE FALSE
## pitch_dumbbell        2.264151   83.78176383   FALSE FALSE
## yaw_dumbbell          1.152174   85.49395298   FALSE FALSE
## total_accel_dumbbell  1.099502    0.29215926   FALSE FALSE
## gyros_dumbbell_x      1.037778    1.60347873   FALSE FALSE
## gyros_dumbbell_y      1.262791    1.84128278   FALSE FALSE
## gyros_dumbbell_z      1.082589    1.34529148   FALSE FALSE
## accel_dumbbell_x      1.075000    2.81288219   FALSE FALSE
## accel_dumbbell_y      1.099448    3.12542465   FALSE FALSE
## accel_dumbbell_z      1.202312    2.73814377   FALSE FALSE
## magnet_dumbbell_x     1.071942    7.36513113   FALSE FALSE
## magnet_dumbbell_y     1.143885    5.59179236   FALSE FALSE
## magnet_dumbbell_z     1.007143    4.49789374   FALSE FALSE
## roll_forearm         13.380090   13.36458758   FALSE FALSE
## pitch_forearm        68.744186   18.31091181   FALSE FALSE
## yaw_forearm          15.802139   12.41337138   FALSE FALSE
## total_accel_forearm   1.119741    0.46881370   FALSE FALSE
## gyros_forearm_x       1.030303    1.97717081   FALSE FALSE
## gyros_forearm_y       1.013793    4.91914662   FALSE FALSE
## gyros_forearm_z       1.121127    2.01793722   FALSE FALSE
## accel_forearm_x       1.234375    5.30642750   FALSE FALSE
## accel_forearm_y       1.000000    6.65851338   FALSE FALSE
## accel_forearm_z       1.016129    3.81845359   FALSE FALSE
## magnet_forearm_x      1.033898   10.00815328   FALSE FALSE
## magnet_forearm_y      1.423729   12.47452099   FALSE FALSE
## magnet_forearm_z      1.047619   11.12922951   FALSE FALSE
## classe                1.469452    0.03397201   FALSE FALSE
```
As discussed, we want to use only data from the sensors (and Euler derived pitch, roll), so grep for the appropriate variable names. Then, use
principal component analysis to reduce the number of predictors to relevant set of 23. 

```r
g=grep("(^roll)|(^accel)|(^gyros)|(^magnet)|(^pitch)", names(data2))

p2=preProcess(training[,g,with=F], method="pca")
trainPC=predict(p2, training[,g,with=F])
testPC=predict(p2, testing[,g,with=F])
```
Now, run the model "rf" (random forest) and see how it does on the training set. 

```r
c=trainControl(method="repeatedcv")
modelFit=train(training$classe ~ ., method="rf", data=trainPC, trControl=c)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
modelFit
```

```
## Random Forest 
## 
## 14718 samples
##    20 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold, repeated 1 times) 
## 
## Summary of sample sizes: 13245, 13245, 13248, 13248, 13247, 13245, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9758790  0.9694855  0.005060171  0.006394690
##   11    0.9737053  0.9667369  0.005539763  0.007002266
##   21    0.9671154  0.9584022  0.006266076  0.007915977
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
confusionMatrix(testing$classe, predict(modelFit, testPC))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1379    5    4    4    3
##          B   11  929    9    0    0
##          C    1   10  838    6    0
##          D    2    5   34  760    3
##          E    0    2    6    1  892
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9784          
##                  95% CI : (0.9739, 0.9823)
##     No Information Rate : 0.2841          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9727          
##  Mcnemar's Test P-Value : 9.706e-06       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9899   0.9769   0.9405   0.9857   0.9933
## Specificity            0.9954   0.9949   0.9958   0.9894   0.9978
## Pos Pred Value         0.9885   0.9789   0.9801   0.9453   0.9900
## Neg Pred Value         0.9960   0.9944   0.9869   0.9973   0.9985
## Prevalence             0.2841   0.1939   0.1817   0.1572   0.1831
## Detection Rate         0.2812   0.1894   0.1709   0.1550   0.1819
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9927   0.9859   0.9681   0.9875   0.9955
```
The confusion matrix gives me a good estimate of the out-of-sample error, because it uses data that it was not trained on to predict a set of values with the correct answer known. This is called cross-validation. (The model itself already did cross-validation automatically.)

```r
setwd("/Users/mfunke/Downloads")
realtest=read.csv("pml-testing.csv", dec='.',na.strings=c("#DIV/0!", "NA"), stringsAsFactors=F)
realtest=data.table(realtest)
# select only the columns where we have data
realtest=realtest[, good_vars, with=F]
realPC=predict(p2, realtest[,g,with=F])
answers=predict(modelFit, realPC)
answers
```

```
##  [1] B A C A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
Finally, we have run the model on the test data provided as part of the assignment and submitted. On the first attempt, got 18/20 correct. It would appear that the accuracy in this instance (90%) is a bit lower than what we estimated above, and I don't have an explanation for that right now. We don't know how the final test set was selected. 
