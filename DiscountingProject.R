# Trevor Volpe

# Read in results from a randomized survey that captures whether an individual
# would try a restaurant, when they previously would not have, given a randomly 
# generated discount amount. Preprocess the data and run queries. Create 
# a logistic regression, KNN, naive bayes, decision tree, and neural network 
# to determine which model will be the best at predicting the outcome.
# Preparing packages and CSVs ---------------------------------------------
# Install required packages
# install.packages("tidyverse")
# install.packages("olsrr")
# install.packages("dummy")
# install.packages("corrplot")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("neuralnet")
# install.packages("e1071")

# Load appropriate packages
library(tidyverse)
library(olsrr)
library(dummy)
library(corrplot)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(e1071)
library(class)

# set working directory
setwd('C:\\Users\\volpe\\Documents\\mis543')

# Read CSV file into a tibble and define column types
# l for logical
# n for numeric
# i for integers
# c for characters
# f for factors
# D for dates
# T for datetimes
restTrial <- read_csv(file = "munch_try_with_discount.csv",
                      col_types = "lniifflililinii",
                      col_names = TRUE)

# Reading the binned smoothed CSV file into an object called restTrialBinned
# using the read_csv() 
# function.
restTrialBinned <- read_csv(file = "munch_try_with_discount_binned.csv",
                            col_types = "lniifflililinii",
                            col_names = TRUE)

# Data Preprocessing ------------------------------------------------------
# Display summary of restTrial and restrial1
summary(restTrial)
summary(restTrialBinned)

# Drop NA Age 
restTrial <- restTrial %>%
  drop_na()
restTrialBinned <-restTrialBinned %>%
  drop_na()

# Create a function called DisplayAllHistograms that take in a tibble parameter
# that will display a histogram for all numeric features in the tibble
displayAllHistograms <- function(tibbleDataSet) {
  tibbleDataSet %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = aes(x = value, fill = key),
                              color = "black") +
    facet_wrap(~ key, scales = "free") +
    theme_minimal()
}

# Call the displayAllHistgoram() functions using our restTrial tibble
displayAllHistograms(restTrial)

# normalize any non-normal features (age, income)
# drop income values of 0 because can't take log of 0
restTrial <- restTrial %>%
  filter(Income != 0)

restTrial <- restTrial %>% 
  mutate(LogIncome = log(Income))

restTrial <- restTrial %>% 
  mutate(LogAge = log(Age))

# Prep scaled tibble
restTrialScaled <- restTrial

# Scale and/or smooth data if required by the model
restTrialScaled <- restTrialScaled %>% 
  mutate(IncomeScaled = (Income - min(Income))/
           (max(Income)-min(Income))) %>%
  mutate(AgeScaled = (Age - min(Age))/
           (max(Age)-min(Age))) %>%
  mutate(MinStarsScaled = (MinStars - min(MinStars))/
           (max(MinStars)-min(MinStars))) %>%
  mutate(MaxDistScaled = (MaxDist - min(MaxDist))/
           (max(MaxDist)-min(MaxDist))) %>%
  mutate(MaxPriceScaled = (MaxPrice - min(MaxPrice))/
           (max(MaxPrice)-min(MaxPrice))) %>%
  mutate(RestDistanceScaled = (RestDistance - min(RestDistance))/
           (max(RestDistance)-min(RestDistance))) %>%
  mutate(RestRatingScaled = (RestRating - min(RestRating))/
           (max(RestRating)-min(RestRating))) %>%
  mutate(RestPriceScaled = (RestPrice - min(RestPrice))/
           (max(RestPrice)-min(RestPrice))) %>%
  mutate(RestDiscountScaled = (RestDiscount - min(RestDiscount))/
           (max(RestDiscount)-min(RestDiscount)))

# Drop non-scaled data from scaled tibble
restTrialScaled <- restTrialScaled %>%
  select(-Income, -MinStars, -MaxDist, -MaxPrice, -RestDistance,
         -RestRating, -RestPrice, -RestDiscount, -LogIncome, -LogAge, -Age)

# Drop non-normalized data after normalization has been done 
restTrial <- restTrial %>%
  select(-Age, -Income)

# Re-check distributions to see if normalization worked
displayAllHistograms(restTrial)

# Convert restTrial into a tibble containing only categorical features
restTrialCat <- restTrial %>% 
  select(TakeReferral, Frequency)

# Convert restTrialCat into a df containing only categorical features
restTrialCatDf <- data.frame(restTrialCat)

# Dummy code the position feature and turn the dataframe back into a tibble
restTrialCatDummy <- as_tibble(dummy(restTrialCatDf, int = TRUE))

# Rename dummy coded features
names(restTrialCatDummy) <- c("TakeRefSomeLike",
                              "TakeRefExtLike",
                              "TakeRefNeither",
                              "TakeRefExtUnlike",
                              "TakeRefSomeUnlike",
                              "FreqMonthly",
                              "FreqEveryFewMonth",
                              "FreqWeekly",
                              "FreqEveryCoupleWeek",
                              "FreqDaily",
                              "FreqNever")

# Combine restTrialCatDummy with other features in restTrial
# excluding TakeReferral, Frequency
restTrialDummied <- cbind(restTrialCatDummy, restTrial) %>%
  select(-TakeReferral,
         -Frequency)

# Remove referential variables for each categorical variable
restTrialDummied <- restTrialDummied %>% 
  select(-TakeRefNeither, -FreqNever)

# Combined dummy coded categorical variables with scaled dataset
restTrialScaled <- cbind(restTrialCatDummy, restTrialScaled)

# Drop original categorical variables from scaled dataset
restTrialScaled <- restTrialScaled %>%
  select(-TakeReferral,
         -Frequency)

# Data collection already controlled for outliars, randomly generated numbers,
# and income is capped at $500,000 (i.e. if someone made over $500,000 they
# only put $500,000)
# Interesting query 1: Difference in average discount between those who 
# try and those who do not try with a discount provided
print(restTrial %>%
        group_by(TryWith) %>%
        summarize(MeanDiscount = mean(RestDiscount)) %>%
        arrange(desc(MeanDiscount)),
      n = Inf)

# Interesting query 2: Average minimum stars required to try a restaurant based
# on dining frequency.
print(restTrial %>%
        group_by(Frequency) %>%
        summarize(MeanMinStars = mean(MinStars)) %>%
        arrange(desc(MeanMinStars)),
      n = Inf)

# Interesting query 3: Average binned maximum price willing to pay by binned
# age groups. 
print(restTrialBinned %>%
        group_by(Age) %>%
        summarize(MeanMaxPrice = mean(MaxPrice)) %>%
        arrange(desc(MeanMaxPrice)),
      n = Inf)

# Correlation plot of restTrial variables
restTrialNumeric <- restTrial %>% 
  keep(is.numeric)

corrplot(cor(restTrialNumeric),
         method = "number",
         type = "lower")

# Set randomization seed to a specified value for recreation purposes
set.seed(203)

# Randomly split dummied data into 75% training and 25% testing sets
sampleSetDummied <- sample(nrow(restTrialDummied),
                    round(nrow(restTrialDummied) * 0.75),
                    replace = FALSE)

# Randomly split binned data into 75% training and 25% testing sets
sampleSetBinned <- sample(nrow(restTrialBinned),
                          round(nrow(restTrialBinned) * 0.75),
                          replace = FALSE)

# Randomly split scaled data into 75% training and 25% testing sets
sampleSetScaled <- sample(nrow(restTrialScaled),
                         round(nrow(restTrialScaled) * 0.75),
                         replace = FALSE)

# Put the records from the 75% sample into restTrialDummiedTraining and
# restTrialBinnedTraining
restTrialDummiedTraining <- restTrialDummied[sampleSetDummied, ]
restTrialBinnedTraining <- restTrialBinned[sampleSetBinned, ]
restTrialScaledTraining <- restTrialScaled[sampleSetScaled, ]

# Put the records from the 25% sample into restTrialDummiedTesting and
# restTrialBinnedTesting
restTrialDummiedTesting <- restTrialDummied[-sampleSetDummied, ]
restTrialBinnedTesting <- restTrialBinned[ -sampleSetBinned, ]
restTrialScaledTesting <- restTrialScaled[-sampleSetScaled, ]

# Check for class imbalance of TryWith
print(summary(restTrialDummiedTraining$TryWith))

# Store magnitude of class imbalance of TryWith
classImbalanceMagnitude <- 202 / 66

# Logistic Regression -----------------------------------------------------
# Create logistic regression which uses TryWith as dependent variable
restTrialModelLogistic <- glm(data = restTrialDummiedTraining,
                              family = binomial,
                              formula = TryWith ~ .)

# Display logistic regression model
print(summary(restTrialModelLogistic))

# Predict outcome of the test set using the restTrialModel
RestTrialPredictionLogistic <- predict(restTrialModelLogistic,
                                       restTrialDummiedTesting)

# Set threshold for binary outcome at above 0.5 for 1, anything equal to or
# below 0.5 will be set to 0
RestTrialPredictionLogistic <-
  ifelse(RestTrialPredictionLogistic > 0.5, 1, 0)

# Generate a confusion matrix of predictions
restTrialConfusionMatrixLogistic <- table(restTrialDummiedTesting$TryWith,
                                          RestTrialPredictionLogistic)

# Display confusion matrix of predictions
print(restTrialConfusionMatrixLogistic)

# Calculate the false positive rate of predictions
restTrialConfusionMatrixLogistic[1, 2] /
  (restTrialConfusionMatrixLogistic[1, 2] +
     restTrialConfusionMatrixLogistic[1, 1])

# Calculate the false negative rate of predictions
restTrialConfusionMatrixLogistic[2, 1] /
  (restTrialConfusionMatrixLogistic[2, 1] +
     restTrialConfusionMatrixLogistic[2, 2])

# Calculate the prediction accuracy by dividing the number of true positives
# and true negatives by the total amount of predictions in the testing dataset
sum(diag(restTrialConfusionMatrixLogistic)) / nrow(restTrialDummiedTesting)

# Create linear regression to test multicollinearity
restTrialModelLinear <- lm(data = restTrialDummiedTraining,
                           formula = TryWith ~ .)

# Test for multicollinearity among explanatory variables
ols_vif_tol(restTrialModelLinear)

# KNN ---------------------------------------------------------------------
# Split the tibble into two, one containing only the label, and the other with         
# the rest of the variables
restTrialLabels <- restTrial %>% select(TryWith)
restTrial <- restTrial %>% select(-TryWith)
print(restTrial)
print(restTrialLabels)

# Set the seed as 203, and split the data into training and testing datasets
set.seed(203)
sampleSet <- sample(nrow(restTrial),
                    round(nrow(restTrial)*0.75),
                    replace = FALSE)

# Put records from the 75% sample into restTrialTraining tibble
restTrialTraining <- restTrial[sampleSet, ]
restTrialTrainingLabels <- restTrialLabels[sampleSet, ]

# Put records from the 25% sample into restTrialTesting tibble
restTrialTesting <- restTrial[-sampleSet, ]
restTrialTestingLabels <- restTrialLabels[-sampleSet, ]

# Converting the logical columns to numeric values and dropping them.
restTrialTraining <- restTrialTraining %>% 
  mutate(GenderNumeric = as.numeric(restTrialTraining$Gender)) %>%
  mutate(StudentNumeric = as.numeric(restTrialTraining$Student)) %>%
  mutate(ChildrenNumeric = as.numeric(restTrialTraining$Children))
restTrialTraining <- restTrialTraining %>% 
  select(-Gender, -Student, -Children)

# Converting the logical columns to numeric values.
restTrialTesting <- restTrialTesting %>% 
  mutate(GenderNumeric = as.numeric(restTrialTesting$Gender)) %>%
  mutate(StudentNumeric = as.numeric(restTrialTesting$Student)) %>%
  mutate(ChildrenNumeric = as.numeric(restTrialTesting$Children))
restTrialTesting <- restTrialTesting %>% 
  select(-Gender, -Student, -Children)

# Create a matrix of k-values with their predictive accuracy
kValueMatrix <- matrix(data = NA,
                       nrow = 0,
                       ncol =2)

# Assign column names to the matrix
colnames(kValueMatrix) <- c("k value","Predictive Accuracy")

# Loop through with different values of k to determine the best fitting model
# using odd numbers from 1 to the number of observations in the training data
# set
for(kValue in 1:nrow(restTrialTraining)) {
  if (kValue %% 2 != 0) {
    restTrialPrediction <- knn(train = restTrialTraining,
                               test = restTrialTesting,
                               cl = restTrialTrainingLabels$TryWith,
                               k = kValue)
    
    restTrialConfusionMatrix <- table(restTrialTestingLabels$TryWith,
                                      restTrialPrediction)
    
    predictiveAccuracy <- sum(diag(restTrialConfusionMatrix))/
      nrow(restTrialTesting)
    
    kValueMatrix <- rbind(kValueMatrix, c(kValue, predictiveAccuracy))
  }
}

# Display the kValueMatrix in the console
print(kValueMatrix)

# Regenerate the k-Nearest Neighbors Model with optimal k-value
restTrialPrediction <- knn(train = restTrialTraining,
                           test = restTrialTesting,
                           cl = restTrialTrainingLabels$TryWith,
                           k = 9)

# Display the results of the prediction from the testing dataset on the console 
print(restTrialPrediction)

# Display the summary of the predictions from the testing dataset
print(summary(restTrialPrediction))

# Evaluate the model by forming a confusion matrix
restTrialConfusionMatrix <- table(restTrialTestingLabels$TryWith,
                                  restTrialPrediction)

# Display the confusion matrix
print(restTrialConfusionMatrix)

# Calculate the model predictive accuracy and store it into a variable called 
# predictiveAccuracy
predictiveAccuracy <- sum(diag(restTrialConfusionMatrix))/
  nrow(restTrialTestingLabels)
print(predictiveAccuracy)

# Calculate the false positive rate of predictions
restTrialConfusionMatrix[1, 2] /
  (restTrialConfusionMatrix[1, 2] +
     restTrialConfusionMatrix[1, 1])

# Calculate the false negative rate of predictions
restTrialConfusionMatrix[2, 1] /
  (restTrialConfusionMatrix[2, 1] +
     restTrialConfusionMatrix[2, 2])

# Naive Bayes -------------------------------------------------------------
# Generate naive bayes model
restTrialBinnedModel <- naiveBayes(formula = TryWith ~ .,
                                   data = restTrialBinnedTraining,
                                   laplace = 1)

# Build probabilities for records in the testing dataset and store them
restTrialBinnedProbability <- predict(restTrialBinnedModel,
                                      restTrialBinnedTraining,
                                      type = "raw")

# Print restTrialProbability
print(restTrialBinnedProbability)

# Predict classes for values
restTrialBinnedPrediction <- predict(restTrialBinnedModel,
                                     restTrialBinnedTesting,
                                     type = "class")

# display restTrialBinned Prediction
print(restTrialBinnedPrediction)

# Evaluate the Naive Bayes model with confusion matrix
restTrialBinnedConfusion <- table(restTrialBinnedTesting$TryWith,
                                  restTrialBinnedPrediction)

# Display confusion matrix
print(restTrialBinnedConfusion)

# calculate Naive Bayes model's predictive accuracy
restTrialBinnedPredictiveAccuracy <- sum(diag(restTrialBinnedConfusion))/
  nrow(restTrialBinnedTesting)

# Display predictive accuracy
print(restTrialBinnedPredictiveAccuracy)

# Calculate the false positive rate of predictions
restTrialBinnedConfusion[1, 2] /
  (restTrialBinnedConfusion[1, 2] +
     restTrialBinnedConfusion[1, 1])

# Calculate the false negative rate of predictions
restTrialBinnedConfusion[2, 1] /
  (restTrialBinnedConfusion[2, 1] +
     restTrialBinnedConfusion[2, 2])

# Decision Tree -----------------------------------------------------------
# Combine explanatory variables with labels for training and testing sets
restTrialCombinedTraining <- cbind(restTrialTraining, restTrialTrainingLabels)
restTrialCombinedTesting <- cbind(restTrialTesting, restTrialTestingLabels)

# Generate the decision tree model to predict restTrial based on the other 
# variables in the dataset. Use 0.01 as the complexity parameter.
restTrialDecisionTreeModel0 <- rpart(formula = TryWith ~.,
                                     method = "class",
                                     cp = 0.01,
                                     data = restTrialCombinedTraining)

restTrialDecisionTreeModel1 <- rpart(formula = TryWith ~.,
                                     method = "class",
                                     cp = 0.1,
                                     data = restTrialCombinedTraining)
restTrialDecisionTreeModel2 <- rpart(formula = TryWith ~.,
                                     method = "class",
                                     cp = 0.007,
                                     data = restTrialCombinedTraining)

# Display the decision tree visualization in R
rpart.plot(restTrialDecisionTreeModel0)
rpart.plot(restTrialDecisionTreeModel1)
rpart.plot(restTrialDecisionTreeModel2)

# Predicting the classes for each record in the testing dataset and storing them  
# in restTrialPrediction
restTrialPredictionDT0 <- predict(restTrialDecisionTreeModel0,
                                  restTrialCombinedTesting,
                                  type = "class")
restTrialPredictionDT1 <- predict(restTrialDecisionTreeModel1,
                                  restTrialCombinedTesting,
                                  type = "class")
restTrialPredictionDT2 <- predict(restTrialDecisionTreeModel2,
                                  restTrialCombinedTesting,
                                  type = "class")

# Displaying the restTrialPrediction on the console
print(restTrialPredictionDT0)
print(restTrialPredictionDT1)
print(restTrialPredictionDT2)

# Evaluating the model by forming a confusion matrix
restTrialConfusionMatrixDT0 <- table(restTrialCombinedTesting$TryWith,
                                     restTrialPredictionDT0)
restTrialConfusionMatrixDT1 <- table(restTrialCombinedTesting$TryWith,
                                     restTrialPredictionDT1)
restTrialConfusionMatrixDT2 <- table(restTrialCombinedTesting$TryWith,
                                     restTrialPredictionDT2)

# Displaying the confusion matrix on the console
print(restTrialConfusionMatrixDT0)
print(restTrialConfusionMatrixDT1)
print(restTrialConfusionMatrixDT2)

# Calculating the model predictive accuracy and storing it into a variable 
# called predictiveAccuracy
predictiveAccuracyDT0 <- sum(diag(restTrialConfusionMatrixDT0)) /
  nrow(restTrialCombinedTesting)
predictiveAccuracyDT1 <- sum(diag(restTrialConfusionMatrixDT1)) /
  nrow(restTrialCombinedTesting)
predictiveAccuracyDT2 <- sum(diag(restTrialConfusionMatrixDT2)) /
  nrow(restTrialCombinedTesting)

# Display the predictive accuracy on the console
print(predictiveAccuracyDT0)
print(predictiveAccuracyDT1)
print(predictiveAccuracyDT2)

# Calculate the false positive and negative rate of predictions DT0
restTrialConfusionMatrixDT0[1, 2] /
  (restTrialConfusionMatrixDT0[1, 2] +
     restTrialConfusionMatrixDT0[1, 1])

restTrialConfusionMatrixDT0[2, 1] /
  (restTrialConfusionMatrixDT0[2, 1] +
     restTrialConfusionMatrixDT0[2, 2])

# Calculate the false positive and negative rate of predictions DT1
restTrialConfusionMatrixDT1[1, 2] /
  (restTrialConfusionMatrixDT1[1, 2] +
     restTrialConfusionMatrixDT1[1, 1])

restTrialConfusionMatrixDT1[2, 1] /
  (restTrialConfusionMatrixDT1[2, 1] +
     restTrialConfusionMatrixDT1[2, 2])

# Calculate the false positive and negative rate of predictions DT2
restTrialConfusionMatrixDT2[1, 2] /
  (restTrialConfusionMatrixDT2[1, 2] +
     restTrialConfusionMatrixDT2[1, 1])

restTrialConfusionMatrixDT2[2, 1] /
  (restTrialConfusionMatrixDT2[2, 1] +
     restTrialConfusionMatrixDT2[2, 2])

# Neural Network ----------------------------------------------------------
# Generate the neural network model to predict TryWithout using IncomeScaled, 
# MinStarsScaled, MaxDistScaled, MaxPriceScaled, RestDistanceScaled, 
# RestRatingScaled and RestPriceScaled.. Use 3 hidden layers. Use "logistic" as
# the smoothing method and set linear.output to FALSE.
restTrialNeuralNet <- neuralnet(
  formula = TryWith ~ .,
  data = restTrialScaledTraining,
  hidden = 3,
  act.fct = "logistic",
  linear.output = FALSE
)

# Display the neural network numeric results
print(restTrialNeuralNet$result.matrix)

# Visualize the neural network
plot(restTrialNeuralNet)

# Use restTrialNeuralNet to generate probabilities on the 
# restTrialScaledTesting data set and store this in 
# restTrialNeuralNetProbability
restTrialNeuralNetProbability <- compute(restTrialNeuralNet,
                                         restTrialScaledTesting)

# Displaying the probabilities from the testing dataset on the console
print(restTrialNeuralNetProbability)

# Converting probability predictions into 0/1 predictions and store this into 
# restTrial1Prediction
restTrialNeuralNetPrediction <-
  ifelse(restTrialNeuralNetProbability$net.result > 0.5, 1, 0)

# Display the 0/1 predictions on the console
print(restTrialNeuralNetPrediction)

# Evaluating the model by forming a confusion matrix
restTrialNeuralNetConfusionMatrix <- table(
  restTrialScaledTesting$TryWith,
  restTrialNeuralNetPrediction)

# Displaying the confusion matrix on the console
print(restTrialNeuralNetConfusionMatrix)

# Calculating the model predictive accuracy
predictiveAccuracyNeuralNet <- sum(diag(restTrialNeuralNetConfusionMatrix)) /
  nrow(restTrialScaledTesting)

# Displaying the predictive accuracy on the console
print(predictiveAccuracyNeuralNet)

# Calculate the false positive rate of predictions
restTrialNeuralNetConfusionMatrix[1, 2] /
  (restTrialNeuralNetConfusionMatrix[1, 2] +
     restTrialNeuralNetConfusionMatrix[1, 1])

# Calculate the false negative rate of predictions
restTrialNeuralNetConfusionMatrix[2, 1] /
  (restTrialNeuralNetConfusionMatrix[2, 1] +
     restTrialNeuralNetConfusionMatrix[2, 2])