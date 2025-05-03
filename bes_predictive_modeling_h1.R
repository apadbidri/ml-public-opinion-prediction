# ------ Data Cleaning 
library(haven)
library(caret)
library(dplyr)

data <- read_dta("2019_data.dta")

# View data
summary(data) # or str(data)
sum(is.na(data))

# Percent missing per column
missing_percent <- colSums(is.na(data)) / nrow(data) * 100
sort(missing_percent, decreasing = TRUE)

# Remove columns with more than 80% missingness
threshold <- 0.60
data_cleaned <- data[, colMeans(is.na(data)) <= threshold]

# Calculate percent of missingness per row
row_missingness <- rowMeans(is.na(data_cleaned)) * 100
summary(row_missingness)

# Filter rows where missingness is 50% or less
data_cleaned <- data_cleaned[row_missingness <= 50, ]

#Convert coded unanswered i.e. -1 or -2 to NA
data_cleaned[data_cleaned < 0] <- NA

sum(is.na(data_cleaned))

#View column of interest
summary(data_cleaned$h01)

# Remove rows where h01 is missing
data_cleaned <- data_cleaned[!is.na(data_cleaned$h01), ]

# Convert char columsn to factors
# Convert character columns to factors
char_cols <- names(data_cleaned)[sapply(data_cleaned, is.character)]
data_cleaned[char_cols] <- lapply(data_cleaned[char_cols], as.factor)

# Remove predictors with zero variance (optional but helps modeling)
nzv <- nearZeroVar(data_cleaned, saveMetrics = TRUE)
data_cleaned <- data_cleaned[, !nzv$zeroVar]

# Tried to apply KNN but seems like cannot find enough neighbours 

# So use median for continuous and mode for categorical
# Function to calculate the mode (most frequent value) of a vector
get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Separate columns by type
numeric_cols <- sapply(data_cleaned, is.numeric)
factor_cols <- sapply(data_cleaned, is.factor)

# For numeric columns, replace NAs with the median
data_cleaned[numeric_cols] <- lapply(data_cleaned[numeric_cols], function(x) {
  if (any(is.na(x))) {
    x[is.na(x)] <- median(x, na.rm = TRUE)
  }
  return(x)
})

# For factor columns, replace NAs with the mode
data_cleaned[factor_cols] <- lapply(data_cleaned[factor_cols], function(x) {
  if (any(is.na(x))) {
    mode_value <- get_mode(x[!is.na(x)])
    x[is.na(x)] <- mode_value
  }
  return(x)
})

# Check if there are any remaining missing values
sum(is.na(data_cleaned))

missing_cells <- is.na(data_cleaned)

# View the rows and columns where the missing values are
which(missing_cells, arr.ind = TRUE)

# Found that it's just interview date so removing that variable
data_cleaned <- data_cleaned %>%
  select(-Interview_Date)

# Check if there are any remaining missing values
sum(is.na(data_cleaned))

# -------- Model building
# too many categories in the predictors for random forest 

# Load necessary libraries
library(rpart)
library(rpart.plot)
library(caret)

# 1. Split the data into 80% training and 20% test set
set.seed(123)  # For reproducibility
train_index <- createDataPartition(data_cleaned$h01_class, p = 0.8, list = FALSE)
train_set <- data_cleaned[train_index, ]
test_set <- data_cleaned[-train_index, ]

# 3. Train a decision tree model
tree_model <- rpart(h01_class ~ ., data = train_set, method = "class")

# 4. Make predictions on the test set
test_predictions <- predict(tree_model, test_set, type = "class")

# 5. Evaluate model performance
confusion_matrix <- table(test_set$h01_class, test_predictions)
print("Confusion Matrix:")
print(confusion_matrix)

# Accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(paste("Accuracy: ", round(accuracy, 4)))

# F1 score (for binary classification)
precision <- confusion_matrix[2, 2] / (confusion_matrix[2, 2] + confusion_matrix[1, 2])
recall <- confusion_matrix[2, 2] / (confusion_matrix[2, 2] + confusion_matrix[2, 1])
f1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("F1 Score: ", round(f1_score, 4)))




library(caret)
library(rpart)
library(rpart.plot)

# Step 1: Bin the continuous h01 into categories
# (e.g., Low: 1–3, Medium: 4–7, High: 8–10)
data_cleaned$h01_class <- cut(data_cleaned$h01,
                              breaks = c(-Inf, 3, 7, Inf),
                              labels = c("Low", "Medium", "High"),
                              right = TRUE)

# Step 2: Train-test split
set.seed(123)
train_index <- createDataPartition(data_cleaned$h01_class, p = 0.8, list = FALSE)
train_set <- data_cleaned[train_index, ]
test_set <- data_cleaned[-train_index, ]

# Step 3: Randomly select 20 predictors
set.seed(123)
predictors <- setdiff(names(train_set), c("h01", "h01_class"))  # exclude target
sampled_cols <- sample(predictors, 20)

# Step 4: Subset to sampled predictors + target
small_train <- train_set[, c("h01_class", sampled_cols)]
small_test <- test_set[, c("h01_class", sampled_cols)]

# Step 5: Train decision tree
tree_model <- rpart(h01_class ~ ., data = small_train, method = "class")

# Optional: Visualize
rpart.plot(tree_model)

# Step 6: Predict
test_predictions <- predict(tree_model, small_test, type = "class")

# Step 7: Evaluate
confusionMatrix(test_predictions, small_test$h01_class)
