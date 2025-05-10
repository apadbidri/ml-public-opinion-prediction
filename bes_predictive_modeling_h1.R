# == Data Preprocessing ======
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
sum(is.na(data_cleaned$h01))

# Remove rows where h01 is missing
data_cleaned <- data_cleaned[!is.na(data_cleaned$h01), ]
# Convert factor levels to numeric (careful: as.numeric() on a factor gives level index, not the label!)
data_cleaned$h01 <- as.numeric(as.character(data_cleaned$h01))

str(data_cleaned$h01)
summary(data_cleaned$h01)

# Convert char columns to factors
# Convert character columns to factors
char_cols <- names(data_cleaned)[sapply(data_cleaned, is.character)]
data_cleaned[char_cols] <- lapply(data_cleaned[char_cols], as.factor)

# Remove predictors with zero variance
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

cols_to_remove <- c("a01", "k04", "Constit_Code", "Constit_Name", 
                    "LA_UA_Code", "LA_UA_Name", "interviewer", "Interview_Date", "Stratumlabel", "Stratumlabel2")

data_cleaned <- data_cleaned %>%
  select(-all_of(cols_to_remove))


# Check if there are any remaining missing values
sum(is.na(data_cleaned))

# Check the distribution of the target variable (h01)
table(data_cleaned$h01)
# Convert target to factor
data_cleaned$h01 <- as.factor(data_cleaned$h01)

sum(sapply(data_cleaned, is.numeric))








# ====== New Pipeline ====
library(caret)
set.seed(123)

train_index <- createDataPartition(data_cleaned$h01, p = 0.8, list = FALSE)
train_data <- data_cleaned[train_index, ]
test_data <- data_cleaned[-train_index, ]

# LASSO 
library(glmnet)

# Prepare matrices
x_train <- model.matrix(h01 ~ ., data = train_data)[, -1]  # remove intercept
y_train <- train_data$h01

# Lasso for multinomial classification
cv_lasso <- cv.glmnet(x_train, y_train, family = "multinomial", alpha = 1,
                      type.measure = "class", nfolds = 5)

# Get non-zero coefficients
selected_coef <- coef(cv_lasso, s = "lambda.min")
selected_features <- unique(unlist(lapply(selected_coef, function(x) rownames(x)[x[,1] != 0])))
selected_features <- setdiff(selected_features, "(Intercept)")

# Subset to selected features
train_data_selected <- train_data[, c(selected_features, "h01")]
test_data_selected <- test_data[, c(selected_features, "h01")]

library(randomForest)

# Manually define class weights (more weight to minority classes)
# For example, if class 0 is the most frequent, you might assign it lower weight
class_weights <- c(1, 5, 3, 2, 1.5, 0.6, 1.2, 1, 0.9, 0.7, 0.8)

# Train Random Forest model with class weights
rf_model <- randomForest(h01 ~ ., data = train_data_selected, 
                         classwt = class_weights,  # Adjust the weights as needed
                         importance = TRUE)

# Make predictions
rf_pred <- predict(rf_model, newdata = test_data_selected)

# Evaluate model performance
confusionMatrix(rf_pred, test_data_selected$h01)




# ====
# Install and load necessary packages
library(caret)
library(randomForest)

# Manually define class weights (more weight to minority classes)
class_weights <- c(1, 5, 3, 2, 1.5, 0.6, 1.2, 1, 0.9, 0.7, 0.8)

# Set up a training control object (cross-validation with random search)
train_control <- trainControl(method = "cv", number = 5, search = "random")

# Set up a grid of hyperparameters to try (only mtry is required for randomForest)
rf_grid <- expand.grid(
  mtry = c(2, 5, 10, 15)  # Number of variables to sample at each split
)

# Train the Random Forest model with hyperparameter tuning
rf_tuned <- train(h01 ~ ., data = train_data_selected, method = "rf",
                  trControl = train_control, tuneGrid = rf_grid,
                  classwt = class_weights)  # Using class weights

# View the tuning results
print(rf_tuned)

# Make predictions with the tuned model
rf_pred_tuned <- predict(rf_tuned, newdata = test_data_selected)

# Evaluate model performance
confusionMatrix(rf_pred_tuned, test_data_selected$h01)







# before new pipeline =======

# If not installed yet
install.packages("corrplot")

# Load the package
library(corrplot)

# Calculate correlation matrix (only numeric columns)
cor_matrix <- cor(data_cleaned[sapply(data_cleaned, is.numeric)], use = "pairwise.complete.obs")


# Plot heatmap
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.7)

# Save to a larger PNG file
png("correlation_heatmap.png", width = 2000, height = 2000, res = 300)
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.3)  # Adjust label size
dev.off()

# == Model Training ==  
set.seed(123)

# Split data into training and testing sets
train_index <- createDataPartition(data_cleaned$h01, p = 0.8, list = FALSE)
train_data <- data_cleaned[train_index, ]
test_data <- data_cleaned[-train_index, ]

# === Apply SMOTE using smotefamily ===
library(smotefamily)
X_train <- train_data[, -which(names(train_data) == "h01")]
y_train <- train_data$h01

# Apply SMOTE to balance the classes in the training data
smote_output <- SMOTE(X = X_train, target = y_train, K = 5, dup_size = 0)
train_smote <- smote_output$data

# Rename last column to "h01" and convert to factor
names(train_smote)[ncol(train_smote)] <- "h01"
train_smote$h01 <- as.factor(train_smote$h01)

# === Train Random Forest on SMOTE-processed data ===
library(randomForest)

# Train the Random Forest model
rf_model <- randomForest(h01 ~ ., data = train_smote, ntree = 500, mtry = 20, nodesize = 10)

# === Predict using the trained model ===
rf_predictions <- predict(rf_model, test_data)

# === Evaluate performance ===
library(caret)

# Confusion Matrix
conf_matrix <- confusionMatrix(rf_predictions, test_data$h01)
print(conf_matrix)

# === Accuracy ===
accuracy <- sum(rf_predictions == test_data$h01) / length(test_data$h01)
print(paste("Accuracy: ", accuracy))

# === Precision, Recall, and F1 Score for each class ===
# Calculate precision, recall, and F1 for each class using confusion matrix
precision_all <- sapply(levels(test_data$h01), function(class) {
  cm <- confusionMatrix(rf_predictions, test_data$h01)
  cm$byClass[class, "Precision"]
})

recall_all <- sapply(levels(test_data$h01), function(class) {
  cm <- confusionMatrix(rf_predictions, test_data$h01)
  cm$byClass[class, "Recall"]
})

f1_all <- sapply(levels(test_data$h01), function(class) {
  cm <- confusionMatrix(rf_predictions, test_data$h01)
  cm$byClass[class, "F1"]
})

print("Precision for all classes:")
print(precision_all)

print("Recall for all classes:")
print(recall_all)

print("F1 Score for all classes:")
print(f1_all)

# === AUC (Area Under the ROC Curve) ===
library(pROC)

# For multi-class, we use the one-vs-all approach
roc_curve <- multiclass.roc(test_data$h01, as.numeric(rf_predictions))
auc_value <- auc(roc_curve)
print(paste("AUC (macro-average):", auc_value))

# === Accuracy ===
accuracy <- sum(rf_predictions == test_data$h01) / length(test_data$h01)
print(paste("Accuracy: ", accuracy))

# === Precision, Recall, and F1 Score for each class ===
# Calculate precision, recall, and F1 for each class using confusion matrix
precision_all <- sapply(levels(test_data$h01), function(class) {
  cm <- confusionMatrix(rf_predictions, test_data$h01)
  cm$byClass[class, "Precision"]
})

recall_all <- sapply(levels(test_data$h01), function(class) {
  cm <- confusionMatrix(rf_predictions, test_data$h01)
  cm$byClass[class, "Recall"]
})

f1_all <- sapply(levels(test_data$h01), function(class) {
  cm <- confusionMatrix(rf_predictions, test_data$h01)
  cm$byClass[class, "F1"]
})

print("Precision for all classes:")
print(precision_all)

print("Recall for all classes:")
print(recall_all)

print("F1 Score for all classes:")
print(f1_all)

# === AUC (Area Under the ROC Curve) ===
library(pROC)

# For multi-class, we use the one-vs-all approach
roc_curve <- multiclass.roc(test_data$h01, as.numeric(rf_predictions))
auc_value <- auc(roc_curve)
print(paste("AUC (macro-average):", auc_value))



# ==== 
# Convert h01 to numeric (if it's a factor)
data_cleaned$h01_numeric <- as.numeric(as.character(data_cleaned$h01))

# Recode into binary: 0 = environment (1–5), 1 = economy (6–10)
data_cleaned$h01_binary <- ifelse(data_cleaned$h01_numeric <= 5, 0, 1)

# Make binary outcome a factor
data_cleaned$h01_binary <- as.factor(data_cleaned$h01_binary)

library(smotefamily)

# Recode original h01 variable to numeric first (if still a factor)
data_cleaned$h01_numeric <- as.numeric(as.character(data_cleaned$h01))

# Recode to binary target
data_cleaned$h01_binary <- ifelse(data_cleaned$h01_numeric <= 5, 0, 1)
data_cleaned$h01_binary <- as.factor(data_cleaned$h01_binary)

# Split into train/test
set.seed(123)
train_index <- createDataPartition(data_cleaned$h01_binary, p = 0.8, list = FALSE)
train_data <- data_cleaned[train_index, ]
test_data <- data_cleaned[-train_index, ]

# Prepare inputs for SMOTE
X_train <- train_data[, !(names(train_data) %in% c("h01", "h01_numeric", "h01_binary"))]
y_train <- as.numeric(as.character(train_data$h01_binary))  # Must be numeric 0/1

# Apply SMOTE
smote_output <- SMOTE(X = X_train, target = y_train, K = 5, dup_size = 1)
train_smote <- smote_output$data

# Rename and format target to match original column names
train_smote$h01_binary <- as.factor(train_smote$class)
train_smote$class <- NULL  # Remove the original class column as it's no longer needed

# Check balance
table(train_smote$h01_binary)

# Logistic Regression
library(caret)
library(pROC)

# Train Logistic Regression model
log_reg_model <- glm(h01_binary ~ ., data = train_smote, family = binomial)

# Make predictions
log_reg_preds <- predict(log_reg_model, newdata = test_data, type = "response")
log_reg_preds_class <- ifelse(log_reg_preds > 0.5, 1, 0)  # Convert probabilities to binary outcomes

# Evaluate performance using confusion matrix
conf_matrix_log_reg <- confusionMatrix(factor(log_reg_preds_class), factor(test_data$h01_binary))
print(conf_matrix_log_reg)

# Accuracy
accuracy_log_reg <- sum(log_reg_preds_class == test_data$h01_binary) / length(test_data$h01_binary)
print(paste("Accuracy (Logistic Regression):", accuracy_log_reg))

# AUC (Area Under the Curve)
roc_curve_log_reg <- roc(test_data$h01_binary, log_reg_preds)
auc_log_reg <- auc(roc_curve_log_reg)
print(paste("AUC (Logistic Regression):", auc_log_reg))

# Precision, Recall, and F1 score
precision_log_reg <- posPredValue(factor(log_reg_preds_class), factor(test_data$h01_binary))
recall_log_reg <- sensitivity(factor(log_reg_preds_class), factor(test_data$h01_binary))
f1_log_reg <- F1_Score(factor(log_reg_preds_class), factor(test_data$h01_binary))

print(paste("Precision (Logistic Regression):", precision_log_reg))
print(paste("Recall (Logistic Regression):", recall_log_reg))

# Calculate F1 score manually
f1_log_reg <- 2 * (precision_log_reg * recall_log_reg) / (precision_log_reg + recall_log_reg)
print(paste("F1 Score (Logistic Regression):", f1_log_reg))


# Train Random Forest model
library(randomForest)

# Ensure the target variable is a factor
train_smote$target <- as.factor(train_smote$target)

# Train Random Forest model
rf_model <- randomForest(target ~ ., data = train_smote, ntree = 500, mtry = 20, nodesize = 10)

# Make predictions on the test data
rf_preds <- predict(rf_model, newdata = test_data)

# Convert predictions to binary outcomes
rf_preds_class <- as.factor(ifelse(rf_preds > 0.5, 1, 0))

# Evaluate performance using confusion matrix
library(caret)

conf_matrix_rf <- confusionMatrix(rf_preds_class, factor(test_data$h01_binary))
print(conf_matrix_rf)

# Accuracy
accuracy_rf <- sum(rf_preds_class == factor(test_data$h01_binary)) / length(test_data$h01_binary)
print(paste("Accuracy (Random Forest):", accuracy_rf))

# AUC (Area Under the Curve)
library(pROC)
roc_curve_rf <- roc(test_data$h01_binary, as.numeric(rf_preds))
auc_rf <- auc(roc_curve_rf)
print(paste("AUC (Random Forest):", auc_rf))

# Precision, Recall, and F1 score using confusion matrix
precision_rf <- conf_matrix_rf$byClass["Precision"]
recall_rf <- conf_matrix_rf$byClass["Recall"]

# Calculate F1 score manually for Random Forest
f1_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

# Print the results
print(paste("Precision (Random Forest):", precision_rf))
print(paste("Recall (Random Forest):", recall_rf))
print(paste("F1 Score (Random Forest):", f1_rf))


