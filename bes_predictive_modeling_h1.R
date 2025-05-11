# ==== Data Preprocessing ====
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











# ==== Multi Classification Random Forest ====
library(caret)
library(yardstick)
library(dplyr)
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

# Convert to tibble for yardstick
results_df <- tibble(
  truth = factor(y_true),
  prediction = factor(y_pred, levels = levels(y_true))
)

# Compute metrics
precision_macro <- precision(results_df, truth = truth, estimate = prediction, average = "macro")
f1_macro <- f_meas(results_df, truth = truth, estimate = prediction, average = "macro")

# Display results
cat("Macro Precision:", round(precision_macro$.estimate, 3), "\n")
cat("Macro F1 Score:", round(f1_macro$.estimate, 3), "\n")
# ==== Binary Logistic Regression ====
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

print(paste("Precision (Logistic Regression):", precision_log_reg))

# Calculate F1 score manually
f1_log_reg <- 2 * (precision_log_reg * recall_log_reg) / (precision_log_reg + recall_log_reg)
print(paste("F1 Score (Logistic Regression):", f1_log_reg))