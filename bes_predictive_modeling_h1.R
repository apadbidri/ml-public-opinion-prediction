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

cols_to_remove <- c("a01", "k04", "Constit_Code", "Constit_Name", 
                    "LA_UA_Code", "LA_UA_Name", "interviewer")

data_cleaned <- data_cleaned %>%
  select(-all_of(cols_to_remove))





# Check if there are any remaining missing values
sum(is.na(data_cleaned))




# -------- Model building
# Load necessary libraries
library(glmnet)
library(caret)
library(dplyr)
library(tibble)

# Set seed for reproducibility
set.seed(123)

# Split data into training and testing (80/20)
train_index <- createDataPartition(data_cleaned$h01, p = 0.8, list = FALSE)
train_data <- data_cleaned[train_index, ]
test_data <- data_cleaned[-train_index, ]

# Convert data to model.matrix format (Lasso requires numeric matrix input)
x_train <- model.matrix(h01 ~ ., data = train_data)[, -1]  # Remove intercept
y_train <- train_data$h01

x_test <- model.matrix(h01 ~ ., data = test_data)[, -1]
y_test <- test_data$h01

# Fit Lasso with cross-validation to find optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, standardize = TRUE)

# Get best lambda
best_lambda <- cv_lasso$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Get coefficients of the selected model
lasso_coef <- coef(cv_lasso, s = best_lambda)
selected_features <- rownames(lasso_coef)[lasso_coef[, 1] != 0]
selected_features <- selected_features[!selected_features %in% "(Intercept)"]
cat("Selected features:\n")
print(selected_features)

# Ensure only the columns that exist in train_data are included in the model
selected_features <- selected_features[selected_features %in% colnames(train_data)]

# Subset the original training/testing data to selected features
train_selected <- train_data[, c("h01", selected_features)]
test_selected <- test_data[, c("h01", selected_features)]

# Train linear model on selected features
lm_model <- lm(h01 ~ ., data = train_selected)

# Predict on test set
predictions <- predict(lm_model, newdata = test_selected)

# Evaluate performance
rmse_val <- RMSE(predictions, test_selected$h01)
mae_val <- MAE(predictions, test_selected$h01)
r2_val <- R2(predictions, test_selected$h01)

cat("RMSE:", round(rmse_val, 3), "\n")
cat("MAE:", round(mae_val, 3), "\n")
cat("R-squared:", round(r2_val, 3), "\n")

