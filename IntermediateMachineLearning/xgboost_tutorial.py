import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
data = pd.read_csv('./melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# n_estimators - how many times to go through the loss function minimization cycle
# early_stopping_rounds - how many rounds of deterioration to allow
# before stopping i.e. loss function does keep getting minimized
# learning_rate - factor to multiply predictions by each time. Causes each additional model to help less
# General approach is high estimators with a learning rate but this isn't as performant
# n_jobs - how many jobs to parallelise into to aid "fit time" - bound by number of computer cores
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, early_stopping_rounds=5)
my_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))