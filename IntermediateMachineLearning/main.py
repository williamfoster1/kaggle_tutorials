import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Get names of columns with missing values
cols_with_missing = [col for col in train_X.columns
                     if train_X[col].isnull().any()]

# Approach 1: Drop columns with missing data
# Drop columns in training and validation data
# reduced_train_X = train_X.drop(cols_with_missing, axis=1)
# reduced_val_X = val_X.drop(cols_with_missing, axis=1)
# X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

# To keep things simple, we'll use only numerical predictors
#X = X_full.select_dtypes(exclude=['object'])

# Approach 2: Imputation - give missing data mean
#my_imputer = SimpleImputer()
#imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X))
#imputed_val_X = pd.DataFrame(my_imputer.fit_transform(val_X))

# Imputation removed column names; put them back
#imputed_train_X.columns = train_X.columns
#imputed_val_X.columns = val_X.columns

# Approach 3: Imputation with column signaling missing data was present
train_X_plus = train_X.copy()
val_X_plus = val_X.copy()

for col in cols_with_missing:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    val_X_plus[col + '_was_missing'] = val_X_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_val_X_plus = pd.DataFrame(my_imputer.fit_transform(val_X_plus))

imputed_train_X_plus.columns = train_X.columns
imputed_val_X_plus.columns = val_X.columns

print("MAE from Approach 3 (Drop columns with missing values):")
print(score_dataset(imputed_train_X_plus, imputed_val_X_plus, train_y, val_y))

# Shape of training data
print(train_X.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

