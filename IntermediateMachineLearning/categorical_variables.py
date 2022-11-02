import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price
X = melbourne_data.drop(columns=['Price'], axis=1, inplace=False)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)

# Get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)
print(f'Categorical variables {object_cols}')

# Function for comparing different approaches
def score_dataset(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(val_X)
    return mean_absolute_error(val_y, preds)

# Approach 1: Drop categorical variables
drop_train_X = train_X.select_dtypes(exclude=['object'])
drop_val_X = val_X.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_train_X, drop_val_X, train_y, val_y))

label_train_X = train_X.copy()
label_valid_X = val_X.copy()

# assign each value a value in a single column
ordinal_encoder = OrdinalEncoder()
label_train_X[object_cols] = ordinal_encoder.fit_transform(train_X[object_cols])
label_valid_X[object_cols] = ordinal_encoder.fit_transform(val_X[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_train_X, label_valid_X, train_y, val_y))

# assign each value a column with values 1 or 0
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(val_X[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = train_X.index
OH_cols_valid.index = val_X.index

# Remove categorical columns (will replace with one-hot encoding)
num_train_X = train_X.drop(object_cols, axis=1)
num_val_X = val_X.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_train_X, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_val_X, OH_cols_valid], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):")
print(score_dataset(OH_X_train, OH_X_valid, train_y, val_y))

# Categorical columns in the training data
object_cols = [col for col in train_X.columns if train_X[col].dtype == "object"]

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if
                   set(val_X[col]).issubset(set(train_X[col]))]

# Problematic columns that will be dropped from the dataset
bad_label_cols = list(set(object_cols) - set(good_label_cols))

print('Categorical columns that will be ordinal encoded:', good_label_cols)
print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

# Cardinality
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: train_X[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])

# cardinality is important and typically we only one hot encode columns with low cardinality and ordinally encode high cardinality
# features

# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))

print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

