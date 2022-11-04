import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

y = melbourne_data.Price
X = melbourne_data.drop(columns=['Price'], axis=1, inplace=False)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, train_size=0.8, test_size=0.2)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Get list of categorical variables
s = (train_X.dtypes == 'object')
categorical_cols = list(s[s].index)

# Numerical columns
numerical_cols = list(set(train_X.columns) - set(categorical_cols))

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                              ])

# Preprocessing of training data, fit model
my_pipeline.fit(train_X, train_y)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(val_X)

# Evaluate the model
score = mean_absolute_error(val_y, preds)
print('MAE:', score)