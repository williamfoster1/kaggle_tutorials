import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
%matplotlib inline

melbourne_file_path = './melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Select subset of features
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = melbourne_data[cols_to_use]

# Select target
y = melbourne_data.Price

def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
                              ])
    # Multiply by -1 since sklearn calcs -ive MAR
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'MAE scores: {scores}')
    return scores.mean()

results = {i: get_score(i) for i in range(50, 450, 50)} # Your code here



plt.plot(list(results.keys()), list(results.values()))
plt.show()