#%%
'''
    Kaggle's Intro to Machine Learning
'''

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = './iowa_housing/AmesHousing.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['Lot Area', 'Year Built', '1st Flr SF', '2nd Flr SF', 'Full Bath', 'Bedroom AbvGr', 'TotRms AbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))
# %%

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
data = {}
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    data[max_leaf_nodes] = mae

smallest_val = min(zip(data.values(), data.keys()))

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = smallest_val[1]

# %%

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)
# %%

# Using Random Forest model

from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
val_pred = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, val_pred)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))
# %%

''' 
    Intermediate Machine Learning 
'''

import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('./housing_price/train.csv', index_col='Id')
X_test_full = pd.read_csv('./housing_price/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
# %%
X_train.head()
# %%

from sklearn.ensemble import RandomForestRegressor
'''
    Determine the best model to use
'''

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))
# %%
my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
my_model.fit(X, y)

pred = my_model.predict(X_test)
X_test.head()
# %%
# How to create submission file.
submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': pred})
submission.to_csv('submission.csv', index=False)

# %%
