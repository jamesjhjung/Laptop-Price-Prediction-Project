# imports
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load data
df = pd.read_csv('data/laptop_price_wf.csv')

# create mean price column
df['mean_price'] = df['Price_euros'].mean()

# Creating Train and Test
X = df.drop(['Price_euros', 'mean_price'], axis = 1)
y = df['Price_euros']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 50)

# y_pred
y_pred = df['mean_price'].head(len(y_test))

# calculate MSE, RMSE, MAE with mean price
print('MSE is: ' + str(mean_squared_error(y_test, y_pred)))
print('RMSE is: ' + str(math.sqrt(mean_squared_error(y_test, y_pred))))
print('MAE is: ' + str(mean_absolute_error(y_test, y_pred)))