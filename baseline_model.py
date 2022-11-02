# imports
import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# load data
df = pd.read_csv('data/laptop_price_wf.csv')

# dropping categories with too many unique values
df = df.drop(['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Gpu', 'Memory'], axis = 1)


# Data Preprocessing
categorical_preprocessing = Pipeline([('ohe', OneHotEncoder())])
numerical_preprocessing = Pipeline([('stdscaler', StandardScaler())])

# Applying Transformer 
preprocess = ColumnTransformer([
    ('categorical_preprocessing', categorical_preprocessing, ['Company', 'TypeName', 'Ram', 'OpSys']),
    ('numerical_preprocessing', numerical_preprocessing, ['Inches', 'Weight_fl'])
])

# Creating object for classification
lrm = LinearRegression()

# Final Pipeline
pipeline = Pipeline([
    ('preprocess', preprocess),
    ('lrm', lrm)
])

# Creating Train and Test
X = df.drop('Price_euros', axis = 1)
y = df['Price_euros']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 50)

# Fit pipeline to df_final
pipeline.fit(X_train, y_train)

# Predicting
y_pred = pipeline.predict(X_test)

# Model Performance
print('MSE is: ' + str(mean_squared_error(y_test, y_pred)))
print('RMSE is: ' + str(sqrt(mean_squared_error(y_test, y_pred))))
print('MAE is: ' + str(mean_absolute_error(y_test, y_pred)))