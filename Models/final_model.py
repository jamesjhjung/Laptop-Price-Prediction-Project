# imports
import pandas as pd
import numpy as np
import pickle
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# load data
df = pd.read_csv('data/laptop_price_wf.csv', encoding = 'latin-1', index_col = [0])

# function that preprocesses dataframe
def process_dataframe(input_df):
    # screen resolution
    input_df['Touchscreen'] = np.where(input_df['ScreenResolution'].str.contains('Touchscreen', case = False, na = False), 1, 0)
    input_df['ScreenResolution'] = input_df.apply(lambda x: x['ScreenResolution'][-8:], axis = 1)
    input_df['ScreenResolution'] = input_df['ScreenResolution'].str.replace(' ', '')
    # cpu
    input_df['Cpu'] = input_df['Cpu'].str.replace('Core', '')
    input_df['Cpu'] = input_df['Cpu'].str.split().str[:2].str.join(sep = ' ')
    input_df['Cpu'] = input_df['Cpu'].str.split().str[1:].str.join(sep = ' ')
    # memory
    input_df['ssd'] = np.where(input_df['Memory'].str.contains('ssd|hybrid', case = False, na = False), 1, 0)
    input_df['Memory'] = input_df['Memory'].str.replace('1.0', '1', regex = False)
    input_df['Memory'] = input_df['Memory'].str.replace('1TB', '1024')
    input_df['Memory'] = input_df['Memory'].str.replace('2TB', '2048')
    input_df['Memory'] = input_df['Memory'].str.replace('GB', '')
    input_df['Memory'] = input_df['Memory'].str.split().apply(lambda x: [a for a in x if a.isdigit()])
    input_df['Memory'] = input_df['Memory'].apply(lambda x: [int(a) for a in x])
    input_df['Memory'] = input_df['Memory'].apply(lambda x: sum(x))
    # gpu
    input_df['Gpu'] = input_df['Gpu'].str.split().str[:2].str.join(sep = ' ')
    input_df['Gpu'] = input_df['Gpu'].str.split().str[1:].str.join(sep = ' ')
    # drop columns
    input_df = input_df.drop(['Product', 'laptop_ID'], axis = 1)
    # drop rows
    counts_col2 = input_df.groupby('Cpu')['Cpu'].transform(len)
    counts_col3 = input_df.groupby('Gpu')['Gpu'].transform(len)
    return input_df

# preprocess dataframe
df = process_dataframe(df)

# dropping rows of frequency < 3
counts_col2 = df.groupby('Cpu')['Cpu'].transform(len)
counts_col3 = df.groupby('Gpu')['Gpu'].transform(len)
mask = (counts_col2 > 3) & (counts_col3 > 3)
df = df[mask]

# Data Preprocessing
categorical_preprocessing = Pipeline([('ohe', OneHotEncoder())])
numerical_preprocessing = Pipeline([('stdscaler', StandardScaler())])

# Applying Transformer 
preprocess = ColumnTransformer([
    ('categorical_preprocessing', categorical_preprocessing, ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Ram', 'Gpu', 'OpSys']),
    ('numerical_preprocessing', numerical_preprocessing, ['Inches', 'Memory', 'Weight_fl'])
])

# model pipeline
pipeline = Pipeline([
    ('preprocess', preprocess),
    ('model', RandomForestRegressor(n_estimators = 100, 
                                    max_depth = None, 
                                    min_samples_split = 2, 
                                    min_samples_leaf = 1,
                                    max_features = 'sqrt',
                                    bootstrap = True))])
X = df.drop('Price_euros', axis = 1)
y = df['Price_euros']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 23)
pipeline.fit(X_train, y_train)

# predict laptop price
input_laptop = np.array([['0', 
                         'Asus', 
                         'ZenBook Flip S13 UX371EA', 
                         '2 in 1 Convertible', 
                         13.3, 
                         'Touchscreen 3840x2160', 
                         'Intel Core i7 1165G7 2.8GHz', 
                         '16GB', 
                         '1TB SSD', 
                         'Intel HD Graphics',
                         'Windows 10',
                         0.850,
                         1838.05]])
input_laptop_df = pd.DataFrame(input_laptop, columns = ['laptop_ID','Company','Product','TypeName','Inches','ScreenResolution','Cpu','Ram','Memory','Gpu','OpSys','Weight_fl','Price_euros'])
input_laptop_df = process_dataframe(input_laptop_df)
input_laptop_df = input_laptop_df.drop('Price_euros', axis = 1)

print(pipeline.predict(input_laptop_df))