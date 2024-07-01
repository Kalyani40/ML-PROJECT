import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("yield.csv")
df.head()
df.columns
col = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp','Area', 'Item', 'hg/ha_yield']
df = df[col]
df.head()
X = df.drop('hg/ha_yield', axis = 1)
y = df['hg/ha_yield']
X.shape
y.shape

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0, shuffle=True)


ohe = OneHotEncoder(drop = 'first')
scale = StandardScaler()

preprocessor = ColumnTransformer(
    transformers = [
        ('StandardScale', scale, [0,1,2,3]),
        ('OneHotEncode', ohe, [4,5])
    ], 
    remainder = 'passthrough'
)
X_train_dummy = preprocessor.fit_transform(X_train)
X_test_dummy  = preprocessor.fit_transform(X_test)
preprocessor.get_feature_names_out(col[:-1])
pickle.dump(preprocessor, open("preprocessor.pkl","wb"))
