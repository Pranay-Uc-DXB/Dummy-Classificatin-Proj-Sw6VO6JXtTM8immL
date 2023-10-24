# %%
import pickle


# Load from file
with open('KNN_model.pkl', 'rb') as file:
    Script = pickle.load(file)

Script

# %%
def Predictor(model,test_data):
    Y_pred=model.predict(test_data)
    return Y_pred

# %%
import pandas as pd
import numpy as np

# %%
# loading data

data=pd.read_csv(r"../data/ACME-HappinessSurvey2020.csv")

# %%
X=data.drop(['Y','X2','X4'],axis=1)
X.shape
Y=data['Y']
Y.shape

# %%
from sklearn.model_selection import train_test_split

#Splitting data for training and testing
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, random_state=42, test_size=0.15)

# %%
Predictions=Predictor(Script,X_test)
Predictions


