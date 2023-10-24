# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from sklearn.metrics import confusion_matrix, classification_report,f1_score,accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier

# %%
#Splitting data for training and testing
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, random_state=42, test_size=0.15)

# %%
KNN2=KNeighborsClassifier()

KNN2_param_grid={'n_neighbors':[2,5,7,9,11,13],'weights':['uniform','distance']}
GS_KNN=GridSearchCV(KNN2,param_grid=KNN2_param_grid, cv=3)

GS_KNN.fit(X_train,Y_train)

# %%
print(GS_KNN.best_params_)

GS_KNN_y_pred=GS_KNN.predict(X_test)

# %%
print(classification_report(Y_test,GS_KNN_y_pred))

CM3=confusion_matrix(Y_test,GS_KNN_y_pred)
sns.heatmap(CM3,annot=True)

# %%
import pickle

# Save to file
with open('KNN_model.pkl', 'wb') as file:
    pickle.dump(GS_KNN, file)



