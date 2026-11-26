# Import required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression
import joblib

# Loading the dataset
df=pd.read_csv("data/student.csv")

# Feature selection and data preprocessing
df=df.drop(columns=["Student_ID"])

# Divide into features and target variable
X=df.drop(columns=["Dropout"])
y=df["Dropout"]


# Extract cateogorical and numerical pipelines
cat_cols=X.select_dtypes(include="object").columns

num_cols=X.select_dtypes(include=np.number).columns

# Handling outliers in Family_Income column
X.Family_Income=X.Family_Income.map(lambda val: 100000 if val>100000 else val)

# Pipelines for categorical and numerical features
cat_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])


num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="mean")), 
    ("scaler",StandardScaler())
])


Preprocessor=ColumnTransformer(transformers=[
    ("cat",cat_pipeline,cat_cols),
    ("num",num_pipeline,num_cols)
])

model=Pipeline([
    ("preprocessor",Preprocessor),
    ("classifier",LogisticRegression())
])


# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=32)

# Model training 
model.fit(X_train,y_train)

# save the model
joblib.dump(model,"models/dropout_models.pkl")


# Model prediction
y_pred=model.predict(X_test)

# Model Evaluation
acc=accuracy_score(y_test,y_pred)
cr=classification_report(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)

print("Accuracy Score",acc)

file=open("models/model_evaluation.txt","w")
file.write(f"Accuracy:{acc}\n")
file.write(f"Confusion matrix:\n{cm}\n")
file.write(f"Classification report:\n{cr}\n")
file.close()

print("Model evaluation metrics to models/evaluation.text")
print("Model saved to models/dropout_model.pkl")

print("Model Training Complete successfully")





