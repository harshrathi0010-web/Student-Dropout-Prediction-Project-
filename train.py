# Import required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
df=pd.read_csv(r"../data/student.csv")

# Feature selection and data preprocessing
df=df.drop(columns=["Student_ID"])

# Divide into features and target variable
X=df.drop(columns=["Dropout"])
y=df["Dropout"]



