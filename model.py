import pandas as pd
import numpy as np
from scipy.stats import levene
from scipy.stats import shapiro
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report,precision_score,f1_score,recall_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.ensemble import RandomForestClassifier



import warnings
warnings.filterwarnings('ignore')
Personality = pd.read_csv("personality_data.csv")
Personality["Total"] = Personality["openness"] + Personality["neuroticism"] + \
Personality["conscientiousness"] + Personality["agreeableness"] + Personality["extraversion"]

data = Personality.copy()
encode = LabelEncoder()
columns = ["gender","Personality"]
for i in columns:
    print(data[i].value_counts())
    data[i] = encode.fit_transform(data[i])
    print(data[i].value_counts())
input_cols = ['gender', 'age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']
output_cols = ['Personality']
scaler = StandardScaler()
data[input_cols] = scaler.fit_transform(data[input_cols])
X = data[input_cols]
Y = data[output_cols]
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.15, random_state=250)
lr = LogisticRegression(multi_class='auto', solver='lbfgs',max_iter =1000)
lr.fit(X, Y)
ypred= lr.predict(xTest)
print(accuracy_score(yTest,ypred)*100)

joblib.dump(lr, "train_model.pkl")    
