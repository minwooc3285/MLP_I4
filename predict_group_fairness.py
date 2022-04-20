#%%
#Load the librarys
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt #to plot some parameters in seaborn
import pdb
############################################################################################### 데이터 불러오기 
#Importing the data
df_credit = pd.read_csv("./input/german-credit-data-with-risk-with-target/german_credit_data.csv",index_col=0)

############################################################################################### 데이터 수정하기

######################################################################## 카테고리화
#Let's look the Credit Amount column
# interval = (18, 25, 35, 60, 120)

# cats = ['Student', 'Young', 'Adult', 'Senior']

interval = (18, 25, 35, 120)

cats = ['Student', 'Young', 'Adult']
df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)


df_good = df_credit[df_credit["Risk"] == 'good']
df_bad = df_credit[df_credit["Risk"] == 'bad']


######################################################################## one-hot 인코딩
#pre-processing (OHE)
def one_hot_encoder(df, nan_as_category = False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

######################################################################## 다듬기

df_credit['Saving accounts'] = df_credit['Saving accounts'].fillna('no_inf')
df_credit['Checking account'] = df_credit['Checking account'].fillna('no_inf')

df_credit = df_credit.merge(pd.get_dummies(df_credit.Purpose, drop_first=True, prefix='Purpose'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit.Sex, drop_first=True, prefix='Sex'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit.Housing, drop_first=True, prefix='Housing'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix='Savings'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit.Risk, prefix='Risk'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix='Check'), left_index=True, right_index=True)
df_credit = df_credit.merge(pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix='Age_cat'), left_index=True, right_index=True)

######################################################################## 기존 데이터 제거

#Excluding the missing columns
del df_credit["Saving accounts"]
del df_credit["Checking account"]
del df_credit["Purpose"]
del df_credit["Sex"]
del df_credit["Housing"]
del df_credit["Age_cat"]
del df_credit["Age"]
del df_credit["Risk"]
del df_credit['Risk_good']

###############################################################################################

from sklearn.model_selection import train_test_split, KFold, cross_val_score # to split the data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, fbeta_score, roc_auc_score #To evaluate our model

from sklearn.model_selection import GridSearchCV

# Algorithmns models to be compared
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
df_credit['Credit amount'] = np.log(df_credit['Credit amount'])
############################################################################################### 모델 학습

######################################################################## 데이터 나누기
#Creating the X and y variables
X = df_credit.drop('Risk_bad', 1).values
y = df_credit["Risk_bad"].values
#pdb.set_trace()
# print('headers',df_credit.keys())
# print('headers',df_credit.head())
# print('X.shape',X.shape)
# print('y.shape',y.shape)

# print('X',X)
# print('y',y)

# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

# to feed the random state
seed = 7

# ######################################################################## 모델 학습 및 평가 (8개 모델 중에 GNB 만!)

from sklearn.utils import resample
from sklearn.metrics import roc_curve

# # Criando o classificador logreg
# GNB = GaussianNB()
# # Fitting with train data
# model = GNB.fit(X_train, y_train)
# # Printing the Training Score
# print("Training score data: ")
# print(model.score(X_train, y_train))

# y_pred = model.predict(X_test)
# print("\n accuracy_score")
# print(accuracy_score(y_test,y_pred))
# print("\n confusion matrix")
# print(confusion_matrix(y_test, y_pred))
# print("\n Classification report")
# print(classification_report(y_test, y_pred))

# #Predicting proba
# y_pred_prob = model.predict_proba(X_test)[:,1]

# # ######################################################################## ROC 커브 그리기
# # Generate ROC curve values: fpr, tpr, thresholds
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# # Plot ROC curve
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
# plt.show()

###############################################################################################

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

################################### 파이프라인 구조 ############################################
################# PCA --> feature selection --> feature union --> GNB #########################
features = []
features.append(('pca', PCA(n_components=2)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)
# create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', RandomForestClassifier(random_state=2)))
model = Pipeline(estimators)
# evaluate pipeline
seed = 7
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X_train, y_train, cv=kfold)
print("\n results mean")
print(results.mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

######################################################################## group fairness

####################################### Age 

################## Age Young

is_young = df_credit['Age_cat_Young'] == 1
df_credit_young = df_credit[is_young]

X = df_credit_young.drop('Risk_bad', 1).values
y = df_credit_young["Risk_bad"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Prob. of Age of young', ( sum(y_pred == 1) / len(y_pred)))

################## Age Adult

is_adult = df_credit['Age_cat_Adult'] == 1
df_credit_adult = df_credit[is_adult]
X = df_credit_adult.drop('Risk_bad', 1).values
y = df_credit_adult["Risk_bad"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Prob. of Age of Adult', ( sum(y_pred == 1) / len(y_pred)))

####################################### Sex

################## Sex male

is_male = df_credit['Sex_male'] == 1
df_credit_male = df_credit[is_male]
X = df_credit_male.drop('Risk_bad', 1).values
y = df_credit_male["Risk_bad"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Prob. of Sex of Male', ( sum(y_pred == 1) / len(y_pred)))

################## Sex female

is_female = df_credit['Sex_male'] == 0
df_credit_female = df_credit[is_female]
X = df_credit_female.drop('Risk_bad', 1).values
y = df_credit_female["Risk_bad"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Prob. of Sex of Female', ( sum(y_pred == 1) / len(y_pred)))


######################################################################## evaluate metrics



# %%
