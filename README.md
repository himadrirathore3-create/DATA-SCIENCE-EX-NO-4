# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT
```
        FEATURE SCALING
import pandas as pd
from scipy import stats
import numpy as np
```
```
df=pd.read_csv("/content/bmi.csv")
df.head()
```


## OUTPUT
<img width="933" height="488" alt="image" src="https://github.com/user-attachments/assets/581d131f-3c11-468c-a510-af69da161152" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
## OUTPUT
<img width="733" height="375" alt="image" src="https://github.com/user-attachments/assets/61e30a02-41c6-4e2f-984b-087fcc7e99d5" />


```
df.dropna()
```
## OUTPUT
<img width="1449" height="664" alt="image" src="https://github.com/user-attachments/assets/59c9b145-a4fe-42f3-a15e-15b5acf72079" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).

```
## OUTPUT
<img width="907" height="427" alt="image" src="https://github.com/user-attachments/assets/ae4982e7-4217-4d7e-b417-e667993ab4d9" />

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```
## OUTPUT
<img width="1296" height="474" alt="image" src="https://github.com/user-attachments/assets/aa0e61f1-a3bd-4def-a58f-f309d9b8b7fd" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```


## OUTPUT
<img width="1441" height="670" alt="image" src="https://github.com/user-attachments/assets/7838fbc9-6c89-4ef9-be95-8e6661edaa64" />


```
#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```


## OUTPUT
<img width="1281" height="688" alt="image" src="https://github.com/user-attachments/assets/8f3d09c5-a82e-4553-915a-38cb4b0558ce" />


```
#MAXIMUM ABSOLUTE SCALING:
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
```


## OUTPUT
<img width="980" height="531" alt="image" src="https://github.com/user-attachments/assets/4a1411bf-41d9-48fb-b2c0-6f8cee671118" />



```
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```



## OUTPUT
<img width="1316" height="680" alt="image" src="https://github.com/user-attachments/assets/4c189bc0-a918-4a16-8798-b125182ba565" />


```
#ROBUST SCALING
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()


```


## OUTPUT
<img width="993" height="541" alt="image" src="https://github.com/user-attachments/assets/1ee82969-6d99-44c7-ab21-75453c044d48" />


```
#FEATURE SELECTION:
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
## OUTPUT
<img width="1195" height="636" alt="image" src="https://github.com/user-attachments/assets/b8bb3cc0-86c0-417f-9bb1-d2bbb7620d7a" />


```
df_null_sum=df.isnull().sum()
df_null_sum
```


## OUTPUT

<img width="741" height="761" alt="image" src="https://github.com/user-attachments/assets/c30d5a89-022a-4443-a6d0-b74b058d46eb" />

```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
```

```
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
# process non-numeric features. It ensures the model treats these columns as categories,
# not as continuous numerical values.
df[categorical_columns]
```
## OUTPUT
<img width="1721" height="883" alt="image" src="https://github.com/user-attachments/assets/7708a646-4a3f-4abe-99d9-952b406546a9" />


```
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
```

## OUTPUT
<img width="1259" height="742" alt="image" src="https://github.com/user-attachments/assets/43b37d63-95a7-4329-8455-231decaf2402" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict
```


```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```

## OUTPUT
<img width="1295" height="330" alt="image" src="https://github.com/user-attachments/assets/f47ed77e-7798-4fa8-b11d-2a3d9a796626" />

```
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```


## OUTPUT
<img width="1003" height="628" alt="image" src="https://github.com/user-attachments/assets/8aeee1d2-7b4b-4cf9-9b24-617ead728bc2" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```

## OUTPUT
<img width="1453" height="760" alt="image" src="https://github.com/user-attachments/assets/509d7742-e0ac-48be-904a-f48f5a7aa789" />


```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```

## OUTPUT

<img width="1247" height="669" alt="image" src="https://github.com/user-attachments/assets/a57aac5d-435e-4c95-a99b-96c1aea535bb" />

```

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
```


## OUTPUT

<img width="1209" height="406" alt="image" src="https://github.com/user-attachments/assets/94e9ade3-33a5-4449-bdc5-58c72f0f0a62" />


```

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```


## OUTPUT

<img width="1313" height="470" alt="image" src="https://github.com/user-attachments/assets/43a157c7-f314-4e86-ab27-53da15a254f1" />


```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```


## OUTPUT

<img width="1235" height="193" alt="image" src="https://github.com/user-attachments/assets/4c1fb452-e5d1-4001-ab3e-b4a18344c0b1" />


```
!pip install skfeature-chappers
```


## OUTPUT

<img width="1766" height="485" alt="image" src="https://github.com/user-attachments/assets/b3de7d8a-3bcd-450f-b054-7ed0d5c879af" />


```
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

```
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
## OUTPUT

<img width="1359" height="780" alt="image" src="https://github.com/user-attachments/assets/1c020df4-6f23-43f1-a564-742813cafc50" />

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
```
```
selected_features_anova = X.columns[selector_anova.get_support()]
```
```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
## OUTPUT

<img width="1329" height="191" alt="image" src="https://github.com/user-attachments/assets/3b3ddcff-1347-45bd-a0de-c682721a6812" />



```
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
```

```
# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
```
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
```
```
df[categorical_columns]
```
## OUTPUT

<img width="1496" height="679" alt="image" src="https://github.com/user-attachments/assets/2030b6ef-7828-406d-93d2-097f8ac96beb" />


```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
```


```
logreg = LogisticRegression()
```
```
n_features_to_select =6
```
```
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
## OUTPUT

<img width="1655" height="702" alt="image" src="https://github.com/user-attachments/assets/7ed7cf84-dcb9-4ce3-9d26-029b508b519d" />

<img width="1777" height="793" alt="image" src="https://github.com/user-attachments/assets/d49f5b8d-7ddc-46be-87d7-bb4601c68f16" />


# RESULT:
      
       
       Given code is executed successfully...
