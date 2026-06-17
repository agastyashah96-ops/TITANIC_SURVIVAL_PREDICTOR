import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib

'''


    ALL COLUMNS- 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],

    NULL columns - 1 Embarked     889 non-null    str 
    2 Cabin        204 non-null    str
    3  Age          714 non-null    float64

negative correlation between Age and survival is -6.0 %
negative correlation between Pclass and survival is -34.0 %
positive correlation between Fare and survival is 26.0 %
positive correlation between Has_Cabin and survival is 32.0 %
negative correlation between SibSp and survival is -4.0 %
negative correlation between Sex and survival is -54.0 %
positive correlation between Embarked_from_C and survival is 17.0 %
positive correlation between Embarked_from_Q and survival is 0.0 %
negative correlation between Embarked_from_S and survival is -15.0 %
    
'''
def data_cleaning():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, 'TITANIC.csv'))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    #FILLED NULL VALUES IN EMBARKED COLUMNS WITH THE MOST FREQUENT CITY
    df['Age'] = df['Age'].fillna(df['Age'].median())
    #FILLED NULL VALUES IN AGE COLUMNS WITH THE MEDIAN AGE
    df['Has_Cabin'] = df['Cabin'].notnull().astype(int)
    #DROPPED THE CABIN COLUMN FOR HAS_CABIN BECAUSE OF A LOT OF NON VALUE
    df.drop(columns=['Cabin'], inplace=True)

    df['Sex'] = df['Sex'].map( #ENCODING MALE AND FEMALE
        {
        'male':1,
        'female':0}
    )

    df =  pd.get_dummies(df, columns=['Embarked'], prefix='Embarked_from', dtype=bool)
    return df

df = data_cleaning()

x = df[['Age', 'Pclass', 'Fare', 'Has_Cabin', 'SibSp', 'Parch', 'Sex',
        'Embarked_from_C', 'Embarked_from_Q', 'Embarked_from_S']]
y = df['Survived']
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))



pct_matrix = pd.crosstab(df['Sex'], df['Survived'],normalize='columns') * 100

pct_matrix.index = pct_matrix.index.map({0: 'Female', 1: 'Male'})

print(pct_matrix)




def survival_corr(survival_feat): #correlates the variable most neededfor survival
    for i in survival_feat:
        corr = survival_feat[i].corr(df['Survived']) * 100
        if corr >=0:
           print(f'positive correlation between {i} and survival is {(corr).round()} %')
        else:
           print(f'negative correlation between {i} and survival is {(corr).round()} %')




# Save the model and scaler to disk
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')










