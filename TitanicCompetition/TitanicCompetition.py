import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

y = train_data["Survived"]

# Data visualization
sns.pairplot(train_data)
plt.show()

# Plotting Missing values using SEABORN
sns.heatmap(train_data.isnull(), cbar=True)
plt.show()

# Data pre-processing
train_data['Sex'], train_data['Embarked'] = train_data['Sex'].astype('category'), \
                                            train_data['Embarked'].astype('category')
test_data['Sex'], test_data['Embarked'] = test_data['Sex'].astype('category'), \
                                            test_data['Embarked'].astype('category')
test_data["Fare"], train_data["Fare"] = test_data["Fare"].fillna(0.0), train_data["Fare"].fillna(0.0)

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)
y_pred = model.predict(X)
# Accuracy
cls_rpt = classification_report(y, y_pred, labels=None, target_names=None, sample_weight=None, digits=2,
                                output_dict=False)
print(cls_rpt)

# Naive Bayes
mnb = MultinomialNB()
mnb.fit(X, y)
y_pred = mnb.predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y != y_pred).sum()))

# SVM
clf = svm.SVC()
clf.fit(X, y)
y_pred = clf.predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y != y_pred).sum()))

# more data exploration
for i, col in enumerate (['SibSp', 'Parch']):
    plt.figure (i)
    sns.catplot (x=col, y='Survived', data = train_data, kind = 'point', aspect =2,)

# similar so combine and drop
train_data['Family_Count'] = train_data['Parch']+train_data['SibSp']
test_data['Family_Count'] = test_data['Parch']+test_data['SibSp']
train_data = train_data.drop(['Name','Ticket','SibSp','Parch'], axis=1)
test_data = test_data.drop(['Name','Ticket','SibSp','Parch'], axis=1)
train_data.head()

# Hard coding categorical variables (for 1 being Male and 0 being Female using a Dictionary)
gender_num = {'male':1,'female':0}
train_data['Sex'] = train_data['Sex'].map (gender_num)
test_data['Sex'] = test_data['Sex'].map (gender_num)

# Dropping Embarked, Cabin from the titanic dataframe.
train_data.drop(['Embarked','Cabin'], axis=1,inplace=True)
test_data.drop(['Embarked','Cabin'], axis=1,inplace=True)
train_data.dropna(inplace=True)
train_data.reset_index(inplace=True,drop=True)
test_data.dropna(inplace=True)
test_data.reset_index(inplace=True,drop=True)


# Regression
X = pd.get_dummies(train_data.iloc[:,2:])
X_test = pd.get_dummies(test_data.iloc[:,1:])
y = train_data["Survived"]

LogModel = LogisticRegression(solver='liblinear')
LogModel.fit(X,y)
predictions = LogModel.predict(X_test)
y_pred = LogModel.predict(X)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y != y_pred).sum()))

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Submission was successfully saved!")