import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#this is what we want to predict
#y = train['Survived']

#these are the things we will use to predict the above
#train_features = ['Age', 'Gender', 'Class']
#X = train_data[train_features]

def process_age(df, cut_points, label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points, labels=label_names)
    return df

cut_points = [-1, 0, 5, 12, 18, 35, 60, 100]
label_names = ["Missing", "Infant", "Child", "Teenager", "Young Adult", "Adult", "Senior"]

train = process_age(train, cut_points, label_names)
test = process_age(test, cut_points, label_names)

def create_dummies(df, column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies], axis=1)
    return df

train = create_dummies(train, "Pclass")
test = create_dummies(test, "Pclass")

train = create_dummies(train, "Sex")
test = create_dummies(test, "Sex")

train = create_dummies(train, "Age_categories")
test = create_dummies(test, "Age_categories")

#the model with which we will use to train
lr = LogisticRegression()
gnb = GaussianNB()
clf = DecisionTreeClassifier()
rfc = RandomForestClassifier(n_estimators=100)
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)

columns = ["Pclass_1", "Pclass_2", "Pclass_3", "Sex_female", "Sex_male", "Age_categories_Missing", "Age_categories_Infant", "Age_categories_Child", "Age_categories_Teenager",
           "Age_categories_Young Adult", "Age_categories_Adult", "Age_categories_Senior"]

all_X = train[columns]
all_y = train["Survived"]

train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=0)

lr.fit(train_X, train_y)
gnb.fit(train_X, train_y)
clf.fit(train_X, train_y)
rfc.fit(train_X, train_y)
mlp.fit(train_X, train_y)
knn.fit(train_X, train_y)
predictions = lr.predict(test_X)
gnb_pred = gnb.predict(test_X)
tree_pred = clf.predict(test_X)
for_pred = rfc.predict(test_X)
mlp_pred = mlp.predict(test_X)
knn_pred = knn.predict(test_X)
accuracy = accuracy_score(test_y, predictions)
gnb_acc = accuracy_score(test_y, gnb_pred)
tree_acc = accuracy_score(test_y, tree_pred)
for_acc = accuracy_score(test_y, for_pred)
mlp_acc = accuracy_score(test_y, mlp_pred)
knn_acc = accuracy_score(test_y, knn_pred)
print(accuracy)
print(gnb_acc)
print(tree_acc)
print(for_acc)
print(mlp_acc)
print(knn_acc)

feature_imp = pd.Series(rfc.feature_importances_, index=columns).sort_values(ascending=False)
print(feature_imp)

#from here do the same as above but on entire data set instead of just the training section
#to create submission file kaggle expects 2 columns, survived and passenger iD
#holdouts_id = holdout["PassengerID"]
#submission_df = {"PassengerID: holdouts_ids, "Survived": holdout_predictions}
#submission = pd.DataFrame(submission_DF)
#submission.to_csv('titanic_submission.csv', index=False)

knn.fit(all_X, all_y)
holdout_predictions = knn.predict(test[columns])
holdouts_ids = test["PassengerId"]
submission_df = {"PassengerId": holdouts_ids, "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv('titanic_submission.csv', index=False)