import pandas as pd
import joblib  # used to save and load trained models
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# Load the CSV file for the training data into a pandas dataframe
df_train = pd.read_csv('./data/Train_extra.csv')

# Extract the features and labels from the training data dataframe
X_train = df_train.drop(columns=['label'])  # features
y_train = df_train['label']  # labels

# Load the CSV file for the test data into a pandas dataframe
df_test = pd.read_csv('./data/Test_extra.csv')

# Extract the features and labels from the test data dataframe
X_test = df_test.drop(columns=['label'])  # features
y_test = df_test['label']  # labels

# Create a random forest classifier
clf = RandomForestClassifier()

# Create a support vector machine classifier
clf2 = svm.SVC()

# Train the classifier on the training data
clf.fit(X_train, y_train)

#TRAINING THE SVM
clf2.fit(X_train, y_train)


# Test the classifier on the test data
accuracy = clf.score(X_test, y_test)
print(f'Test accuracy for RF: {accuracy:.2f}')

#TESTING THE SVM
accuracy2 = clf2.score(X_test, y_test)
print(f'Test accuracy for SVM: {accuracy2:.2f}')

# Save the trained model to a file
joblib.dump(clf, 'trained_model_RF_extra.pkl')

# Save the trained model to a file
joblib.dump(clf2, 'trained_model_SVM_extra.pkl')
