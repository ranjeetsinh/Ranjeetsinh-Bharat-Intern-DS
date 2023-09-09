# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# Importing the dataset
data = pd.read_csv(r"C:\Users\HP\Desktop\proj 1\Titanic-Dataset.csv")
data.head()

# No. of Rows and Columns
print(data.shape)

# No. of missing values
print(data.isnull().sum())

# Drop the cabin column from the df
data = data.drop(columns='Cabin', axis=1)

# Replacing the missing values in Age column with mean
print(data["Age"].fillna(data["Age"].mean(), inplace=True))

# Replacing the missing values in Embarked column with
print(data["Embarked"].mode())

print(data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True))
print(data.isnull().sum())

print(data.describe())

# DATA VISUALIZATION
print(sns.set())

# Count plot on the basis of "Gender"
plt.figure(figsize=(6, 4))
print(sns.countplot(data=data, x="Sex"))
plt.title("Passenger Gender Distribution")
plt.show()

# Count plot on the basis of "Survived"

plt.figure(figsize=(6, 4))
print(sns.countplot(data=data, x="Survived"))
plt.title("Survival Distribution")
plt.show()

# Count plot on the basis of "Survived" and  "Gender"
plt.figure(figsize=(6, 4))
print(sns.countplot(data=data, x="Survived", hue="Sex"))
plt.title("Survival by Gender")
plt.show()

# ENCODING
# to know the value counts as per Gender
data["Sex"].value_counts()

# to know the value counts as per Embarked
data["Embarked"].value_counts()

# Encoding the categorical columns
data.replace({"Sex": {"male": 0, "female": 1}, "Embarked": {"S": 0, "C": 1, "Q": 2}}, inplace=True)
data.head()

# SEPARATING FEATURES
x = data.drop(["PassengerId", "Name", "Survived", "Ticket"], axis= 1)
y = data["Survived"]
print(y)

# SPLITTING DATA INTO TEST DATA AND TRAIN DATA
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# LOGISTIC REGRESSION
model = LogisticRegression()
model.fit(x_train, y_train)

# ACCURACY SCORE
# Accuracy on the training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)
print("Accuracy on the training data is:", training_data_accuracy)

# Accuracy on the test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy on the training data is:", test_data_accuracy)

# PREDICTIVE SYSTEM ON GIVING INPUT
input_data = (1, 0, 46, 0, 1, 7.28, 1)
# change the input array into numpy array
input_as_numpy = np.asarray(input_data)

# VISUALIZING INPUT DATA
input_df = pd.DataFrame({"Feature": x.columns, "Value": input_data})
plt.figure(figsize=(8, 5))
sns.barplot(data=input_df, x="Feature", y="Value")
plt.title("Input Features")
plt.xticks(rotation=45)
plt.show()

# Reshape the numpy array as we are predicting for only one instance
input_reshaped = input_as_numpy.reshape(1, -1)
prediction = model.predict(input_reshaped)
print(prediction)

if prediction[0] == 0:
    print("THE PERSON WON'T BE SAVED FROM SINKING.")
else:
    print("THE PERSON WILL BE SAVED FROM SINKING.")






