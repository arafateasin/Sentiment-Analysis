# Import necessary libraries
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Data Preprocessing
# Load the dataset
df = pd.read_csv('Reviews.csv')

# Drop rows with missing 'Text'
df = df.dropna(subset=['Text'])

# Perform necessary preprocessing steps
# This might include removing null values, converting text to lower case, removing punctuation, tokenization, stemming or lemmatization, etc.

# 2. Feature Extraction
# Convert text data into numerical features using Bag-of-Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Text'])
y = df['Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Selection
# Experiment with different methods for sentiment classification

# Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
nb_train_preds = nb_classifier.predict(X_train)
nb_test_preds = nb_classifier.predict(X_test)

# Logistic Regression
lr_classifier = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=10000))
lr_classifier.fit(X_train, y_train)
lr_train_preds = lr_classifier.predict(X_train)
lr_test_preds = lr_classifier.predict(X_test)

# Support Vector Machine
svm_classifier = make_pipeline(StandardScaler(with_mean=False), LinearSVC(dual=False, max_iter=10000))
svm_classifier.fit(X_train, y_train)
svm_train_preds = svm_classifier.predict(X_train)
svm_test_preds = svm_classifier.predict(X_test)


# Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_train_preds = rf_classifier.predict(X_train)
rf_test_preds = rf_classifier.predict(X_test)

# Decision Tree
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_train_preds = dt_classifier.predict(X_train)
dt_test_preds = dt_classifier.predict(X_test)

# 4. Model Evaluation
# Evaluate the performance of each model
# This might include calculating accuracy, precision, recall, F1 score, ROC AUC score, etc. for both the training and testing sets.

# 5. Discussion
# Discuss the strengths and weaknesses of the selected models for sentiment classification
# This might include discussing overfitting or underfitting, interpretability, efficiency, scalability, etc. of each model.

