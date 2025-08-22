#Metrics

#Accuracy

#Accuracy = (Number of Correct Predictions) / (Total Number of Predictions)
#Accuracy = (TP + TN) / (TP + TN + FP + FN)

#1
from sklearn.metrics import accuracy_score

# y_true: Ground truth (correct) labels
# y_pred: Predicted labels, as returned by a classifier

y_true = [0, 1, 0, 1, 0]
y_pred = [0, 1, 1, 1, 0]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")


#Confusion Matrix

#Accuracy = (TP + TN) / (TP + TN + FP + FN)
#1
# from sklearn.metrics import confusion_matrix

# confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)

#2
from sklearn.metrics import confusion_matrix
import numpy as np

y_true = np.array([0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 0, 0, 1, 1, 1])

cm = confusion_matrix(y_true, y_pred)
print(cm)

#Precision

#Precision = True Positives / (True Positives + False Positives)

#1
from sklearn.metrics import precision_score

# Assuming y_true contains the true labels and y_pred contains the predicted labels
# For binary classification, ensure the positive class is designated (e.g., 1)
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

#Recall

#Recall = TP / (TP + FN)
# TP
# (True Positives) represents the number of actual positive cases that were correctly predicted as positive by the model.
# FN
# (False Negatives) represents the number of actual positive cases that were incorrectly predicted as negative by the model.

#1
from sklearn.metrics import recall_score

# Example: Actual labels (true values)
y_true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]

# Example: Predicted labels from a classification model
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

# Calculate recall
recall = recall_score(y_true, y_pred)

print(f"Recall: {recall}")

#F1-score

#F1 = 2 * (Precision * Recall) / (Precision + Recall)
#    Precision = True Positives / (True Positives + False Positives)
#    Recall = True Positives / (True Positives + False Negatives)

#1
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

# Sample data
y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1}")

# You can also calculate precision and recall separately
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Example with a simple model
X = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20]])
y = np.array([0,1,0,1,0,1,0,0,1,1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
y_pred_model = model.predict(X_test)

f1_model = f1_score(y_test, y_pred_model)
print(f"F1 Score for the model: {f1_model}")





#Reference - Google Search
#Reference - https://www.v7labs.com/blog/f1-score-guide