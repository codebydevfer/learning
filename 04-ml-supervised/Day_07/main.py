# Key Concepts of Support Vector Machine


# Hyperplane: A decision boundary separating different classes in feature space and is represented by the equation wx + b = 0 in linear classification.
# Support Vectors: The closest data points to the hyperplane, crucial for determining the hyperplane and margin in SVM.
# Margin: The distance between the hyperplane and the support vectors. SVM aims to maximize this margin for better classification performance.
# Kernel: A function that maps data to a higher-dimensional space enabling SVM to handle non-linearly separable data.
# Hard Margin: A maximum-margin hyperplane that perfectly separates the data without misclassifications.
# Soft Margin: Allows some misclassifications by introducing slack variables, balancing margin maximization and misclassification penalties when data is not perfectly separable.
# C: A regularization term balancing margin maximization and misclassification penalties. A higher C value forces stricter penalty for misclassifications.
# Hinge Loss: A loss function penalizing misclassified points or margin violations and is combined with regularization in SVM.
# Dual Problem: Involves solving for Lagrange multipliers associated with support vectors, facilitating the kernel trick and efficient computation.


#Kernel Trick

# The kernel trick is a method used in SVMs to enable them to classify non-linear data using a linear classifier. By applying a kernel function, SVMs can implicitly map input data into a higher-dimensional space where a linear separator (hyperplane) can be used to divide the classes. This mapping is computationally efficient because it avoids the direct calculation of the coordinates in this higher space.

#1

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles # Example for non-linear data
import matplotlib.pyplot as plt
import numpy as np
    
X, y = make_circles(n_samples=100, factor=0.1, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Using the RBF kernel for non-linear separation
model = svm.SVC(kernel='rbf', gamma='auto') # 'auto' for gamma uses 1/n_features
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Plotting the decision boundary (for 2D data)
def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none', edgecolors='k')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
plot_svc_decision_function(model)
plt.title('SVM with RBF Kernel Decision Boundary')
plt.show()

#Margin

#Hard Margin

# In a hard margin SVM, the objective is to identify a hyperplane that completely separates data points belonging to different classes, ensuring a clear demarcation with the utmost margin width possible. This margin is the distance between the hyperplane and the nearest data point, also known as the support vectors.

#1

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Generate some linearly separable data
X = np.array([
    [1, 2], [2, 3], [3, 3], [1, 1],
    [5, 6], [6, 7], [7, 7], [5, 5]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Create an SVC classifier with a large C value for hard margin approximation
# The 'linear' kernel is used for linear separability
clf = SVC(kernel='linear', C=1e10) 

# Fit the model to the data
clf.fit(X, y)

# Plotting the data and the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')

# Plot the decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# Plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.title('Hard Margin SVM Approximation with scikit-learn')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()



#Soft Margin

# Soft Margin SVM introduces flexibility by allowing some margin violations (misclassifications) to handle cases where the data is not perfectly separable. Suitable for scenarios where the data may contain noise or outliers. It Introduces a penalty term for misclassifications, allowing for a trade-off between a wider margin and a few misclassifications.

#1

from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate some sample data
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Soft Margin SVM classifier
# Adjust the 'C' parameter to control the soft margin behavior
# A larger C means less tolerance for misclassification (harder margin)
# A smaller C means more tolerance for misclassification (softer margin)
svm_model = SVC(kernel='linear', C=1.0) # Using a linear kernel for simplicity

# Train the model
svm_model.fit(X_train, y_train)

# Evaluate the model
accuracy = svm_model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")









#Reference - https://www.geeksforgeeks.org/machine-learning/using-a-hard-margin-vs-soft-margin-in-svm/
#Reference - geeksforgeeks