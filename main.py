import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

#Model Selection and Hyper-parameters Tunning
#part one

# Step 1: Read data from the CSV file
data = pd.read_csv('data_reg.csv')
#since pandas have a really good data frames to manipulate data
#we can use it to read the data from the csv file especially if the data is in a table format

# Step 2: Split the data into training, validation, and testing sets
training_set = data[:120]#0-120 is the training set
validation_set = data[120:160] #120-160 is the validation set
testing_set = data[160:] #160-200 is the testing set

# Step 3: Create a 3D scatter plot
fig = plt.figure(figsize=(8,8))
figg = fig.add_subplot(111, projection='3d')
#111 means 1 row, 1 column, and the 1st plot it means that we only have 1 plot and in 3D mode

# Scatter plot for training set
figg.scatter(training_set['x1'], training_set['x2'], training_set['y'], c='r', marker='o', label='Training Set')

# Scatter plot for validation set
figg.scatter(validation_set['x1'], validation_set['x2'], validation_set['y'], c='g', marker='o', label='Validation Set')

# Scatter plot for testing set
figg.scatter(testing_set['x1'], testing_set['x2'], testing_set['y'], c='b', marker='^', label='Testing Set')

# Set labels for axes
figg.set_xlabel('X1')
figg.set_ylabel('X2')
figg.set_zlabel('Y')

# Set legend
figg.legend()#legend is the label of the plot
# Set title
figg.set_title('3D Scatter Plot of data_reg.csv')

# Show the plot
plt.show()
# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************

#part two

# Data Preparation
#training_set = data[:120]
#validation_set = data[120:160]
#they have been saved in features and target variables
features_train = training_set[['x1', 'x2']]
target_train = training_set['y']
features_validation = validation_set[['x1', 'x2']]
target_validation = validation_set['y']

# List to store validation errors for each degree
#i will use it to store the validation errors for each degree
validation_errors = []

def generate_meshgrid(features_train, num_points=100):
    x1_range = np.linspace(features_train['x1'].min(), features_train['x1'].max(), num_points)
    x2_range = np.linspace(features_train['x2'].min(), features_train['x2'].max(), num_points)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    return x1_mesh, x2_mesh
#meshgrid is a function that will create a grid of points
#we will use it to create a grid of points for the x1 and x2 features
#we will use the grid of points to plot the surface of the polynomial function


def transform_and_predict(regression_model, poly_features, x1_mesh, x2_mesh):
    new_data = poly_features.transform(np.c_[x1_mesh.ravel(), x2_mesh.ravel()])
    predicted_y = regression_model.predict(new_data).reshape(x2_mesh.shape)
    return predicted_y
#transform is a function that will transform the original features to polynomial features
#we will use it to transform the original features to polynomial features with the degree we want
#we will use the polynomial features to train the model and predict the target values and to plot the surface of the polynomial function


def plot_surface_and_points(ax, x1_mesh, x2_mesh, predicted_y, training_set, validation_set, degree):
    ax.scatter(training_set['x1'], training_set['x2'], training_set['y'], c='r', marker='o', label='Training Set')
    ax.scatter(validation_set['x1'], validation_set['x2'], validation_set['y'], c='g', marker='o', label='Validation Set')
    ax.plot_surface(x1_mesh, x2_mesh, predicted_y, alpha=0.5, cmap='viridis', label=f'Ploynomial Degree {degree} Surface')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.legend()
    ax.set_title(f'Degree {degree}')
#plot_surface is a function that will plot the surface of the polynomial function
#we will use it to plot the surface of the polynomial function of the specified degree
#we will use the polynomial function to transform the original features to polynomial features with the degree we want


def plot_surface_with_degree(regression_model, poly_features, training_set, validation_set, degree):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    x1_mesh, x2_mesh = generate_meshgrid(training_set)

    predicted_y = transform_and_predict(regression_model, poly_features, x1_mesh, x2_mesh)

    plot_surface_and_points(ax, x1_mesh, x2_mesh, predicted_y, training_set, validation_set, degree)

    #plt.show()


for degree in range(1, 11):
    # Create polynomial features function
    poly_features = PolynomialFeatures(degree)
    #degree is the degree of the polynomial function
    #we will use the polynomial function to transform the original features to polynomial features with the degree we want

    # Transform training and validation features to fit in the polynomial function of the specified degree
    X_train_poly = poly_features.fit_transform(features_train)
    X_validation_poly = poly_features.transform(features_validation)

    # Initialize linear regression model
    regression_model = LinearRegression(fit_intercept=True)
    #fit_intercept=True means that we will have a bias


    # Train the model
    regression_model.fit(X_train_poly, target_train)
    #fit is a function that will train the model regarding input data and target data

    # Predict on the validation set
    YpredictedValidation = regression_model.predict(X_validation_poly)

    # Calculate mean squared error
    validation_error = mean_squared_error(target_validation, YpredictedValidation)
    # mean_squared_error is a function that will calculate the mean squared error between predicted and actual values
    # target_validation is the actual values and y_val_pred is the predicted values
    # we will store the validation error in the validation_errors list
    validation_errors.append(validation_error)
    plot_surface_with_degree(regression_model, poly_features, training_set, validation_set, degree)

# Print validation errors for each degree
for degree, error in zip(range(1, 11), validation_errors):
    print(f'Degree {degree}: Validation Error = {error}')

# Determine the degree with the least error
best_degree = np.argmin(validation_errors) + 1  # Add 1 to convert index to degree
best_error = min(validation_errors)
print(f'The best polynomial degree is: {best_degree} with a validation error of {best_error}')

# Plot validation error vs polynomial degree
plt.figure(figsize=(6, 6))
plt.plot(range(1, 11), validation_errors, marker='o', color='green')
plt.xlabel('Polynomials Degree')
plt.ylabel('Validations Error')
plt.title('Validation Error vs. Polynomial Degree')
plt.show()


# **********************************************************************************************************************
# **********************************************************************************************************************
# **********************************************************************************************************************
#part three
# Step 3: Apply Ridge regression on the training set with polynomial degree 8
degree = 8
poly = PolynomialFeatures(degree=degree)

X_poly_train = poly.fit_transform(training_set[['x1', 'x2']])
X_poly_val = poly.transform(validation_set[['x1', 'x2']])

# Regularization parameters to try
alpha_values = [0.001, 0.005, 0.01, 0.1, 10]
mse_values = []

for alpha in alpha_values:
    # Fit Ridge regression model
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_poly_train, training_set['y'])

    # Predict on validation set
    y_val_pred = ridge_model.predict(X_poly_val)

    # Calculate Mean Squared Error
    mse = mean_squared_error(validation_set['y'], y_val_pred)
    mse_values.append(mse)
#mean_squared_error is a function that will calculate the mean squared error between predicted and actual values
#print the mse_values

# Print MSE for each regularization parameter
print('MSE for each regularization parameter:')
for alpha, mse in zip(alpha_values, mse_values):
    print(f'alpha={alpha}: MSE={mse}')

# Determine the regularization parameter with the least error
best_alpha = alpha_values[np.argmin(mse_values)]
best_mse = min(mse_values)
print(f'The best regularization parameter is: {best_alpha} with a MSE of {best_mse}')

# Plot MSE on validation vs regularization parameter
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, mse_values, marker='o', color='red')
plt.xscale('log')  # Use log scale for better visualization
plt.xlabel('Regularization Parameter (alpha)', fontsize=14, fontweight='bold', color='blue')
plt.ylabel('Mean Squared Error (MSE) on Validation Set', fontsize=14, fontweight='bold', color='blue')
plt.title('MSE on Validation vs Regularization Parameter for Ridge Regression')
plt.grid(True)
plt.show()
########################################################################################################################
########################################################################################################################
########################################################################################################################
#logistic regression
#part1
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Step 1: Read data from the CSV files
train_data = pd.read_csv('train_cls.csv')
test_data = pd.read_csv('test_cls.csv')

# Step 2: Prepare the data
X_train = train_data[['x1', 'x2']]
y_train = train_data['class']


X_test = test_data[['x1', 'x2']]
y_test = test_data['class']

# Step 3: Learn a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Plot the decision boundary
x_min, x_max = X_train['x1'].min() - 1, X_train['x1'].max() + 1
y_min, y_max = X_train['x2'].min() - 1, X_train['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Check if predictions contain non-numeric values
if not np.issubdtype(Z.dtype, np.number):
    # If not, convert them to numeric values (e.g., using label encoding)
    unique_labels = np.unique(Z)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    Z = np.array([label_mapping[label] for label in Z], dtype=float)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
# Contour plot with a custom color for decision boundary
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
plt.colorbar()
# Scatter plot for training set
plt.scatter(X_train['x1'], X_train['x2'], c=y_train, edgecolors='g', cmap=plt.cm.Paired)
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# Step 5: Compute training accuracy
y_train_pred = model.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Training Accuracy: {training_accuracy}')


# Step 6: Compute testing accuracy
y_test_pred = model.predict(X_test)
testing_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Testing Accuracy: {testing_accuracy}')
########################################################################################################################

#part2

from sklearn.preprocessing import PolynomialFeatures, LabelEncoder


# Step 1: Read data from the CSV files
train_data = pd.read_csv('train_cls.csv')
test_data = pd.read_csv('test_cls.csv')

# Step 2: Prepare the data and convert class labels to numerical values
X_train = train_data[['x1', 'x2']]
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_data['class'])

X_test = test_data[['x1', 'x2']]
y_test = label_encoder.transform(test_data['class'])

# Step 3: Create quadratic features
poly = PolynomialFeatures(degree=2)
X_train_quadratic = poly.fit_transform(X_train)
X_test_quadratic = poly.transform(X_test)

# Step 4: Learn a logistic regression model with quadratic features
model_quadratic = LogisticRegression()
model_quadratic.fit(X_train_quadratic, y_train)

# Step 5: Draw the decision boundary on a scatterplot of the training set
plt.figure(figsize=(8, 8))
# Scatter plot for class 0 ( class 0 is represented by the color green)
plt.scatter(X_train[y_train == 0]['x1'], X_train[y_train == 0]['x2'], c='lightgreen', edgecolors='k', marker='o', label='Class 0')

# Scatter plot for class 1 ( class 1 is represented by the color lightblue)
plt.scatter(X_train[y_train == 1]['x1'], X_train[y_train == 1]['x2'], c='lightblue', edgecolors='k', marker='o', label='Class 1')

plt.title('Scatterplot of Training Set with Quadratic Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')

# Plot the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X_train['x1'].min() - 1, X_train['x1'].max() + 1
y_min, y_max = X_train['x2'].min() - 1, X_train['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model_quadratic.predict(poly.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# Explicitly convert Z to a NumPy array and handle invalid values
Z = np.array(Z, dtype=float)
Z[Z == 0] = np.nan

# Plot the decision boundary using contourf
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
plt.colorbar()
plt.show()

# Step 6: Compute training accuracy
y_train_pred_quadratic = model_quadratic.predict(X_train_quadratic)
training_accuracy_quadratic = accuracy_score(y_train, y_train_pred_quadratic)
print(f'Training Accuracy (Quadratic): {training_accuracy_quadratic:.2%}')

# Step 7: Compute testing accuracy
y_test_pred_quadratic = model_quadratic.predict(X_test_quadratic)
testing_accuracy_quadratic = accuracy_score(y_test, y_test_pred_quadratic)
print(f'Testing Accuracy (Quadratic): {testing_accuracy_quadratic:.2%}')
