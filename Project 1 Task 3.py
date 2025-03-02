# Task 3: Classification Using Real Data

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = load_breast_cancer()
X = data.data
y = data.target

# Generate 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# We will make models with several kernels:
# RBF (Gaussian); Linear; and Polynomial of Degrees 2, 3, 4, 5
# We let RBF have "degree" 0
kernel_degrees = [0, 1, 2, 3, 4, 5]
# Train a SVM model for each kernel type
for kernel_degree in kernel_degrees:
    if kernel_degree == 0:
        kernel_type = 'rbf'
        print("Gaussian Kernel:")
    elif kernel_degree == 1:
        kernel_type = 'linear'
        print("Linear Kernel:")
    else:
        kernel_type = 'poly'
        print(f"Degree {kernel_degree} Polynomial Kernel:")
    # Create the kernel SVM model
    # Degree is ignored for RBF and linear kernels
    kernel_SVM = make_pipeline(StandardScaler(), SVC(kernel = kernel_type, degree = kernel_degree))
    # Fit the model to the data
    kernel_SVM.fit(X_train, y_train)
    # Calculate predicted labels
    y_predict = kernel_SVM.predict(X_test)

    # Assemble test results
    test_results = list(zip(y_predict, y_test))
    # Count numbers of correct/wrong test labels
    num_correct = 0
    num_wrong = 0
    for predict, test in test_results:
        if predict == test:
            num_correct += 1
        else:
            num_wrong += 1
    # Calculate the % rate of correct labels
    percent_correct = num_correct/len(test_results)
    # Print out results
    print(f"    Labels Predicted Correctly: {num_correct}/{len(test_results)}")
    print(f"    Percent Labeled Correctly = {percent_correct: .1%}")
