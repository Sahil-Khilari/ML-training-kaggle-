# ==============================
# 1. LOAD DATASET (DIGITS)
# ==============================
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("Loading Digit dataset...\n")

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==============================
# 2. BASELINE SVM (LINEAR)
# ==============================
print("Training Linear SVM...\n")

linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)

pred_linear = linear_svm.predict(X_test)
acc_linear = accuracy_score(y_test, pred_linear)

print("Linear SVM Accuracy:", acc_linear)


# ==============================
# 3. INSPECT SUPPORT VECTORS
# ==============================
print("\nSupport Vector Analysis")
print("Number of Support Vectors:", len(linear_svm.support_vectors_))
print("Support Vectors per class:", linear_svm.n_support_)


# ==============================
# 4. KERNEL SVM (RBF)
# ==============================
print("\nTraining RBF Kernel SVM...\n")

rbf_svm = SVC(kernel='rbf', gamma='scale')
rbf_svm.fit(X_train, y_train)

pred_rbf = rbf_svm.predict(X_test)
acc_rbf = accuracy_score(y_test, pred_rbf)

print("RBF SVM Accuracy:", acc_rbf)


# ==============================
# 5. ONE VS REST STRATEGY
# ==============================
print("\nTraining One-vs-Rest SVM...\n")

ovr_model = OneVsRestClassifier(SVC(kernel='linear'))
ovr_model.fit(X_train, y_train)

pred_ovr = ovr_model.predict(X_test)
acc_ovr = accuracy_score(y_test, pred_ovr)

print("OvR SVM Accuracy:", acc_ovr)


# ==============================
# 6. PCA + SVM EXPERIMENT
# ==============================
print("\nApplying PCA before SVM...\n")

pca = PCA(n_components=30)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca_svm = SVC(kernel='rbf')
pca_svm.fit(X_train_pca, y_train)

pred_pca = pca_svm.predict(X_test_pca)
acc_pca = accuracy_score(y_test, pred_pca)

print("PCA + SVM Accuracy:", acc_pca)


# ==============================
# 7. COMPARE RESULTS
# ==============================
print("\nModel Comparison")

results = pd.DataFrame({
    "Model": [
        "Linear SVM",
        "RBF SVM",
        "One-vs-Rest SVM",
        "PCA + RBF SVM"
    ],
    "Accuracy": [
        acc_linear,
        acc_rbf,
        acc_ovr,
        acc_pca
    ]
})

print(results)


# ==============================
# 8. SVR (REGRESSION)
# ==============================
print("\nLoading Housing Dataset for SVR...\n")

housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Support Vector Regression...\n")

svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

pred = svr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)

print("SVR Results")
print("RMSE:", rmse)
print("MAE:", mae)