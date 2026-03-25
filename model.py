import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import SVC

df = pd.read_csv(r"C:\Users\sahar\OneDrive\Desktop\ds and ml files\Pharma_Industry.csv")

print(df.info())
print(df.describe())
print("Shape:", df.shape)

X = df.drop(columns="Drug Response")
y = df["Drug Response"]

print("Feature shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm_linear = SVC(kernel='linear', C=1.0, probability=True)
svm_linear.fit(X_train, y_train)

print("Linear Accuracy:", svm_linear.score(X_test, y_test))

y_pred_linear = svm_linear.predict(X_test)

print("Precision:", precision_score(y_test, y_pred_linear))
print("Recall:", recall_score(y_test, y_pred_linear))
print("F1 Score:", f1_score(y_test, y_pred_linear))

cm = confusion_matrix(y_test, y_pred_linear)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Linear)")
plt.show()

y_pred_proba = svm_linear.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve (Linear)')
plt.legend()
plt.show()

svm_rbf = SVC(kernel="rbf", gamma=0.1, C=1.0, probability=True)
svm_rbf.fit(X_train, y_train)

print("RBF Accuracy:", svm_rbf.score(X_test, y_test))

y_pred_rbf = svm_rbf.predict(X_test)

print("RBF Precision:", precision_score(y_test, y_pred_rbf))
print("RBF Recall:", recall_score(y_test, y_pred_rbf))
print("RBF F1:", f1_score(y_test, y_pred_rbf))

y_pred_proba_rbf = svm_rbf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rbf)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve (RBF)')
plt.legend()
plt.show()

scores = cross_val_score(svm_rbf, X, y, cv=5)
print("CV Mean:", scores.mean())

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Stratified CV:", cross_val_score(svm_rbf, X, y, cv=skf).mean())

kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("K-Fold CV:", cross_val_score(svm_rbf, X, y, cv=kf).mean())

ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
print("Shuffle Split CV:", cross_val_score(svm_rbf, X, y, cv=ss).mean())

kernels = ['linear', 'rbf']
C_values = [0.1, 1, 10,100]
gamma_values = [0.001, 0.01, 0.1, 1]

results = []

for kernel in kernels:
    for C in C_values:
        for gamma in gamma_values:

            if kernel == "linear":
                model = SVC(kernel=kernel, C=C)
            else:
                model = SVC(kernel=kernel, C=C, gamma=gamma)

            scores = cross_val_score(model, X, y, cv=skf)

            results.append({
                "Kernel": kernel,
                "C": C,
                "Gamma": gamma,
                "Mean Score": scores.mean()
            })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="Mean Score", ascending=False)

print(df_results)



joblib.dump(svm_rbf, "svm_rbf_best_model.joblib")
print("Model saved successfully!")