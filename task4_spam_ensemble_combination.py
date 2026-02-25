#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--kfold", type=int, default=5)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

def load_sms(path):
    try:
        df = pd.read_csv(path, sep='\t', header=None, names=['label','message'])
    except:
        df = pd.read_csv(path)
    if 'message' not in df.columns:
        df = df.iloc[:, :2]
        df.columns = ['label','message']
    return df

df = load_sms(args.data)
df = df.dropna(subset=['message'])

le = LabelEncoder()
y = le.fit_transform(df[args.target])

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['message'])

nb = MultinomialNB()
lr = LogisticRegression(max_iter=1000, solver='liblinear')
svm = CalibratedClassifierCV(LinearSVC(max_iter=10000), cv=3)

hard_voting = VotingClassifier(estimators=[('nb', clone(nb)), ('lr', clone(lr)), ('svm', clone(svm))], voting='hard')
soft_voting = VotingClassifier(estimators=[('nb', clone(nb)), ('lr', clone(lr)), ('svm', clone(svm))], voting='soft')
stacking = StackingClassifier(estimators=[('nb', clone(nb)), ('lr', clone(lr)), ('svm', clone(svm))], final_estimator=LogisticRegression(max_iter=1000), cv=args.kfold)
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), random_state=args.random_state)

models = {
    'MultinomialNB': (nb, {'alpha':[0.5,1.0]}),
    'LogisticRegression': (lr, {'C':[0.5,1,2]}),
    'LinearSVC-Calibrated': (svm, {}),
    'HardVoting': (hard_voting, {}),
    'SoftVoting': (soft_voting, {}),
    'Stacking': (stacking, {}),
    'AdaBoost_Stumps': (ada, {'n_estimators':[30,50]})
}

skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.random_state)

results = []
roc_plot_data = {}

for name,(model,grid) in models.items():
    if grid:
        gs = GridSearchCV(model, grid, cv=3, scoring='f1')
        gs.fit(X,y)
        model = gs.best_estimator_

    precision_scores, recall_scores, f1_scores, roc_scores = [],[],[],[]
    y_true_all, y_pred_all, y_prob_all = [],[],[]

    for train_idx, test_idx in skf.split(X,y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m = clone(model)
        m.fit(X_train,y_train)
        y_pred = m.predict(X_test)
        try:
            y_prob = m.predict_proba(X_test)[:,1]
        except:
            if hasattr(m,"decision_function"):
                scaler = MinMaxScaler()
                y_prob = scaler.fit_transform(m.decision_function(X_test).reshape(-1,1)).ravel()
            else:
                y_prob = np.zeros_like(y_pred,dtype=float)

        precision_scores.append(precision_score(y_test,y_pred,zero_division=0))
        recall_scores.append(recall_score(y_test,y_pred,zero_division=0))
        f1_scores.append(f1_score(y_test,y_pred,zero_division=0))
        roc_scores.append(roc_auc_score(y_test,y_prob))

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)
        y_prob_all.append(y_prob)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    y_prob_all = np.concatenate(y_prob_all)

    tn,fp,fn,tp = confusion_matrix(y_true_all,y_pred_all).ravel()

    fpr,tpr,_ = roc_curve(y_true_all,y_prob_all)
    roc_plot_data[name]=(fpr,tpr,auc(fpr,tpr))

    results.append({
        'model':name,
        'precision_mean':np.mean(precision_scores),
        'precision_std':np.std(precision_scores),
        'recall_mean':np.mean(recall_scores),
        'recall_std':np.std(recall_scores),
        'f1_mean':np.mean(f1_scores),
        'f1_std':np.std(f1_scores),
        'roc_mean':np.mean(roc_scores),
        'roc_std':np.std(roc_scores),
        'tn':tn,'fp':fp,'fn':fn,'tp':tp
    })

pd.DataFrame(results).to_csv('ensemble_comparison.csv',index=False)

plt.figure()
for name,(fpr,tpr,roc_auc) in roc_plot_data.items():
    plt.plot(fpr,tpr,label=f'{name} (AUC={roc_auc:.3f})')
plt.plot([0,1],[0,1],'--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Comparison')
plt.legend()
plt.savefig('roc_comparison.png')

best_model_name = sorted(results,key=lambda x:x['f1_mean'],reverse=True)[0]['model']
best_model = models[best_model_name][0]
best_model.fit(X,y)

try:
    probs = best_model.predict_proba(X)[:,1]
except:
    scaler = MinMaxScaler()
    probs = scaler.fit_transform(best_model.decision_function(X).reshape(-1,1)).ravel()

preds = best_model.predict(X)

pd.DataFrame({
    'MessageId':np.arange(len(df)),
    'Actual':y,
    'Predicted':preds,
    'Probability':probs
}).to_csv('final_model_predictions.csv',index=False)

print("Best model:",best_model_name)
print("Saved: ensemble_comparison.csv, final_model_predictions.csv, roc_comparison.png")