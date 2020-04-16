import tqdm
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn as sk
from sklearn import ensemble, model_selection, metrics, tree
import matplotlib.pyplot as plt

SEED = 10

wine_white = pd.read_csv("/Users/fbergh/Documents/Radboud/master/1/NatCom/NatCom-code/assignment5/data/winequality-white.csv",sep=";")
y = np.array(wine_white["quality"].tolist())
X = wine_white.drop("quality", axis=1)

plt.hist(y, bins=len(np.unique(y)))
plt.show()

base = tree.DecisionTreeClassifier(max_depth=1, random_state=SEED)
ada = ensemble.AdaBoostClassifier(base_estimator=base, n_estimators=100, learning_rate=1, random_state=SEED)
skf = model_selection.StratifiedKFold(n_splits=100, shuffle=True, random_state=SEED)
preds, ground_truths = [], []

for train_idx, test_idx in tqdm.tqdm(skf.split(X, y), total=skf.get_n_splits(), ncols=80):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ada.fit(X_train, y_train)
    preds.extend(ada.predict(X_test))
    ground_truths.extend(y_test)

preds = np.array(preds)
ground_truths = np.array(ground_truths)

print(f"Accuracy = {np.sum(preds==ground_truths)/float(len(preds))}")
print(f'{"weighted"}')
print(f'F1 score = {metrics.f1_score(ground_truths, preds, average="weighted")}')
print(f"Feature importance = {ada.feature_importances_}")

conf_mat = metrics.confusion_matrix(ground_truths, preds)
df_cm = pd.DataFrame(conf_mat, index=range(3,10), columns=range(3,10))
sn.heatmap(df_cm, annot=True)
plt.show()
