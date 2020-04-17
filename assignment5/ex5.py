import os
import tqdm
import numpy as np
import pandas as pd
import seaborn as sn
import sklearn as sk
from sklearn import ensemble, model_selection, metrics, tree
import matplotlib.pyplot as plt

if not os.path.exists("./img/"):
    os.makedirs("./img/")

SEED = 10

wine_white = pd.read_csv("/Users/fbergh/Documents/Radboud/master/1/NatCom/NatCom-code/assignment5/data/winequality-white.csv",sep=";")
y = np.array(wine_white["quality"].tolist())
X = wine_white.drop("quality", axis=1)

fig, ax = plt.subplots(1, 1)
ax.grid(zorder=0)
ax.hist(y, bins=np.arange(3,11)-0.5, rwidth=0.9, zorder=2)
ax.set_xlabel("Class")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of class distribution")
plt.savefig(f"img/ex5-class-distribution.png", bbox_inches='tight')

max_max_depth = 10
max_depths = range(1,max_max_depth+1)
learning_rates = np.arange(0.2, 1.1, 0.2)
accuracies, f_scores, features = {}, {}, {}

for lr in learning_rates:
    accs, fs = [], []
    for md in max_depths:
        print(f'Running AdaBoost with learning_rate={lr} and max_depth={md}')
        base = tree.DecisionTreeClassifier(max_depth=md, random_state=SEED)
        ada = ensemble.AdaBoostClassifier(base_estimator=base, n_estimators=100, learning_rate=lr, random_state=SEED)
        skf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        preds, ground_truths = [], []

        for train_idx, test_idx in tqdm.tqdm(skf.split(X, y), total=skf.get_n_splits(), ncols=80):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            ada.fit(X_train, y_train)
            preds.extend(ada.predict(X_test))
            ground_truths.extend(y_test)

        preds = np.array(preds)
        ground_truths = np.array(ground_truths)

        accs.append(np.sum(preds==ground_truths)/float(len(preds)))
        fs.append(metrics.f1_score(ground_truths, preds, average="weighted"))
        # Round learning rate to one decimal due to float precision
        features[np.round(lr,1),md] = ada.feature_importances_

        # conf_mat = metrics.confusion_matrix(ground_truths, preds)
        # df_cm = pd.DataFrame(conf_mat, index=range(3,10), columns=range(3,10))
        # fig, ax = plt.subplots(1, 1)
        # sn.heatmap(df_cm, annot=True)
        # ax.set_xlabel("Predicted label")
        # ax.set_ylabel("True label")
        # ax.set_title(f"Confusion matrix for AdaBoost with lr={np.round(lr,1)} and max_depth={md}")
        # plt.savefig(f"img/ex5-confusion-matrix-lr={np.round(lr,1)}-max_depth={md}.png", bbox_inches='tight')

    accuracies[lr] = accs
    f_scores[lr] = fs

def plot_metric(metric):
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0)
    for lr in learning_rates:
        if metric == "Accuracy":
            ax.plot(accuracies[lr], label=f"LR = {np.round(lr,1)}", zorder=2)
        else:
            ax.plot(f_scores[lr], label=f"LR = {np.round(lr,1)}", zorder=2)
    ax.legend(loc=4, framealpha=1)
    ax.set_xticks(range(max_max_depth))
    ax.set_xticklabels(max_depths)
    ax.set_xlabel("max_depth")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} for different max_depth values")
    plt.savefig(f'img/ex5-{metric.lower()}.png', bbox_inches='tight')

plot_metric("Accuracy")
plot_metric("F1-score")

features_d10 = features[0.6,10]
features_d1  = features[0.2,1]

scale = 3
displace = 0.5
fig, ax = plt.subplots(1, 1, figsize=(10,5))
ax.grid(zorder=0)
ax.bar(np.arange(len(features_d1))*scale-displace, features_d1, label="max_depth=1, LR=0.2", zorder=2)
ax.bar(np.arange(len(features_d10))*scale+displace, features_d10, label="max_depth=10, LR=0.6", zorder=2)
ax.set_xticks(np.arange(len(features_d10))*scale)
ax.set_xticklabels([colname.replace(" ", "\n") for colname in X.columns])
ax.set_xlabel("Features")
ax.set_ylabel("Feature importance")
ax.set_title("Feature importance for two different AdaBoost configurations")
ax.legend(framealpha=1)
plt.savefig(f'img/ex5-feature-importance.png', bbox_inches='tight')