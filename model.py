import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df_train = pd.read_csv('train.csv') # converting the training csv into a pandas dataframe object
#print(df_train)

df_train.groupby("target").size() # first look at response variable and labels


####################### SPLITTING TRAINING INTO TRAINING AND VALIDATION ############################

predictor_variables = [var for var in df_train.columns if var not in ['target', 'ID_code']]

y = df_train.loc[:, 'target'] # the label (response) variable only with record
x = df_train.loc[:, predictor_variables] # the predictor variables with record

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)

initial_tree = DecisionTreeClassifier(max_leaf_nodes=8, class_weight='balanced')
initial_tree.fit(x_train, y_train)

def print_tree(fitted_tree, predictors: list):
    """
    Takes a fitted tree and a list of predictor variables as input and visualises the tree.
    :param fitted_tree:
    :return:
    """

    # Create the figure
    plt.figure(figsize=(20, 10))

    # Create the tree plot
    plot_tree(fitted_tree,
              feature_names=predictors,  # Feature names
              class_names=["0", "1"],  # Class names
              rounded=True,
              filled=True)

    plt.show()

# print_tree(initial_tree, predictor_variables)

y_train_pred = initial_tree.predict(x_train) # the model predicting the output based on the data
                                             # it was built upon

y_train_valid = initial_tree.predict(x_valid) # the model preditcting the output for the validation
                                              # dataset
# calculating the auc score for both the training and validation sets

auc_valid = metrics.roc_auc_score(y_valid, y_train_valid)

auc_train = metrics.roc_auc_score(y_train, y_train_pred)

#print({'Auc score for validiation set' : auc_valid,'Auc score for training set': auc_train})


def tree_training(max_leaf_nodes, X_train, y_train, X_valid, y_valid):
    model_tree = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, class_weight='balanced')
    model_tree.fit(X_train, y_train)

    y_train_pred = model_tree.predict(X_train)
    y_valid_pred = model_tree.predict(X_valid)

    auc_train = metrics.roc_auc_score(y_train, y_train_pred)
    auc_valid = metrics.roc_auc_score(y_valid, y_valid_pred)

    print("Nodes:{}, Train:{:.4f}, Valid:{:.4f}, Diff:{:.4f}".format(max_leaf_nodes,
                                                                     auc_train,
                                                                     auc_valid,
                                                                     auc_train - auc_valid))


# Run few iterations to find which max_tree_nodes works best
for i in range(2, 20):
    tree_training(i, x_train, y_train, x_valid, y_valid)

zeros_probs = [0 for _ in range(len(y))]
fpr_zeros, tpr_zeros, _ = metrics.roc_curve(y, zeros_probs)

# Plot the roc curve for the model
plt.plot(fpr_zeros, tpr_zeros, linestyle='--', label='No Model')
plt.plot(fpr, tpr, marker='.', label='Model')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Add legend
plt.legend()

plt.show()
