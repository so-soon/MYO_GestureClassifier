import numpy as np
import pandas as pd
import random
from pprint import pprint

from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions
from helper_functions import train_test_split, calculate_accuracy





ges = ['','Pain in neck' , 'headache', 'Injection', 'Hearing-Aid', 'Nurse' , 'Blood Pressure', 'Surgery', 'Test', 'Prescription', 'Wheelchair']
def transform_label(value):
    return ges[int(value)]



random.seed(0)
# Split the data



def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]

    return df_bootstrapped


def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest


def random_forest_predictions(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    return random_forest_predictions





def get_forest():
    dataset = pd.read_csv("/Users/sosoon/coding/myo_train_data/train_x_last.csv")
    dataset["label"] = dataset.gesture
    dataset = dataset.drop("gesture", axis=1)

    column_names = []
    for column in dataset.columns:
        name = column.replace(" ", "_")
        column_names.append(name)
    dataset.columns = column_names

    dataset["label"] = dataset.label.apply(transform_label)

    train_df, test_df = train_test_split(dataset, test_size=0.2)

    forest = random_forest_algorithm(train_df, n_trees=20, n_bootstrap=800, n_features=2, dt_max_depth=4)
    #predictions = random_forest_predictions(test_df, forest)

    return forest

get_forest()