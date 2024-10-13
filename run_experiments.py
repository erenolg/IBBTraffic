import argparse
from utils import get_graph_topology
from node2vec import Node2Vec
from karateclub import GLEE, NetMF, GraphWave
import pickle
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score


def get_snapshot(timestamp, month="01", year="2023"):
    traffic_data_path = f"./data/{year}{month}/dynamic_X.csv"
    static_data_path = f"./data/{year}{month}/static_X.csv"
    traffic_df = pd.read_csv(traffic_data_path)
    static_df = pd.read_csv(static_data_path)

    snapshot = traffic_df.loc[traffic_df["TIMESTAMP"] == timestamp].sort_values(by=["NODE_ID"]).copy()

    prev_hour = traffic_df.loc[traffic_df["TIMESTAMP"] == timestamp-1].sort_values(by=["NODE_ID"]).copy()
    if timestamp > 168:
        prev_week = traffic_df.loc[traffic_df["TIMESTAMP"] == timestamp-(7*24)].sort_values(by=["NODE_ID"]).copy()

    density = calculate_density(snapshot)
    prev_hour_density = calculate_density(prev_hour)
    if timestamp > 168:
        prev_week_density = calculate_density(prev_week)

    snapshot.loc[:,"label"] = ((density > 0.5) & (snapshot["NUMBER_OF_VEHICLES"] > 10)).astype(int)
    print(prev_hour_density.mean())
    print(snapshot.shape)
    snapshot.loc[:,"prev_hour_density"] = prev_hour_density
    if timestamp > 168:
        snapshot.loc[:,"prev_week_density"] = prev_week_density

    train_features = ["NODE_ID","district","continent","population_density","freeway","prev_hour_density","label"]
    if timestamp > 168:
        train_features += ["prev_week_density"]
    snapshot = pd.merge(static_df, snapshot, on="NODE_ID")
    snapshot_dataframe = snapshot[train_features]
    return snapshot_dataframe


def calculate_density(snapshot, normalization="tanh"):
    num_of_vehicles = snapshot["NUMBER_OF_VEHICLES"].values
    n_num_of_vehicles = (num_of_vehicles - num_of_vehicles.min()) / (num_of_vehicles.max() - num_of_vehicles.min() + 1e-6)
    average_speed = snapshot["AVERAGE_SPEED"].values
    n_average_speed = (average_speed - average_speed.min()) / (average_speed.max() - average_speed.min() + 1e-6)

    if normalization == "sigmoid":
        density = 1 / (1 + np.exp(- (n_num_of_vehicles / (n_average_speed + 1e-6))))
    elif normalization == "tanh":
        density = np.tanh(n_num_of_vehicles / (n_average_speed + 1e-6))
    return density

def timestamp2time(timestamp):
    hour = timestamp % 24
    day = (timestamp // 24) + 1
    return day, hour

def time2timestamp(day, hour):
    timestamp = (day - 1) * 24 + hour
    return timestamp

def run_experiment(input,
                    target,
                    feats,
                    use_node_embeddings=True, 
                    embedding_method="glee", 
                    embedding_size=4):
    models = {"lr":LogisticRegression(max_iter=1000), "knn":KNeighborsClassifier(n_neighbors=3), "rfc":RandomForestClassifier(),\
          "xgb": XGBClassifier(), "catboost": CatBoostClassifier(verbose=0), "lgbm": LGBMClassifier(verbose=-1), "xtree": ExtraTreesClassifier()}

    input = input[feats + ["NODE_ID"]]
    print(input.columns)
    if use_node_embeddings:
        emb_path = f"./node_embeddings/{embedding_method}/{embedding_size}.pkl"
        with open(emb_path, "rb") as file:
            emb_file = pickle.load(file)
        embeddings = input["NODE_ID"].map(emb_file).apply(pd.Series)
        embeddings.columns = [f"e{i}" for i in embeddings.columns]
        X = pd.concat([input, embeddings], axis=1)
    else:
        X = input
    X = X.drop(columns=["NODE_ID"])
    X_train, X_test, y_train, y_test = train_test_split(X.values, target, test_size=0.2, random_state=2)
    for model_name, m in models.items():
      model = m
      m.fit(X_train, y_train)
      preds = model.predict(X_test)
      preds_proba = model.predict_proba(X_test)[:,1]
      f1 = f1_score(y_test, preds)
      acc = accuracy_score(y_test, preds)
      precision = precision_score(y_test, preds)
      recall = recall_score(y_test, preds)
      roc_auc = roc_auc_score(y_test, preds_proba)
      print(f"Model: {model_name:<8} || Acc: {acc:.5f} || F1: {f1:.5f} || Precision: {precision:.5f} || Recall: {recall:.5f} || AUC: {roc_auc:.5f}")
        

def main(args):
    df = get_snapshot(args.timestamp)
    X = df.drop(columns=["label"])
    y = df["label"]
    feats = ["district", "continent", "population_density", "freeway"]
    if args.use_traffic_feats:
        feats.extend(["prev_hour_density", "prev_week_density"])
    run_experiment(X, y, feats, use_node_embeddings=args.use_node_embeddings, 
                                embedding_method=args.emb_method,
                                embedding_size=args.emb_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument("-t", "--timestamp", type=int, help="Timestamp of snapshot (one timestamp for each hour)", default=200)
    parser.add_argument("--use-node-embeddings", action="store_true", help="Whether to use graph information")
    parser.add_argument("--use-traffic-feats", action="store_true", help="Whether to use traffic features")
    parser.add_argument("--emb-method", type=str, help="Embedding method to use", default="glee")
    parser.add_argument("--emb-size", type=int, help="Size of node embeddings", default=4)

    args = parser.parse_args()
    main(args)