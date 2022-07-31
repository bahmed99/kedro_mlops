import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
   
    X = data.drop("TARGET", axis=1)
    y = data["TARGET"]
    ss=StandardScaler()
    X=ss.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    
    xgboost=xgb.XGBClassifier(
    objective="binary:logistic",
    scale_pos_weight=8,
    eval_metric="auc",
    max_depth=7,
    tree_method='gpu_hist'
    
    )
    xgboost.fit(X_train,y_train)
    return xgboost


def evaluate_model(
    xgboost: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
):
   
    y_pred = xgboost.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a accuracy on test data.", acc)
    logger.info("Model has a f1 score on test data.", f1)