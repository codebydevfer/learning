#1 - Imports & Setup

import os, re, joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

#2 - Config
 
RANDOM_STATE = 42
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test.csv")

#3 - Load Data

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

#4 - Feature Engineering

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Title from Name
    def extract_title(name: str) -> str:
        m = re.search(r",\s*([^\.]+)\.", str(name))
        title = m.group(1).strip() if m else "Unknown"
        title_map = {...}   # merge rare titles
        return title_map.get(title, title)

    out["Title"] = out["Name"].apply(extract_title)

    # Family-related
    out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)

    # Cabin present?
    out["CabinKnown"] = out["Cabin"].notna().astype(int)

    # Ticket group size
    ticket_counts = out["Ticket"].value_counts()
    out["TicketGroup"] = out["Ticket"].map(ticket_counts)

    # Fare per person
    out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
    return out

#5 - Split Features & Target

def split_features_target(df: pd.DataFrame, target_col: str = "Survived") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

#6 - Preprocessing Pipeline

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    candidate_numeric = ["Age","SibSp","Parch","Fare","FamilySize","IsAlone","TicketGroup","FarePerPerson"]
    candidate_categor = ["Pclass","Sex","Embarked","Title","CabinKnown"]

    num_cols = [c for c in candidate_numeric if c in X.columns]
    cat_cols = [c for c in candidate_categor if c in X.columns]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    return preprocessor, num_cols, cat_cols

#7 - Model Evaluation Helper

def evaluate(model, X_valid, y_valid, title="Model"):
    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {...}  # accuracy, precision, recall, f1, roc_auc
    print(metrics)

    # Confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    # ROC curve if probs available
    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_valid, y_proba)

    return metrics

#8 - Main Function (Training Workflow)

def main():
    # 1) Load + engineer features
    train_df = load_data(TRAIN_PATH)
    train_df = add_engineered_features(train_df)

    # 2) Split train/valid
    X, y = split_features_target(train_df)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # 3) Preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(X_train)

    # 4) Baseline: Logistic Regression
    logreg = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    logreg.fit(X_train, y_train)
    evaluate(logreg, X_valid, y_valid, title="Logistic Regression")

    # 5) Stronger model: Random Forest with GridSearch
    rf = Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])
    param_grid = {...}
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    evaluate(best_rf, X_valid, y_valid, title="Random Forest")

    # 6) Probability calibration
    calibrated = CalibratedClassifierCV(best_rf, cv=5, method="isotonic")
    calibrated.fit(X_train, y_train)
    evaluate(calibrated, X_valid, y_valid, title="Calibrated RF")

    # 7) Train on all data + save model
    final_model = CalibratedClassifierCV(grid.best_estimator_, cv=5, method="isotonic")
    final_model.fit(X, y)
    joblib.dump(final_model, "models/titanic_model.joblib")
