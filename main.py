from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, Form
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
MODEL_DIR = "saved_models"
from fastapi import Body

from fastapi.middleware.cors import CORSMiddleware

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # or ["http://localhost:3000"] for React
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app = FastAPI()

# CORS middleware (in case we connect frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
def read_root():
    return {"message": "ML Chatbot Backend is Running ✅"}

# Upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        preview = df.head().to_dict(orient="records")
        return {
            "filename": file.filename,
            "columns": list(df.columns),
            "preview": preview,
            "rows": len(df)
        }
    except Exception as e:
        return {"error": str(e)}

# Model options
class ColumnRequest(BaseModel):
    target_column: str

# Analyze dataset
@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...), request: ColumnRequest = None):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()  # sanitize column names

        target_col = request.target_column.strip()
        if target_col not in df.columns:
            return {"error": f"Column '{target_col}' not found. Available: {df.columns.tolist()}"}

        target_data = df[target_col]
        class_counts = target_data.value_counts().to_dict()
        num_classes = len(class_counts)
        sample_per_class = df.groupby(target_col).head(1).to_dict(orient="records")

        return {
            "target_column": target_col,
            "unique_classes": num_classes,
            "class_distribution": class_counts,
            "available_models": ["Logistic Regression", "Decision Tree", "Random Forest", "KNN"],
            "sample_per_class": sample_per_class
        }

    except Exception as e:
        return {"error": str(e)}

from fastapi import Form

from fastapi import UploadFile, File, Form
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    model_name: str = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df.columns = df.columns.str.strip()
        target_column = target_column.strip()

        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found. Available: {df.columns.tolist()}"}

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # One-hot encode categorical features in X
        X = pd.get_dummies(X)

        # Determine task type
        if y.dtype == object or y.nunique() <= 20:
            task_type = "classification"
        else:
            task_type = "regression"

        # Encode y if classification
        if task_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            class_names = le.classes_
        else:
            class_names = None

        stratify = y if task_type == "classification" and len(y) >= 10 and len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        # Model selection
        if task_type == "classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "KNN": KNeighborsClassifier()
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor()
            }

        if model_name not in models:
            return {"error": f"Model '{model_name}' is not supported for {task_type}."}

        model = models[model_name]
        model.fit(X_train, y_train)

        # Save model and its columns
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_filename = f"{MODEL_DIR}/{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, model_filename)

        # Save feature names
        with open(f"{MODEL_DIR}/{model_name.replace(' ', '_').lower()}_columns.json", "w") as f:
            json.dump(list(X.columns), f)

        # Evaluation
        y_pred = model.predict(X_test)

        if task_type == "classification":
            report = classification_report(y_test, y_pred, output_dict=True)
            matrix = confusion_matrix(y_test, y_pred).tolist()
            return {
                "task": "classification",
                "model": model_name,
                "accuracy": report["accuracy"],
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1_score": report["weighted avg"]["f1-score"],
                "confusion_matrix": matrix,
                "classes": class_names.tolist()
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return {
                "task": "regression",
                "model": model_name,
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
                "r2_score": r2
            }

    except Exception as e:
        return {"error": str(e)}


import os
@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        contents = await file.read()
        input_df = pd.read_csv(io.BytesIO(contents))

        # Load model
        model_filename = f"{MODEL_DIR}/{model_name.replace(' ', '_').lower()}_model.pkl"
        if not os.path.exists(model_filename):
            return {"error": f"Model '{model_name}' not found."}
        model = joblib.load(model_filename)

        # Load expected columns
        columns_path = f"{MODEL_DIR}/{model_name.replace(' ', '_').lower()}_columns.json"
        if not os.path.exists(columns_path):
            return {"error": f"Columns for model '{model_name}' not found."}
        with open(columns_path, "r") as f:
            training_columns = json.load(f)

        # Preprocess input
        input_df = pd.get_dummies(input_df)
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[training_columns]

        # Predict
        predictions = model.predict(input_df).tolist()
        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}

from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io, os, json, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
from fastapi import Body
@app.post("/predict_single")
async def predict_single(data: dict = Body(...)):
    try:
        model_name = data.pop("model_name", None)
        if not model_name:
            return {"error": "Model name is required."}

        model_key = model_name.replace(" ", "_").lower()
        model_path = os.path.join(MODEL_DIR, f"{model_key}_model.pkl")
        columns_path = os.path.join(MODEL_DIR, f"{model_key}_columns.json")

        if not os.path.exists(model_path):
            return {"error": f"Model '{model_name}' not found."}

        if not os.path.exists(columns_path):
            return {"error": f"Feature columns for model '{model_name}' not found."}

        with open(columns_path, "r") as f:
            training_columns = json.load(f)

        model = joblib.load(model_path)
        input_df = pd.DataFrame([data])
        input_df = pd.get_dummies(input_df)

        # Align input features
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[training_columns]

        prediction = model.predict(input_df).tolist()
        return {"prediction": prediction[0]}  # Always return single value

    except Exception as e:
        return {"error": str(e)}


@app.post("/train_all")
async def train_all_models(file: UploadFile = File(...), target_column: str = Form(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found."}

        df = df.dropna()
        X = df.drop(columns=[target_column])
        y = df[target_column]

        is_classification = y.dtype == 'object' or len(y.unique()) < 10
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(y)

        X = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "KNN Regressor": KNeighborsRegressor()
        }

        results = []
        best_score = -float("inf")
        best_model = None
        best_model_name = ""
        best_metrics = {}

        for name, model in models.items():
            try:
                is_regressor = "Regressor" in name or "Linear Regression" in name
                if is_classification and is_regressor:
                    continue
                if not is_classification and not is_regressor:
                    continue

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                if is_classification:
                    acc = accuracy_score(y_test, y_pred)
                    score = acc
                    metrics = {
                        "accuracy": acc,
                        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    }
                else:
                    r2 = r2_score(y_test, y_pred)
                    score = r2
                    metrics = {
                        "r2_score": r2,
                        "mse": mean_squared_error(y_test, y_pred),
                        "mae": mean_absolute_error(y_test, y_pred)
                    }

                results.append({
                    "model": name,
                    "score": round(score, 4),
                    **metrics
                })

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_model_name = name
                    best_metrics = metrics

            except Exception as e:
                results.append({"model": name, "error": str(e)})

        # Save best model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_filename = f"{MODEL_DIR}/{best_model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(best_model, model_filename)

        # Save input columns
        with open(f"{MODEL_DIR}/{best_model_name.replace(' ', '_').lower()}_columns.json", "w") as f:
            json.dump(list(X.columns), f)

        # Save best model name
        with open("last_model.json", "w") as f:
            json.dump({"model_name": best_model_name}, f)

        return {
            "best_model": best_model_name,
            "score": round(best_score, 4),
            "metrics": best_metrics,
            "all_models": results
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict_manual")
async def predict_manual(
    model_name: str = Form(...),
    features: str = Form(...)
):
    try:
        import json
        input_data = json.loads(features)  # Expecting a dict: {"feature1": val1, "feature2": val2, ...}
        input_df = pd.DataFrame([input_data])

        # Load model
        model_filename = f"{MODEL_DIR}/{model_name.replace(' ', '_').lower()}_model.pkl"
        if not os.path.exists(model_filename):
            return {"error": f"Model '{model_name}' not found."}
        model = joblib.load(model_filename)

        # One-hot encode and align
        input_df = pd.get_dummies(input_df)
        training_columns = model.feature_names_in_
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[training_columns]

        prediction = model.predict(input_df).tolist()
        return {"prediction": prediction}

    except Exception as e:
        return {"error": str(e)}

@app.get("/get_input_features")
def get_input_features(model_name: str):
    try:
        model_key = model_name.replace(' ', '_').lower()
        columns_path = f"{MODEL_DIR}/{model_key}_columns.json"

        if not os.path.exists(columns_path):
            return {"error": f"No columns found for model '{model_name}'."}

        with open(columns_path, "r") as f:
            input_columns = json.load(f)

        return {"features": input_columns}

    except Exception as e:
        return {"error": str(e)}





