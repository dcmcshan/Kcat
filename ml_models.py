from typing import List, Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def prepare_ml_data(results: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning.

    Args:
        results (List[Dict]): List of analysis results for each structure.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X (features) and y (target) arrays for ML.
    """
    X = []
    y = []
    for result in results:
        features = []
        for res_data in result['dihedral_data'].values():
            features.extend([res_data['cos_phi'], res_data['cos_psi'], res_data['sin_phi'], res_data['sin_psi']])
        if features:  # Only include if we have features (i.e., dihedral angles were calculated)
            X.append(features)
            y.append(result['kcat_mut'])
    
    return np.array(X), np.array(y)

def train_and_evaluate_model(X: np.ndarray, y: np.ndarray, model_type: str = 'rf'):
    """
    Train and evaluate a machine learning model.

    Args:
        X (np.ndarray): Feature array.
        y (np.ndarray): Target array.
        model_type (str): Type of model to use ('rf' for Random Forest, 'gb' for Gradient Boosting).

    Returns:
        Tuple[Pipeline, Dict]: Trained model pipeline and evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'rf' or 'gb'.")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }

    return pipeline, metrics