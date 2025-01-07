import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend

class QSARModelTrainer:
    def __init__(self, args):
        self.args = args
        self.result_dir = self.args.path_qsar
        os.makedirs(self.result_dir, exist_ok=True)

    def train_and_evaluate(self, X, y):
        """
        Train and evaluate the models
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.args.seed)

        # Standardize data for SVR and MLPRegressor
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if self.args.model in ['sv_regressor', 'all']:
            self._sv_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['rf_regressor', 'all']:
            self._rf_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['mlp_regressor', 'all']:
            self._mlp_regressor(X_train, y_train, X_test, y_test)

    def _sv_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a regression model
        """
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 1],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        with parallel_backend('threading'):
            grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'Model': 'SVR',
            'Best Params': grid_search.best_params_,
            'MSE': mse,
            'R2': r2
        }
        
        self._save_results(results, 'sv_regressor_results.csv')
    
    def _rf_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Random Forest regressor model
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        with parallel_backend('threading'):
            grid_search = GridSearchCV(RandomForestRegressor(random_state=self.args.seed), param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'Model': 'Random Forest Regressor',
            'Best Params': grid_search.best_params_,
            'MSE': mse,
            'R2': r2
        }
        
        self._save_results(results, 'rf_regressor_results.csv')

    def _mlp_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Multi-Layer Perceptron regressor
        """
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['tanh', 'relu'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [500, 1000]
        }
        with parallel_backend('threading'):
            grid_search = GridSearchCV(MLPRegressor(random_state=self.args.seed), param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'Model': 'MLP Regressor',
            'Best Params': grid_search.best_params_,
            'MSE': mse,
            'R2': r2
        }

        self._save_results(results, 'mlp_regressor_results.csv')
    
    def _save_results(self, results, filename):
        """
        Save the results to a CSV file
        """
        results_df = pd.DataFrame([results])
        results_path = os.path.join(self.result_dir, filename)
        results_df.to_csv(results_path, index=False)