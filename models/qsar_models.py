""" 
@Author: Federica Santarcangelo
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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

        if self.args.model in ['rf_regressor', 'all']:
            self._rf_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['ab_regressor', 'all']:
            self._ab_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['mlp_regressor', 'all']:
            self._mlp_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['svr_regressor', 'all']:
            self._svr_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['xgb_regressor', 'all']:
            self._xgb_regressor(X_train, y_train, X_test, y_test)

    def _evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model using multiple metrics
        """
        y_pred = model.predict(X_test)
    
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    
        return {'MSE': mse, 'R2': r2, 'MAE': mae}

    def _rf_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Random Forest regressor using GridSearchCV
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_depth': [None, 10, 20, 30]
        }
        grid_search = GridSearchCV(RandomForestRegressor(random_state=self.args.seed), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        results = self._evaluate_model(best_model, X_test, y_test)
        results['Model'] = 'Random Forest'
        results['Best Params'] = grid_search.best_params_

        self._save_results(results, 'rf_regressor_results.csv')

    def _ab_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate an AdaBoost regressor using GridSearchCV
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(AdaBoostRegressor(random_state=self.args.seed), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        results = self._evaluate_model(best_model, X_test, y_test)
        results['Model'] = 'AdaBoost'
        results['Best Params'] = grid_search.best_params_

        self._save_results(results, 'ab_regressor_results.csv')

    def _mlp_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Multi-Layer Perceptron regressor using GridSearchCV
        """
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
        grid_search = GridSearchCV(MLPRegressor(random_state=self.args.seed), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        results = self._evaluate_model(best_model, X_test, y_test)
        results['Model'] = 'MLP'
        results['Best Params'] = grid_search.best_params_

        self._save_results(results, 'mlp_regressor_results.csv')

    def _svr_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Support Vector Regressor using GridSearchCV
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['linear', 'rbf']
        }
        grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        results = self._evaluate_model(best_model, X_test, y_test)
        results['Model'] = 'SVR'
        results['Best Params'] = grid_search.best_params_

        self._save_results(results, 'svr_regressor_results.csv')

    def _xgb_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate an XGBoost regressor using GridSearchCV
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        grid_search = GridSearchCV(XGBRegressor(random_state=self.args.seed), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        results = self._evaluate_model(best_model, X_test, y_test)
        results['Model'] = 'XGBoost'
        results['Best Params'] = grid_search.best_params_

        self._save_results(results, 'xgb_regressor_results.csv')

    def _save_results(self, results, filename):
        """
        Save the results to a CSV file
        """
        results_df = pd.DataFrame([results])
        results_path = os.path.join(self.result_dir, filename)
        results_df.to_csv(results_path, index=False)