from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import os
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

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

        if self.args.model in ['svr_regressor', 'all']:
            self._sv_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['rf_regressor', 'all']:
            self._rf_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['gb_regressor', 'all']:
            self._gb_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['xgb_regressor', 'all']:
            self._xgb_regressor(X_train, y_train, X_test, y_test)

    def _sv_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a regression model using GridSearchCV for SVR
        """
        param_distributions = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1],
            'kernel': ['rbf', 'sigmoid']
        }
        random_search = RandomizedSearchCV(SVR(), param_distributions, n_iter=5, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=self.args.seed)
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'Model': 'SVR',
            'Best Params': random_search.best_estimator_,
            'MSE': mse,
            'R2': r2
        }

        self._save_results(results, 'svr_results.csv')

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

    def _gb_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Gradient Boosting regressor using GridSearchCV
        """
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(GradientBoostingRegressor(random_state=self.args.seed), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'Model': 'Gradient Boosting',
            'Best Params': grid_search.best_params_,
            'MSE': mse,
            'R2': r2
        }
        self._save_results(results, 'gb_regressor_results.csv')

    
    def _xgb_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate an XGBoost regressor using GridSearchCV
        """
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        grid_search = GridSearchCV(xgb.XGBRegressor(random_state=self.args.seed, use_label_encoder=False, eval_metric='rmse'), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'Model': 'XGBoost',
            'Best Params': grid_search.best_params_,
            'MSE': mse,
            'R2': r2
        }
        self._save_results(results, 'xgb_regressor_results.csv')

    def _save_results(self, results, filename):
        """
        Save the results to a CSV file
        """
        results_df = pd.DataFrame([results])
        results_path = os.path.join(self.result_dir, filename)
        results_df.to_csv(results_path, index=False)