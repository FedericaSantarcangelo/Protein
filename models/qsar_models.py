from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from joblib import parallel_backend
import os
import pandas as pd

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

    def _sv_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a regression model
        """
   
        best_model = SVR(C=100, epsilon=1, kernel='rbf')
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
    
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'Model': 'SVR',
            'Best Params': {'C': 100, 'epsilon': 1, 'kernel': 'rbf'},
            'MSE': mse,
            'R2': r2
        }

        self._save_results(results, 'svr_results.csv')

    def _rf_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Random Forest regressor
        """
    
        best_model = RandomForestRegressor(
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=2,
            min_samples_split=2,
            n_estimators=100,
            random_state=self.args.seed
        )
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'Model': 'Random Forest Regressor',
            'Best Params': {
                'max_depth': None,
                'max_features': 'sqrt',
                'min_samples_leaf': 2,
                'min_samples_split': 2,
                'n_estimators': 100
            },
            'MSE': mse,
            'R2': r2
        }
        self._save_results(results, 'rf_regressor_results.csv')

    def _save_results(self, results, filename):
        """
        Save the results to a CSV file
        """
        results_df = pd.DataFrame([results])
        results_path = os.path.join(self.result_dir, filename)
        results_df.to_csv(results_path, index=False)