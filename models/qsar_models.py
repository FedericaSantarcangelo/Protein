import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

        if self.args.model in ['lin_regressor', 'all']:
            self._lin_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['rf_regressor', 'all']:
            self._rf_regressor(X_train, y_train, X_test, y_test)
        if self.args.model in ['mlp_regressor', 'all']:
            self._mlp_regressor(X_train, y_train, X_test, y_test)

    def _lin_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a regression model
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'Model': 'Linear Regression',
            'MSE': mse,
            'R2': r2
        }
        
        self._save_results(results, 'lin_regressor_results.csv')
    
    def _rf_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a classification model
        """
        model = RandomForestRegressor(random_state=self.args.seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'Model': 'Random Forest Regressor',
            'MSE': mse,
            'R2': r2
        }
        
        self._save_results(results, 'rf_regressor_results.csv')

    def _mlp_regressor(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate a Multi-Layer Perceptron classifier
        """
        model = MLPRegressor(
            hidden_layer_sizes=(self.args.hidden_layer_sizes,),
            activation=self.args.activation,
            solver=self.args.solver,
            alpha=self.args.alpha,
            learning_rate=self.args.learning_rate,
            learning_rate_init=self.args.learning_rate_init,
            tol=self.args.tol,
            early_stopping=self.args.early_stopping,
            validation_fraction=self.args.validation_fraction,
            beta_1=self.args.beta_1,
            beta_2=self.args.beta_2,
            epsilon=self.args.epsilon,
            random_state=self.args.seed
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results = {
            'Model': 'MLP Regressor',
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
        print(f"Results saved to {results_path}")