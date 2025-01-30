import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class QSARModelTrainer:
    def __init__(self, args):
        self.args = args
        self.result_dir = self.args.path_qsar
        os.makedirs(self.result_dir, exist_ok=True)
        self.best_model = None

    def calculate_q2(self, model, X_test, y_test):
        """ Calcola il Q² sul test set senza riaddestrare il modello """
        y_pred = model.predict(X_test)  # Usa il modello già addestrato
        numerator = np.sum((y_test - y_pred) ** 2)
        denominator = np.sum((y_test - np.mean(y_test)) ** 2)
        q2 = 1 - (numerator / denominator)
    
        return q2

    def train_and_evaluate(self, X, y, component):
        """
        Train and evaluate multiple regression models
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.args.seed)
        models = {
            'PLS': (PLSRegression(), {'n_components': range(1, min(X_train.shape[1], len(X_train)) + 1)}),
            'Random Forest': (RandomForestRegressor(random_state=self.args.seed), {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}),
            'AdaBoost': (AdaBoostRegressor(random_state=self.args.seed), {'n_estimators': [50, 100]}),
            'Gradient Boosting': (GradientBoostingRegressor(random_state=self.args.seed), {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.01]}),
            'MLP': (MLPRegressor(random_state=self.args.seed), {'hidden_layer_sizes': [(100,), (100, 100)], 'alpha': [0.0001, 0.001]}),
            'SVR': (SVR(), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}),
            'KNN': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
            'XGBoost': (XGBRegressor(random_state=self.args.seed), {'n_estimators': [100, 200], 'max_depth': [3, 6, 9]})
        }

        for model_name, (model, param_grid) in models.items():
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            y_train_pred = best_model.predict(X_train)
            r2_train = r2_score(y_train, y_train_pred)
            q2 = self.calculate_q2(best_model, X_test, y_test)

            self._save_results(best_model, X_test, y_test, model_name, component, best_params, r2_train, q2)

    def _save_results(self, model, X_test, y_test, model_name, component, params, r2, q2):
        """
        Save the evaluation metrics and best parameters to a CSV file
        """
        y_pred = model.predict(X_test).ravel()
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results = {
            'Model': model_name,
            'PC': component,
            'MSE': mse,
            'R2': r2,
            'Q2': q2,
            'MAE': mae,
            'Best Params': params
        }
        results_path = os.path.join(self.result_dir, f'{model_name.lower().replace(" ", "_")}_results.csv')

        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
            results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        else:
            results_df = pd.DataFrame([results])

        results_df.to_csv(results_path, index=False)
        return results

    def select_best_model(self):
        """
        Select the best model based on the highest Q2 score
        """
        best_results = []
        for file in os.listdir(self.result_dir):
            if file.endswith('_results.csv'):
                results_df = pd.read_csv(os.path.join(self.result_dir, file))
            
            # Filter results where R2 > Q2
                filtered_results_df = results_df[results_df['R2'] > results_df['Q2']]
                if not filtered_results_df.empty:
                # Select the best result based on the highest Q2 score
                    best_result = filtered_results_df.loc[filtered_results_df['Q2'].idxmax()]
                    best_results.append(best_result)

        best_results_df = pd.DataFrame(best_results)
        best_results_df.to_csv(os.path.join(self.result_dir, 'best_model.csv'), index=False)
        return best_results_df

    def retrain_best_model(self, X_train, y_train, X_test, y_test):
        """
        Retrain the best models with the best parameters on the input data and test on other data
        """
        best_model_info_df = pd.read_csv(os.path.join(self.result_dir, 'best_model.csv'))
        retrain_results = []

        for _, best_model_info in best_model_info_df.iterrows():
            model_name = best_model_info['Model']
            best_params = eval(best_model_info['Best Params'])
            num_components = best_model_info['PC']
        
            if num_components > X_train.shape[1]:
                num_components = X_train.shape[1]
        
            X_train_subset = X_train[:, :num_components]
            X_test_subset = X_test[:, :num_components]
        
            model = self._get_model_by_name(model_name, best_params)
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_test_subset)
        
            r2 = r2_score(y_test, y_pred)
            q2 = self.calculate_q2(model, X_test_subset, y_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
        
            retrain_results.append({
                'Model': model_name,
                'R2': r2,
                'Q2': q2,
                'MSE': mse,
                'MAE': mae
            })
    
        retrain_results_df = pd.DataFrame(retrain_results)
        retrain_results_df.to_csv(os.path.join(self.result_dir, 'best_retrain.csv'), index=False)
        
    def _get_model_by_name(self, model_name, params):
        """
        Get the model instance by name and set the parameters
        """
        models = {
            'PLS': PLSRegression,
            'Random Forest': RandomForestRegressor,
            'AdaBoost': AdaBoostRegressor,
            'Gradient Boosting': GradientBoostingRegressor,
            'MLP': MLPRegressor,
            'SVR': SVR,
            'KNN': KNeighborsRegressor,
            'XGBoost': XGBRegressor
        }
        model_class = models[model_name]
        model = model_class(**params)
        return model