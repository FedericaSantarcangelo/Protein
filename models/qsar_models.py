"""Script to train and evaluate QSAR models
@Author: Federica Santarcangelo
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from models.utils import models


class QSARModelTrainer:
    def __init__(self, args):
        self.args = args
        self.result_dir = self.args.path_qsar
        os.makedirs(self.result_dir, exist_ok=True)
        self.best_model = None

    def calculate_q2(self, model, X_test, y_test):
        """ Compute the Q2 score for the model """
        y_pred = model.predict(X_test)
        numerator = np.sum((y_test - y_pred) ** 2)
        denominator = np.sum((y_test - np.mean(y_test)) ** 2)
        q2 = 1 - (numerator / denominator)
        return q2

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, component): #
        """
        Train and evaluate multiple regression models
        """
        for model_name, (model, param_grid) in models.items():
            search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring='r2')
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            y_train_pred = best_model.predict(X_train)
            r2_train = r2_score(y_train, y_train_pred)
            q2 = self.calculate_q2(best_model, X_test, y_test)
            self._save_results(best_model, X_test, y_test, model_name, best_params, r2_train, q2, component) #
            model_config = {
                'model': best_model,
                'params': best_params
            }
            model_path = os.path.join(self.result_dir, f'file_pkl/{model_name}_{component}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_config, f)

    def _save_results(self, model, X_test, y_test, model_name, params, r2, q2, component): #
        """
        Save the evaluation metrics and best parameters to a CSV file
        """
        y_pred = model.predict(X_test).ravel()
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        pred_results = pd.DataFrame({
        'y_test': y_test,
        'y_pred': y_pred
        })
        pred_path = os.path.join(self.result_dir, f'predictions/{model_name}_{component}_predictions_train.csv')
        os.makedirs(os.path.dirname(pred_path), exist_ok=True) 
        pred_results.to_csv(pred_path, index=False)
        results = {
            'Model': model_name,
            'PC': component, #
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

    def retrain_best_model(self, X_train, y_train, X_test, y_test, seed=42):
        """
        Retrain the best models with the best parameters on the input data and test on other data,
        applying PCA transformation within this function.
        """
        best_model_info_df = pd.read_csv(os.path.join(self.result_dir, 'best_model.csv'))
        retrain_results = []
        for _, best_model_info in best_model_info_df.iterrows():
            model_name = best_model_info['Model']
            num_components = best_model_info['PC']
            if num_components > X_train.shape[1]:
               num_components = X_train.shape[1]
            X_train_subset = X_train[:, :num_components]
            X_test_subset = X_test[:, :num_components]
            model_path = os.path.join(self.result_dir, f'file_pkl/{model_name}_{num_components}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_config = pickle.load(f)
                    model = model_config['model']
                    best_params = model_config['params']
            else:
                raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
            model.set_params(**best_params)
            model.fit(X_train_subset, y_train)
            y_train_pred = model.predict(X_train_subset)
            y_test_pred = model.predict(X_test_subset)
            r2 = r2_score(y_train, y_train_pred)
            q2 = self.calculate_q2(model, X_test_subset, y_test)
            mse = mean_squared_error(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            retrain_results.append({
                'Model': model_name,
                'R2': r2,
                'Q2': q2,
                'MSE': mse,
                'MAE': mae
            })
            pred_results = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_test_pred
            })
            pred_path = os.path.join(self.result_dir, f'predictions/{model_name}_predictions_retrain_{num_components}.csv')
            os.makedirs(os.path.dirname(pred_path), exist_ok=True) 
            pred_results.to_csv(pred_path, index=False)
        retrain_results_df = pd.DataFrame(retrain_results)
        retrain_results_df.to_csv(os.path.join(self.result_dir, 'best_retrain.csv'), index=False)

    def test_model(self, X_test, y_test):
        """
        Test a pre-trained model and compute evaluation metrics
        """
        best_model_info_df = pd.read_csv(os.path.join(self.result_dir, 'best_model.csv'))
        test_results = []
        for _, best_model_info in best_model_info_df.iterrows():
            model_name = best_model_info['Model']
            num_components = best_model_info['PC']
            model_path = os.path.join(self.result_dir, f'file_pkl/{model_name}_{num_components}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_config = pickle.load(f)
                    model = model_config['model']
                    best_params = model_config['params']
            else:
                raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
            model.set_params(**best_params)
            X_test_subset = X_test[:, :num_components]
            y_pred = model.predict(X_test_subset)
            q2 = self.calculate_q2(model, X_test_subset, y_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            results = {
                'Model': model_name,
                'Q2': q2,
                'MSE': mse,
                'MAE': mae
            }
            test_results.append(results)
            pred_results = pd.DataFrame({
            'y_test': y_test,
            'y_pred': y_pred
            })
            pred_path = os.path.join(self.result_dir, f'predictions/{model_name}_predictions_test.csv')
            os.makedirs(os.path.dirname(pred_path), exist_ok=True) 
            pred_results.to_csv(pred_path, index=False)
        results_df = pd.DataFrame(test_results)
        results_df.to_csv(os.path.join(self.result_dir, 'test_results.csv'), index=False)