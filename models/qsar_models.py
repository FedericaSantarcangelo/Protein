import pandas as pd
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, component):
        """
        Train and evaluate multiple regression models
        """
        models = {
            'Random Forest': (RandomForestRegressor(random_state=self.args.seed), {'n_estimators': [10, 25, 50],'max_depth': [3, 5, 7],'min_samples_split': [2, 5],'min_samples_leaf': [1, 2],'max_features': ['sqrt', 'log2'],'bootstrap': [True],'criterion': ['squared_error']}), 
            'AdaBoost': (AdaBoostRegressor(random_state=self.args.seed),{'n_estimators': [50, 100, 200, 500],'learning_rate': [0.01, 0.1, 0.5, 1.0]}),
            'Gradient Boosting': (GradientBoostingRegressor(random_state=self.args.seed),{'n_estimators': [10, 25, 50],'learning_rate': [0.1, 0.05],'max_depth': [3, 5],'subsample': [0.8, 1.0]}),  
            'MLP': (MLPRegressor(random_state=self.args.seed),{'hidden_layer_sizes': [(50,), (100,), (100, 100), (200, 100)],'alpha': [0.0001, 0.001, 0.01],'learning_rate_init': [0.001, 0.01]}), 
            'SVR': (SVR(),{'C': [0.1, 1, 10, 100],'gamma': ['scale', 'auto', 0.1, 1],'kernel': ['rbf', 'linear']}), 
            'KNN': (KNeighborsRegressor(),{'n_neighbors': [3, 5, 7, 10, 15],'weights': ['uniform'],'metric': ['euclidean', 'manhattan', 'chebyshev']}),
            'XGBoost': (XGBRegressor(random_state=self.args.seed),{'n_estimators': [10, 25, 50],'max_depth': [3, 5],'learning_rate': [0.1, 0.05],'subsample': [0.8, 1.0],'colsample_bytree': [0.8, 1.0]})
        }
        for model_name, (model, param_grid) in models.items():
            search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, scoring='r2')
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            y_train_pred = best_model.predict(X_train)
            r2_train = r2_score(y_train, y_train_pred)
            q2 = self.calculate_q2(best_model, X_test, y_test)
            self._save_results(best_model, X_test, y_test, model_name, component, best_params, r2_train, q2, y_train, y_train_pred)
            model_path = os.path.join(self.result_dir, f'file_pkl/{model_name}_best_model_{component}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(best_model, f)

    def _save_results(self, model, X_test, y_test, model_name, component, params, r2, q2,y_train, y_train_pred):
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
                filtered_results_df = results_df[results_df['R2'] > results_df['Q2']]
                filtered_results_df = filtered_results_df[filtered_results_df['R2'] < 1]
                if not filtered_results_df.empty:
                    filtered_results_df['R2+Q2'] = filtered_results_df['R2'] + filtered_results_df['Q2']
                    filtered_results_df = filtered_results_df.sort_values('R2+Q2', ascending=False)
                    filtered_results_df['Vote sum'] = range(1, len(filtered_results_df) + 1)
                    filtered_results_df['R2-Q2'] = filtered_results_df['R2']-filtered_results_df['Q2']
                    filtered_results_df = filtered_results_df.sort_values('R2-Q2', ascending=True)
                    filtered_results_df['Vote diff'] = range(1, len(filtered_results_df) + 1)
                    filtered_results_df['Vote'] = filtered_results_df['Vote sum'] + filtered_results_df['Vote diff']
                    filtered_results_df = filtered_results_df.sort_values('Vote', ascending=True)
                    filtered_results_df['(R2+Q2)-(R2-Q2)'] = filtered_results_df['R2+Q2']-filtered_results_df['R2-Q2']
                    filtered_results_df = filtered_results_df.sort_values('(R2+Q2)-(R2-Q2)', ascending=False)
                    index = filtered_results_df.index[0]
                    filtered_results_df.to_csv(os.path.join(self.result_dir, file), index=False)
                    best_results.append(filtered_results_df.loc[index].to_dict())
        best_results_df = pd.DataFrame(best_results)
        best_results_df.to_csv(os.path.join(self.result_dir, 'best_model.csv'), index=False)
        return best_results_df

    def retrain_best_model(self, X_train, y_train, X_test, y_test, seed=42):
        """
        Retrain the best models with the best parameters on the input data and test on other data,
        applying PCA transformation within this function.
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
            pred_path = os.path.join(self.result_dir, f'predictions/{model_name}_predictions_retrain.csv')
            os.makedirs(os.path.dirname(pred_path), exist_ok=True) 
            pred_results.to_csv(pred_path, index=False)

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

    def test_model(self, X_test, y_test):
        """
        Test a pre-trained model and compute evaluation metrics
        """
        best_model_info_df = pd.read_csv(os.path.join(self.result_dir, 'best_model.csv'))
        test_results = []
        for _, best_model_info in best_model_info_df.iterrows():
            model_name = best_model_info['Model']
            best_params = eval(best_model_info['Best Params'])
            num_components = best_model_info['PC']
            model_path = os.path.join(self.result_dir, f'file_pkl/{model_name}_best_model_{num_components}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                raise FileNotFoundError(f"Pre-trained model not found at {model_path}")
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