from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_regressor(data, args):
    X = data.drop('target', axis=1)  # Supponendo che 'target' sia la colonna target
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    regressor = RandomForestRegressor(random_state=args.seed, n_estimators=args.n_estimators)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
