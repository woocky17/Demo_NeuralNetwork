import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_and_prepare_data(csv_path, features, target, test_size=0.3, random_state=42):
    df = pd.read_csv(csv_path)
    X = df[features].values
    y = df[[target]].values
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, df
