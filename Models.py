from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error

class Treino:
    @staticmethod
    def train_decision_tree(X_train, X_test, y_train, y_test):
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        return y_pred, mse

    @staticmethod
    def train_naive_bayes(X_train, X_test, y_train, y_test):
        model = GaussianNB()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        return y_pred, mse
