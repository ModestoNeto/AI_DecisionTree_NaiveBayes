import pandas as pd
from sklearn.model_selection import train_test_split

class Data: 
    @staticmethod
    def loader(file_path):
        df = pd.read_csv(file_path)
        
        fetures = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB']
        X = df[fetures]
        Y = df['CO2EMISSIONS']
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        return X_train, X_test, Y_train, Y_test