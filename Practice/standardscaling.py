import numpy as np

import pandas as pd

class standardscaling:
    def __init__(self):
        self.mean = 0
        self.standard_deviation = 0    
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.standard_deviation = np.std(X, axis=0)
    
    def transform(self, X):
        return (X - self.mean)/self.standard_deviation
        
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

df = pd.read_csv("/home/varsha/PracticePython/100DaysOfMl/Practice/SalaryData.csv")
X = df['Age'].values.reshape(-1,1)

standardscaler = standardscaling()
X_scaled = standardscaler.fit_transform(X)

print(X_scaled)




        
        



