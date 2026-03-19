import numpy as np

import pandas as pd

class standardscaling:
    def __init__(self):
        self.mean = 0
        self.standard_deviation = 0    
        
    def fit(self, X):
        self.mean = np.mean(X)
        self.standard_deviation = np.std(X)
        return self.mean, self.standard_deviation
    
    def transform(self, X):
        if self.standard_deviation == 0:
            return (X-self.mean)
        else:
            return (X-self.mean)/self.standard_deviation
        
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    

df = pd.read_csv("/home/varsha/PracticePython/100DaysOfMl/Practice/SalaryData.csv")

X = df['Age']
# print(X[3])
# X = [100,120,130,150]

standardscaler = standardscaling()
X_scaled = standardscaler.fit_transform(X)

print(X_scaled)

