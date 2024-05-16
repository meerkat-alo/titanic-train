import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle

from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split 

class TITANIC:
    def __init__(self, asset):
        self.asset = asset # for using alolib API
    
    def train(self, df, x_columns, y_column): 
        X = pd.get_dummies(df[x_columns])
        X_train, X_test, y_train, y_test = train_test_split(X, df[y_column], test_size=0.2, random_state=42)
        n_estimators = self.asset.load_args()['n_estimators']

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        precision = precision_score(y_test, y_pred, average='macro')
        
        # save trained model
        model_path = self.asset.get_model_path()  
        try: 
            with open(model_path + 'random_forest_model.pkl', 'wb') as file:
                pickle.dump(model, file)
        except Exception as e: 
            self.asset.save_error("Failed to save trained model" + str(e)) # error logging
        
        return model_path, precision
            