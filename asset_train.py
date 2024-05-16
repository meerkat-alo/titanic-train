#asset_[step_name].py
 
# -*- coding: utf-8 -*-
from datetime import datetime, timezone
import os
import random
import sys
from alolib.asset import Asset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from titanic_source import TITANIC
#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config() # config from input asset
        self.data       = self.asset.load_data() # data from input asset 
 
    @Asset.decorator_run
    def run(self):
        df = self.data['dataframe0']
        x_columns = self.config['x_columns']
        y_column = self.config['y_column']
        
        titanic = TITANIC(self.asset)
        model_path, precision = titanic.train(df, x_columns, y_column)
        self.config['model_path'] = model_path 
        
        self.asset.save_data(self.data)
        self.asset.save_config(self.config)

        score = random.uniform(0.1, 1.0)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.asset.save_summary(
            result=f"precision: {precision}",  
            note=f"Test Titanic-demo (date: {now})",
            score=score,
        )
 
 
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
