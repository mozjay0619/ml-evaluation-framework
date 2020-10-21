import xgboost as xgb

class XgboostRegressor():
    
    def __init__(self):
        pass
    
    def fit(self, X, y, parameters):
        
        learning_rate = parameters[0]
        gamma = parameters[1]
        max_depth = int(parameters[2])
        n_estimators = int(parameters[3])
        learning_rate = learning_rate / float(n_estimators)
        min_child_weight = int(parameters[4])
        colsample_bytree = parameters[5]
        subsample = parameters[6]

        algo = xgb.XGBRegressor(objective='reg:squarederror', 
                                learning_rate=learning_rate,
                                gamma=gamma,
                                max_depth=max_depth,
                                n_estimators=n_estimators,
                                min_child_weight=min_child_weight,
                                colsample_bytree=colsample_bytree,
                                subsample=subsample,
                                n_jobs=2,
                                tree_method='hist')
        
        self.model_object = algo.fit(X, y)

        
    def predict(self, X):

        
        return self.model_object.predict(X)

