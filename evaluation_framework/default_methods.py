




def default_preprocess_train_data(train_data, configs):
    
    preprocessed_train_data = train_data
    return preprocessed_train_data



def default_preprocess_test_data(test_data, preprocessed_train_data, configs):

    preprocessed_test_data = test_data
    return preprocessed_test_data



def default_model_fit(preprocessed_train_data, hyperparameters, estimator, feature_names, target_name):
    
    X = preprocessed_train_data[feature_names]  # need optimization
    
    if False:
        pass
        # read from memmap
    else:
        y = preprocessed_train_data[target_name]
        
    estimator.fit(X, y, hyperparameters)
    
    return estimator


    
def default_model_predict(preprocessed_test_data, trained_estimator, feature_names, target_name):
    
    X = preprocessed_test_data[feature_names]
    prediction_result = trained_estimator.predict(X)

    # consider returning two arrays target nd pred
    
    return prediction_result



def default_evaluate_prediction(preprocessed_test_data, prediction_result):
    
    return 100

# def default_store_prediction()
