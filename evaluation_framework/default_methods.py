






def default_preprocess_train_data(train_data, configs):
    preprocessed_train_data = train_data
    return preprocessed_train_data


def default_preprocess_test_data(test_data, preprocessed_train_data, configs):
    preprocessed_test_data = test_data
    return preprocessed_test_data


def default_evaluate_prediction(preprocessed_test_data, prediction_result):
    
    return prediction_result.mean()







#######################################################################
#######################################################################
## CAUTION : Although you can, try not to override the below methods ##
#######################################################################
#######################################################################
# If you have an unusual way of fitting the model or predicting using a fitted model,
# please specify them in the Estimator class's fit() and predict() method, rather
# than overriding the methods below. 

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
    
    preprocessed_test_data = preprocessed_test_data.reset_index(drop=True)
    
    X = preprocessed_test_data[feature_names]
    print(X)
    preprocessed_test_data['specialEF_float32_predictions'] = trained_estimator.predict(X)

    prediction_result = preprocessed_test_data[['specialEF_float32_UUID', 'specialEF_float32_predictions']]    
    return prediction_result


