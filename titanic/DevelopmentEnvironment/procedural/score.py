import titanic.DevelopmentEnvironment.procedural.preprocessing_functions as pf
import titanic.DevelopmentEnvironment.procedural.config as config

# import preprocessing_functions as pf
# import config

# =========== scoring pipeline =========

# impute categorical variables
def predict(data):
    
    # extract first letter from cabin
    data['cabin'] = pf.extract_cabin_letter(data, 'cabin')

    # impute NA categorical
    for var in config.CATEGORICAL_VARS:
        data[var + '_NA'] = pf.add_missing_indicator(data, var)
        data[var] = pf.impute_na(data, var)
    
    
    # impute NA numerical
    for var in config.NUMERICAL_TO_IMPUTE:
        data[var] = pf.impute_na(X_train, var, replacement=config.IMPUTATION_DICT[var])
    
    
    # Group rare labels
    for var in config.CATEGORICAL_VARS:
        data[var] = pf.remove_rare_labels(data,
                                          var,
                                          config.FREQUENT_LABELS[var])

    return data

    #
    # # encode variables
    # for var in config.CATEGORICAL_VARS:
    #     data = pf.encode_categorical(data, var)
    #
    #
    #
    # # check all dummies were added
    #
    #
    # # scale variables
    # data = pf.scale_features(data, config.OUTPUT_SCALER_PATH)
    #
    # # make predictions
    # predictions = pf.predict(data, config.OUTPUT_MODEL_PATH)


    
    return predictions

# ======================================
    
# small test that scripts are working ok
    
if __name__ == '__main__':
        
    from sklearn.metrics import accuracy_score    
    import warnings
    warnings.simplefilter(action='ignore')
    
    # Load data
    data = pf.load_data(config.PATH_TO_DATASET)
    
    X_train, X_test, y_train, y_test = pf.divide_train_test(data,
                                                            config.TARGET)
    
    pred = predict(X_test)
    # # encode variables
    for var in config.CATEGORICAL_VARS:
        print(var)
        data = pf.encode_categorical(pred, var)

    data = pf.scale_features(pred, config.OUTPUT_SCALER_PATH)
    
    # evaluate
    # if your code reprodues the notebook, your output should be:
    # test accuracy: 0.6832
    print('test accuracy: {}'.format(accuracy_score(y_test, pred)))
    print()
        