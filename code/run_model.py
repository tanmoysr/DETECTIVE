import configure as config
import utils
import model
# Reading Data
import scipy.io
import pickle
# import torch

def train():
    # logger
    logger = utils.set_loggger(__name__, config.param_file)

    # data processing
    config.write_params()
    data = scipy.io.loadmat(config.data_location) # data format: dict
    [X_train, Y_train, X_test, Y_test, adj_data] = utils.data_formulate(data)
    print('Data Loaded')

    # defining model
    MITOR_V2 = model.Model(config.input_dim, config.hidden_dim_1, config.hidden_dim_2, logger)
    print(config.dt_string)
    print("Training Model")
    for adi_split in range(0, config.adi_range, config.split_range):
        if adi_split==0:
            trained_param = MITOR_V2.fit(X_train, Y_train, adj_data, config.lr, config.alpha,
                                         config.beta, config.split_range, adi_split)
        else:
            trained_param = MITOR_V2.fit(X_train, Y_train, adj_data, config.lr, config.alpha,
                                         config.beta, config.split_range, adi_split, False)
        logger.info('Training {} Complete'.format(adi_split+config.adi_range))
        print("----Saving Splitted Model----")
        model_dump_file = open(config.model_location + '{}.obj'.format(adi_split+config.split_range), 'wb')
        pickle.dump(MITOR_V2, model_dump_file)
        print("Testing Model")
        [mze, mae] = MITOR_V2.predict(X_test, Y_test)
        test_str_split = "On Test Data: Mean Zero-one Error = {}, Mean Absolute Error = {}".format(mze, mae)
        [mze_min, mae_min] = utils.predict_from_minModel(config.minModel_file, X_test, Y_test)
        test_str_split_min = "With min model on Test Data: Mean Zero-one Error = {}, Mean Absolute Error = {}".format(mze_min, mae_min)
        print(test_str_split)
        logger.info(test_str_split)
        print(test_str_split_min)
        logger.info(test_str_split_min)

    print("----Saving Model----")
    model_dump_file = open(config.model_location +'.obj', 'wb')
    pickle.dump(MITOR_V2, model_dump_file)
    print("Model Saved")

    print("Testing Model")
    [mze, mae] = MITOR_V2.predict(X_test, Y_test)
    test_str = "On Test Data: Mean Zero-one Error = {}, Mean Absolute Error = {}".format(mze, mae)
    print(test_str)
    logger.info(test_str)
    logger.disabled = True

def test(model_name):
    print("Testing Model")

    data = scipy.io.loadmat(config.data_location)
    [X_train, Y_train, X_test, Y_test, adj_data] = utils.data_formulate(data)
    print('Data Loaded from {}'.format(config.data_location))

    model_location = '../saved_model/' + model_name
    print("Loading Model from {}".format(model_location))

    model_load_file = open(model_location, 'rb')
    MITOR_V2_trained = pickle.load(model_load_file)
    # # MITOR_V2_trained.to(device)
    print(vars(MITOR_V2_trained))
    if hasattr(MITOR_V2_trained, 'hidden_dim_2')!=True:
        MITOR_V2_trained.mlp_model.hidden_dim_2= None
    [mze, mae, Y_pred] = MITOR_V2_trained.predict(X_test, Y_test, return_pred=True)
    test_str = "On Test Data: Mean Zero-one Error = {}, Mean Absolute Error = {}".format(mze, mae)
    print(test_str)
    # Y_pred_dict = {'Y_pred': Y_pred.detach().cpu().numpy()}
    # Y_test_dict = {'Y_test': Y_test.detach().cpu().numpy()}
    # scipy.io.savemat('../data/debug_data/y_pred_flu12wihY_woMLP.mat', Y_pred_dict)
    # scipy.io.savemat('../data/debug_data/y_test_flu12wihY_woMLP.mat', Y_test_dict)

if __name__ == '__main__':
    train()
    # test(input('What is the model name?: '))
    test('04_07_2022_10_38_33_MITOR_V2_mlp500.obj') # Flu 11-12



