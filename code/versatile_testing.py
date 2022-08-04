'This file can run model developed at different faces'

import configure as config
import utils
import pickle
import scipy.io

def versatile_test(model_name, data_location, neg_w=False, base=False, adj_name = 'State_adj', y_key = 'Y5'):
    'Capable to run differently trained model'
    print("Testing Model")
    print('Data Loaded from {}'.format(data_location))

    model_location = '../saved_model/' + model_name
    print("Loading Model from {}".format(model_location))

    model_load_file = open(model_location, 'rb')
    MITOR_V2_trained = pickle.load(model_load_file)
    # # MITOR_V2_trained.to(device)
    print(vars(MITOR_V2_trained))
    data = scipy.io.loadmat(data_location)
    if base:
        [X_train, Y_train, X_test, Y_test, adj_data] = utils.data_formulate(data = data,
                                                                            adj_name = adj_name, y_key=y_key)
        config.using_mlp = False
    else:
        config.using_mlp = True
        [X_train, Y_train, X_test, Y_test, adj_data] = utils.data_formulate(data)
        if hasattr(MITOR_V2_trained.mlp_model, 'hidden_dim_2') != True:  # for running the old models
            MITOR_V2_trained.mlp_model.hidden_dim_2 = None

    if neg_w:
        MITOR_V2_trained.W = -MITOR_V2_trained.W
    # if hasattr(MITOR_V2_trained, 'hidden_dim_2')!=True:
    #     MITOR_V2_trained.mlp_model.hidden_dim_2= None
    [mze, mae, Y_pred] = MITOR_V2_trained.predict(X_test, Y_test, return_pred=True)
    test_str = "On Test Data: Mean Zero-one Error = {}, Mean Absolute Error = {}".format(mze, mae)
    print(test_str)
    # Y_pred_dict = {'Y_pred': Y_pred.detach().cpu().numpy()}
    # Y_test_dict = {'Y_test': Y_test.detach().cpu().numpy()}
    # scipy.io.savemat('../data/debug_data/y_pred_flu12wihY_woMLP.mat', Y_pred_dict)
    # scipy.io.savemat('../data/debug_data/y_test_flu12wihY_woMLP.mat', Y_test_dict)

if __name__ == '__main__':

    # base model

    versatile_test('28_04_2022_08_07_02_MITOR_V2.obj', config.data_brazil, neg_w=True, base=True)  # Brazil data, base model
    versatile_test('02_06_2022_05_56_14_MITOR_V2.obj', config.data_flu_11_12, neg_w=True, base=True)  # Flu 11-12, base model

    # based on old model
    versatile_test('03_05_2022_16_10_54_MITOR_V2_mlp.obj', config.data_argentina, neg_w=True) # Argentina
    versatile_test('28_04_2022_08_34_05_MITOR_V2_mlp.obj', config.data_brazil, neg_w=True)  # Brazil
    versatile_test('03_05_2022_17_01_31_MITOR_V2_mlp.obj', config.data_chile, neg_w=True)  # Chile
    versatile_test('03_05_2022_17_23_02_MITOR_V2_mlp.obj', config.data_columbia, neg_w=True)  # Columbia
    versatile_test('03_05_2022_17_40_07_MITOR_V2_mlp.obj', config.data_mexico, neg_w=True)  # Mexico
    versatile_test('03_05_2022_17_48_48_MITOR_V2_mlp.obj', config.data_paraguay, neg_w=True)  # Paraguay
    versatile_test('03_05_2022_17_54_10_MITOR_V2_mlp.obj', config.data_uruguay, neg_w=True)  # Uruguay
    versatile_test('03_05_2022_18_48_09_MITOR_V2_mlp.obj', config.data_venezuela, neg_w=True)  # Venezuela
    # based on new model
    versatile_test('04_07_2022_10_38_33_MITOR_V2_mlp500.obj', config.data_flu_11_12)  # Flu 11-12
    # versatile_test('', config.data_flu_13_14, neg_w=True)  # Flu 13-14