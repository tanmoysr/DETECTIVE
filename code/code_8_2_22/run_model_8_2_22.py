import configure as config
import utils
import model
# Reading Data
import scipy.io
import pickle
# import torch
# torch.set_default_dtype(torch.double)

# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
#     print("Running on the GPU")
#     print("Number of GPU {}".format(torch.cuda.device_count()))
#     print("GPU type {}".format(torch.cuda.get_device_name(0)))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
#     torch.backends.cudnn.benchmark = False
#     # torch.cuda.memory_summary(device=None, abbreviated=False)
#     # print(torch.backends.cudnn.version())
# else:
#     device = torch.device("cpu")
#     print("Running on the CPU")

def train():
    # logger
    logger = utils.set_loggger(__name__, config.param_file)
    # print(logger.__dict__)

    # data processing
    config.write_params()
    data = scipy.io.loadmat(config.data_location) # data format: dict
    [X_train, Y_train, X_test, Y_test, adj_data] = utils.data_formulate(data)
    print('Data Loaded')

    # defining model
    # MITOR_V2 = model.Model(config.input_dim, config.hidden_dim_1)
    MITOR_V2 = model.Model(config.input_dim, config.hidden_dim_1, config.hidden_dim_2, logger)
    # MITOR_V2.to(device)
    print("Training Model")
    trained_param = MITOR_V2.fit(X_train,Y_train,adj_data,config.lr, config.alpha, config.beta)
    logger.info('Training Complete')

    print("----Saving Model----")
    model_dump_file = open(config.model_location, 'wb')
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

    data = scipy.io.loadmat(config.data_location)  # data format: dict
    [X_train, Y_train, X_test, Y_test, adj_data] = utils.data_formulate(data)
    print('Data Loaded')

    model_location = '../saved_model/' + model_name
    print("Loading Model from {}".format(model_location))

    model_load_file = open(model_location, 'rb')
    MITOR_V2_trained = pickle.load(model_load_file)
    # # MITOR_V2_trained.to(device)
    [mze, mae, Y_pred] = MITOR_V2_trained.predict(X_test, Y_test, return_pred=True)
    test_str = "On Test Data: Mean Zero-one Error = {}, Mean Absolute Error = {}".format(mze, mae)
    print(test_str)
    Y_pred_dict = {'Y_pred': Y_pred.detach().cpu().numpy()}
    Y_test_dict = {'Y_test': Y_test.detach().cpu().numpy()}
    scipy.io.savemat('../data/debug_data/y_pred_flu12wihY_woMLP.mat', Y_pred_dict)
    scipy.io.savemat('../data/debug_data/y_test_flu12wihY_woMLP.mat', Y_test_dict)
    # print(Y_pred)
    # print(Y_test)

if __name__ == '__main__':
    train()
    # test(input('What is the model name?: ')) # 22_05_2022_14_23_35_MITOR_V2_mlp.obj
    # 22_05_2022_20_21_15_MITOR_V2_mlp.obj for Uruguay with mlp
    # 22_05_2022_20_34_34_MITOR_V2_mlp.obj for Brazil with mlp
    # 22_05_2022_20_59_37_MITOR_V2.obj for Brazil without mlp



