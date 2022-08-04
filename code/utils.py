import torch
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
from sklearn import metrics
import logging
# torch.set_default_dtype(torch.double)
import pickle

def set_loggger(log_name, filename):
    logger = logging.getLogger(log_name)
    # set log level
    logger.setLevel(logging.INFO)
    # define file handler and set formatter
    file_handler = logging.FileHandler(filename)
    # formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    file_handler.setFormatter(formatter)
    # add file handler to logger
    logger.addHandler(file_handler)
    return logger

def data_formulate(data, adj_name = 'State_adj',test_size = 0.5, shuffle=False):
    # State_adj
    x_data = np.array(data[
        'X'],dtype=np.float16)  # shape (26,550,1289) (Task ID (Spatial location), Sample (Temporal-Days), Features(TF-IDF)) # Civil Unrest Data
    y_data = np.array(data['Y5'],dtype=np.float16)  # shape (26, 550) # Y5
    adj_data = np.array(data[adj_name],dtype=np.float16)  # shape (26, 26)

    y_data_3d = np.expand_dims(y_data, axis=2)
    data_concat = np.swapaxes(np.concatenate((x_data, y_data_3d), axis=2), 0, 1)
    data_train, data_test = train_test_split(data_concat, test_size=test_size, random_state=42, shuffle=shuffle)  # Split the set
    with torch.no_grad():
        X_train = torch.swapaxes(torch.FloatTensor(data_train[:, :, 0:-1]), 0, 1)  # Shape([26, 275, 1289])
        Y_train = torch.swapaxes(torch.FloatTensor(data_train[:, :, -1]), 0, 1)
        X_test = torch.swapaxes(torch.FloatTensor(data_test[:, :, 0:-1]), 0, 1)
        Y_test = torch.swapaxes(torch.FloatTensor(data_test[:, :, -1]), 0, 1)
    values = Y_train.unique()
    if values[0] == 0:  # This logic was added for fixing the label issue.
        Y_train += 1
        Y_test += 1
        print('Label increased by 1 to fix the label issue')
    return [X_train, Y_train, X_test, Y_test, adj_data]

def predict_from_minModel(model_location, X_test, Y_test):
    model_load_file = open(model_location, 'rb')
    MITOR_V2_trained = pickle.load(model_load_file)
    print(MITOR_V2_trained)
    [mze, mae, Y_pred] = MITOR_V2_trained.predict(X_test, Y_test, True)
    return [mze, mae]

def predict_from_mat(B, X_test, Y_test):
    values = Y_test.unique()
    k = values.shape[0]
    T = B[0:k-1,:]
    W = B[k-1:,:]
    TASK_num = Y_test.shape[0]
    samples = Y_test.shape[1]


    for i in range(TASK_num):
        X_hidden = X_test[i, :, :]
        tmp = torch.zeros((k, samples))
        out = torch.matmul(X_hidden, W[:, i])
        sig1= 1 / (1 + torch.exp(-(out+T[:, i][0])))
        sig2= 1 / (1 + torch.exp(-(out+T[:, i][1])))
        for j in range(k):
            if j==0:
                sig1 = 1 / (1 + torch.exp(-(out + T[:, i][j])))
                tmp[j,:]=sig1-0
            elif j==(k-1):
                sig2 = 1 / (1 + torch.exp(-(out + T[:, i][-1])))
                tmp[j,:]=1-sig2
            else:
                sig1 = 1 / (1 + torch.exp(-(out + T[:, i][j - 1])))
                sig2 = 1 / (1 + torch.exp(-(out + T[:, i][j])))
                tmp[j,:]=sig2-sig1
        pred = torch.argmax(tmp,dim=0)+1
        if i == 0:
            Y_pred = pred.unsqueeze(0)
        else:
            Y_pred = torch.cat((Y_pred, pred.unsqueeze(0)), 0)

    zero_one_sum = 0
    for i in range(TASK_num):
        zero_one_task = metrics.zero_one_loss(Y_test[i, :], Y_pred[i, :])
        zero_one_sum += zero_one_task
    mze = ((Y_pred != Y_test).long().sum() / Y_test.ravel().shape[0]).item()
    mae = (abs(Y_pred - Y_test).sum() / Y_test.ravel().shape[0]).item()


    return [mze, mae, Y_pred]

'''takes in a module and applies the specified weight initialization
Define a function that assigns weights by the type of network layer, then
Apply those weights to an initialized model using model.apply(fn), 
which applies a function to each model layer.
self.mlp_model.apply(utils.weights_init_uniform)
# self.mlp_model.apply(utils.weights_init_uniform_rule)
# self.mlp_model.apply(utils.weights_init_normal)'''
def weights_init_identity(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a identity matrix to the weights and a bias=0
        if m.in_features != m.out_features:
            print('Cannot be intialized with identity matrix. Make the input output dimension same.')
            return
        m.weight.data.copy_(torch.eye(m.in_features))
        m.bias.data.fill_(0)
def weights_init_uniform(m):

    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

def weights_init_uniform_rule(m):
    '''
    Good practice is to start your weights in the range of [-y, y] where y=1/sqrt(n)
    (n is the number of inputs to a given neuron).
    '''
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

def weights_init_normal(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
        # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        # m.bias.data should be 0
        m.bias.data.fill_(0)