import scipy.io
import utils
import scipy.io
import torch

# Reading Data
# _sample
B_location = '../data/debug_data//B.mat'
Y_test_location = '../data/debug_data//Y_test.mat'
X_test_location = '../data/debug_data//X_test.mat'

B_mat = scipy.io.loadmat(B_location)
B = torch.tensor(B_mat['B']).float()

Y_test_mat = scipy.io.loadmat(Y_test_location)
Y_test_m = torch.tensor(Y_test_mat['Y_test']).float()

X_test_mat = scipy.io.loadmat(X_test_location)
X_test_m = torch.tensor(X_test_mat['X_test']).float()

[mze, mae, Y_pred] = utils.predict_from_mat(B, X_test_m, Y_test_m)
print('On test set: Mean zero-one Error = {} Mean Absolute Error = {}'.format(mze,mae))