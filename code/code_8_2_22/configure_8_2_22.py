from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
data_location = '../data/flu_datasets/2011-2012_flu_normalized.mat'
'''
../data/MITOR_Data.mat
../data/flu_datasets/2011-2012_flu_data.mat
../data/flu_datasets/2011-2012_flu_data.mat
../data/flu_datasets/2011-2012_flu_normalized.mat
../data/flu_datasets/2013-2014_flu_normalized.mat
../data/civil_datasets/2013-2014_argentina.mat
../data/civil_datasets/2013-2014_brazil.mat
../data/civil_datasets/2013-2014_chile.mat
../data/civil_datasets/2013-2014_colombia.mat
../data/civil_datasets/2013-2014_mexico.mat
../data/civil_datasets/2013-2014_paraguay.mat
../data/civil_datasets/2013-2014_uruguay.mat
../data/civil_datasets/2013-2014_venezuela.mat
../data/matlab_variables/W_init_train_brazil.mat
../data/matlab_variables/W_init_train_flu1_1_12.mat'
'''
train_weight_location = '../data/matlab_variables/W_init_train_flu1_1_12.mat'
w_from_matlab = False # False when using MLP
break_on = True
using_mlp = True
if using_mlp==True:
    model_location = '../saved_model/'+dt_string+'_MITOR_V2_mlp.obj'
    training_progress_log = '../saved_model/'+dt_string+'_tr_prog_mlp.csv'
    param_file = '../saved_model/'+dt_string+'_params_mlp.txt'
else:
    model_location = '../saved_model/' + dt_string + '_MITOR_V2.obj'
    training_progress_log = '../saved_model/' + dt_string + '_tr_prog.csv'
    param_file = '../saved_model/' + dt_string + '_params.txt'

input_dim = 525 # argentina: 1250, brazil/sample: 1289, chille: 1215, columbia: 1231,
# mexico: 1263, paraguay: 1106, uruguay: 1150, venezuela: 1252, Flu: 525
hidden_dim_1 = 525 # starting from 1k, 1200:800:600:200:10
hidden_dim_2 = 0 # starting from 1k, 1200:800:600:200:10
# Parameters
rho = 10
# learning rate for gradient descent for final layer
lr= 0.8  # default: 0.08 # play with this
mlp_optimizer = 'adam' # adam, sgd
if mlp_optimizer=='sgd':
    learning_rate_mlp = 0  # Mitor_data: 0.01, flu 1e-4
    sgd_momentum = 0 #1e-10
    sgd_dampening = 0 #1e-10
    sgd_weight_decay = 0 #1e-10
    sgd_nesterov = False
    optimizer_config_lines = ['lr for MLP: {}'.format(learning_rate_mlp),
                              'sgd_momentum: {}'.format(sgd_momentum),
                              'sgd_dampening: {}'.format(sgd_dampening),
                              'sgd_weight_decay: {}'.format(sgd_weight_decay),
                              'sgd_nesterov: {}'.format(sgd_nesterov)]
elif mlp_optimizer=='adam':
    learning_rate_mlp = 0 # Mitor_data: 0.01, Flu: 1e-3
    adam_weight_decay = 0  # Default: 0, Mior data: 0, Flu: 1e-2
    adam_amsgrad = False # Default: False, Mitor data: False, Flu: True
    optimizer_config_lines = ['lr for MLP: {}'.format(learning_rate_mlp),
                              'adam_weight_decay: {}'.format(adam_weight_decay),
                              'adam_amsgrad: {}'.format(adam_amsgrad)]
else:
    optimizer_config_lines = ['N/A']

gradient_clipping = False
if gradient_clipping==True:
    clip_max_norm = 0.001
    clip_norm_type = 2
    clip_error_if_nonfinite = True
    clip_config_lines = ['clip_max_norm: {}'.format(clip_max_norm),
                         'clip_norm_type: {}'.format(clip_norm_type),
                         'clip_error_if_nonfinite: {}'.format(clip_error_if_nonfinite)]
else:
    clip_config_lines = ['N/A']

# L2,1 norm co-effecient (can be chose using validation set)
alpha=0 # default:0, Mitor_data: 1 (flu: 1e-6)
# threshold theta constriants co-efficient (can be chose using validation set)
beta=1e-3 # default: 0.01 controling adjacency matrix
# Tolerance
tol_L = 1e-3 # default: 1e-3, wo_mlp: 1e-3
tol_r = 1e-10 # default: 1e-5
tol_s = 1e-10 # default: 1e-5
tol_r_s = 10 # default: 10, Do not change it, wo_mlp: 20
# set max number of iterations for GD on W,T  update (can be shorter if meet stop criteria)
adi_range = 5000 #default: 500, Mitor_data:27
max_iteration= 1000 #default: 1000

def write_params():
    lines = ['model name: {}'.format(model_location),
             'data: {}'.format(data_location),
             'w_from_matlab: {}'.format(w_from_matlab),
             'break_on: {}'.format(break_on),
             'rho {}'.format(rho),
             'lr for ADMM: {}'.format(lr),
             'alpha: {}'.format(alpha),
             'beta: {}'.format(beta),
             'tol_L: {}'.format(tol_L),
             'tor_r: {}'.format(tol_r),
             'tol_s: {}'.format(tol_s),
             'tol_r_s: {}'.format(tol_r_s),
             'ADI: {}'.format(adi_range),
             'Max_iteration: {}'.format(max_iteration)]
    if using_mlp==True:
        lines.append('using_mlp: {}'.format(using_mlp))
        lines.append('Hidden Dimension 1: {}'.format(hidden_dim_1))
        lines.append('Hidden Dimension 2: {}'.format(hidden_dim_2))
        lines.append('optimzer for MLP: {}'.format(mlp_optimizer))
        for optim_line in optimizer_config_lines:
            lines.append(optim_line)
        lines.append('gradient_clipping: {}'.format(gradient_clipping))
        for clip_line in clip_config_lines:
            lines.append(clip_line)
    with open(param_file, 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    f.close()

# Try to with max_iteration 1000 and check the performance
# loss.backward() for nlp layer
# check code migration by using raw features
# Check all the hyper parameter and initialization
# Store all the updates matlab vs pytorch for first iteration and compare

# initialization from matlab. Double check all the updated variable.
# run for 1 adi for each of the variable. old u, v, w vs new u, v, w.
# Check performance testing.
# Check more specifically with u, v, w.