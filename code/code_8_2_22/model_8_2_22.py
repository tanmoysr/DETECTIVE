import configure as config
import utils
import torch
import scipy.io
import numpy as np
import torch.nn.functional as F
import proximal_gradient.proximalGradient as pg
from sklearn import metrics
import pandas as pd
import time
from decimal import Decimal
import math


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
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2=0):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim_1)
        # changed in 9th edition
        # self.fc1.weight.data.copy_(torch.eye(self.hidden_dim_1))  # weight initialization by identity matrix
        # self.fc1.bias.data.fill_(0)  # weight initialization by zero
        if hidden_dim_2!=0:
            self.hidden_dim_2 = hidden_dim_2
            self.fc2 = torch.nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        else:self.hidden_dim_2 = None
    def forward(self, x):
        if self.hidden_dim_2 != None:
            X_hidden = F.relu(self.fc2(self.fc1(x)))
        else:
            X_hidden = F.relu(self.fc1(x)) # relu
        return X_hidden


class Model:
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2=0, logger = None):
        if logger==None:
            self.no_init_logger = True
            self.logger = utils.set_loggger(__name__, config.param_file)
        else:
            self.no_init_logger = False
            self.logger = logger
            # self.logger.name = __name__ # raise pickle.PicklingError('logger cannot be pickled')
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        if config.using_mlp==True:
            self.mlp_model = MLP(self.input_dim, self.hidden_dim_1, self.hidden_dim_2)
            # self.mlp_model.apply(utils.weights_init_uniform) # Uniform is performing better
            # self.mlp_model.apply(utils.weights_init_uniform_rule)
            # self.mlp_model.apply(utils.weights_init_normal)
            if config.mlp_optimizer=='sgd':
                self.optimizer = torch.optim.SGD(self.mlp_model.parameters(), lr=config.learning_rate_mlp, momentum=config.sgd_momentum,
                                                 dampening=config.sgd_dampening, weight_decay=config.sgd_weight_decay, nesterov=config.sgd_nesterov) # momentum=0.9, weight_decay=0, nesterov=false
            elif config.mlp_optimizer == 'adam':
                # self.optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=config.learning_rate_mlp)
                self.optimizer = torch.optim.Adam(self.mlp_model.parameters(), lr=config.learning_rate_mlp,
                                                  weight_decay=config.adam_weight_decay, amsgrad=config.adam_amsgrad)
        self.logger.info('Model Initialized')

    def logsig2(self, X):
        return torch.log(1 / (1 + torch.exp(-X)))

    def fit(self, X_train,Y_train,adj_data,lr,alpha,beta, pred_too=True):

        adi_list = []
        adi_time_list = []
        max_iter_list = []
        r_list = []
        s_list = []
        rho_list = []
        mze_list = []
        mae_list = []

        values = Y_train.unique()
        # if values[0]==0: # This logic was added for fixing the label issue.
        #     Y_train+=1
        #     values = Y_train.unique()
            # print(values)

        k = values.shape[0]
        TRAIN_size = Y_train.shape[1]
        TASK_num = Y_train.shape[0]
        BIG = float('inf')
        SMALL = float('-inf')
        # d=demension of feature vector
        if config.using_mlp:
            if self.hidden_dim_2!=0:
                d = self.hidden_dim_2
            else:
                d = self.hidden_dim_1
        else:
            d = self.input_dim
        with torch.no_grad():
            R = torch.zeros((TASK_num, TASK_num))
            for i in range(TASK_num):
                nb = sum(adj_data[:, i] == 1) # bug fixed coverting 1 to i
                if nb > 0:
                    R[:, i] = -torch.from_numpy(adj_data[:, i] / nb) # bug fixed coverting 1 to i
                    R[i, i] = 1
            L = torch.matmul(R, np.transpose(R))
            E = torch.eye((TASK_num))
            # initialize parameters for training data for each task(size= d*TASK_num)
            if config.w_from_matlab:
                W_mat = scipy.io.loadmat(config.train_weight_location)
                W = torch.tensor(W_mat['W']).float()
            else:
                W = torch.nn.Parameter(torch.empty(d, TASK_num).normal_(mean=0, std=0.1)) # default: 0.1 # compare this with matlab
            # transfer matlab initialization and load it to pytorch. In that case follow the matlab parameter literraly.
            U = W
            Y1 = torch.zeros(W.shape)
            # initialize threshold size : (k-1)*TASK_num
            pi = torch.nn.Parameter(torch.ones(k - 1, 1) * 1 / k)
            # assign each class with same probability
            cum_pi = torch.cumsum(pi, dim=0)
            T = torch.log(cum_pi / (1 - cum_pi))
            T = T.repeat(1, TASK_num)
            Y2 = torch.zeros(T.shape)
            V = T
            Y3 = torch.zeros(k - 2, T.shape[1])
            # initialize learning rate lambda
            lambda_w = lr #lr
            lambda_t = lr #lr
            # admm param
            rho = config.rho

            # set max number of iterations for GD on W,T  update (can be shorter if meet stop critira)
            Max_iteration = config.max_iteration
            f1inv = torch.linalg.inv(beta * L + 2 * rho * E)
            f2inv = torch.linalg.inv(beta * L + rho * E)

        # outer loop (W{Mitor W+h(x)}, U, V, Y, )

        for adi in range(config.adi_range):
            with torch.no_grad():
                adi_tic = time.perf_counter()
                L_old = float('inf') # Previous loss
                I = torch.zeros(Max_iteration)
                L_train = torch.zeros(Max_iteration)

            # Update W,T via gradient decent
            for t in range(Max_iteration):
                # print(t)
                print("ADI: {}, Iteration: {}".format(adi, t), end='\r')
                # cross tasks overall loss
                loss = 0
                with torch.no_grad():
                    GW = torch.zeros((d, TASK_num))
                    GT = torch.zeros((k - 1, TASK_num))
                # changed for version 9
                if config.using_mlp:
                    self.optimizer.zero_grad()
                for i in range(TASK_num):
                    for j in range(TRAIN_size):
                        # print(i,j)
                        # print("ADI: {}, Iteration: {}, Task: {}, Sample: {}".format(adi, t, i, j), end='\r')
                        # calculate what is the model predict is
                        if config.using_mlp:
                            # self.optimizer.zero_grad()
                            X_hidden = self.mlp_model(X_train[i,j,:])
                        else:
                            X_hidden = X_train[i,j,:] # Alert: It will not match with matlab due to random_suffle in data processing
                        if Y_train[i,j]==k:
                            sig1=1
                            # sig2= self.logsig(T[int(Y_train[i,j])-2,i] - torch.matmul(X_hidden,W[:,i])) # logsig result is not same as matlab
                            sig2= 1 / (1 + torch.exp(-(T[int(Y_train[i,j])-2,i] - torch.matmul(X_hidden,W[:,i]))))
                            # sig2 = 1 / (1 + torch.exp(-(T[int(Y_train[i, j]) - 2, i] + torch.matmul(X_hidden, W[:, i])))) # Bug warning: according to eq1 it may be +
                            dif=SMALL # default: SMALL (Bug Warning: Not sure if it is BIG or not)
                        elif Y_train[i,j]==1:
                            # sig1= self.logsig(T[int(Y_train[i,j])-1,i] - torch.matmul(X_hidden,W[:,i])) # Bug Warning: why not just sigmoid? Equ 1 paper
                            sig1= 1 / (1 + torch.exp(-(T[int(Y_train[i,j])-1,i] - torch.matmul(X_hidden,W[:,i]))))
                            sig2=0
                            dif=SMALL
                        else:
                            # sig1= self.logsig2(T[int(Y_train[i,j])-1,i] - torch.matmul(X_hidden,W[:,i]))
                            sig1= 1 / (1 + torch.exp(-(T[int(Y_train[i,j])-1,i] - torch.matmul(X_hidden,W[:,i]))))
                            # sig1 = 1 / (1 + torch.exp(-(T[int(Y_train[i, j]) - 1, i] + torch.matmul(X_hidden, W[:, i])))) # Bug warning: according to eq1 it may be +
                            # sig2= self.logsig2(T[int(Y_train[i,j])-2,i] - torch.matmul(X_hidden,W[:,i]))
                            sig2= 1 / (1 + torch.exp(-(T[int(Y_train[i,j])-2,i] - torch.matmul(X_hidden,W[:,i]))))
                            # sig2 = 1 / (1 + torch.exp(-(T[int(Y_train[i, j]) - 2, i] + torch.matmul(X_hidden, W[:, i])))) # Bug warning: according to eq1 it may be +
                            dif= (T[int(Y_train[i,j])-2,i]-T[int(Y_train[i,j])-1,i])
                        # loss

                        l_cum=-torch.log(sig1-sig2)

                        if(l_cum==float('inf')):
                            loss_str = "error: inf loss, adi {}, t {}, task {}, sample {}, Y_value {}, l_cum {}".format(adi, t, i, j, Y_train[i,j], l_cum)
                            print(loss_str)
                            # Logs
                            self.logger.info(loss_str)
                            # break # this break is not in matlab.
                        loss=loss+l_cum
                        # loss = loss.detach()
                        # Commented out in 9th edition
                        # if config.using_mlp:
                        #     torch.autograd.backward(l_cum)
                        #     if config.gradient_clipping==True:
                        #         try:
                        #             torch.nn.utils.clip_grad_norm_(self.mlp_model.parameters(), max_norm=config.clip_max_norm,
                        #                                            norm_type=config.clip_norm_type, error_if_nonfinite=config.clip_error_if_nonfinite)
                        #             # torch.nn.utils.clip_grad_value_(self.mlp_model.parameters(), clip_value=1.0)
                        #         except Exception as e:
                        #             print(e)
                        #             print(adi, t, i, j)
                        #             break
                        #
                        #     self.optimizer.step()
                        # else:
                        #     pass
                        # we want to minimize p_cum w.r.t W and T
                        # Gradient Decent
                        gradient_w = X_hidden*(1-sig1-sig2)
                        GW[:,i]=GW[:,i]+gradient_w
                        with torch.no_grad():
                            gradient_t=torch.zeros(k-1)
                        # Form canonical vector e_yi and e_yi-1
                        if Y_train[i,j]==k:
                            gradient_t[int(Y_train[i,j])-2]=sig2
                        elif Y_train[i,j]==1:
                            gradient_t[int(Y_train[i,j])-1]=sig1-1
                        else:
                            gradient_t[int(Y_train[i,j])-1]=sig1-1+1/(1-torch.exp(-dif))
                            gradient_t[int(Y_train[i,j])-2]=sig2-1+1/(1-torch.exp(dif))
                        GT[:,i]=GT[:,i]+gradient_t
                # added in 9th edition
                if config.using_mlp:
                    # torch.autograd.backward(loss)
                    loss.backward()
                    if config.gradient_clipping:
                        try:
                            torch.nn.utils.clip_grad_norm_(self.mlp_model.parameters(), max_norm=config.clip_max_norm,
                                                           norm_type=config.clip_norm_type,
                                                           error_if_nonfinite=config.clip_error_if_nonfinite)
                            # torch.nn.utils.clip_grad_value_(self.mlp_model.parameters(), clip_value=1.0)
                        except Exception as e:
                            print(e)
                            print(adi + adi_split, t, i, j)
                            break

                    self.optimizer.step()
                else:
                    pass

                with torch.no_grad():
                    GW = 1 / TRAIN_size * 1 / TASK_num * GW + Y1 + rho * (W - U)
                    GT = 1 / TRAIN_size * 1 / TASK_num * GT + Y2 + rho * (T - V)
                    # update parameters
                    W = W - lambda_w * GW
                    T = T - lambda_t * GT
                    I[t] = t
                    # print('before L-train')
                    L_train[t] = 1 / TRAIN_size * 1 / TASK_num * loss + torch.sum(torch.sum(torch.matmul(Y1, torch.transpose((W - U), 0, 1)))) + (rho / 2.) * torch.sum(
                        torch.sum(torch.square((W - U)))) + torch.sum(torch.sum(torch.matmul(Y2,  torch.transpose((T - V), 0, 1)))) + (rho / 2) * torch.sum(
                        torch.sum(torch.square(T - V))) # bug fixed by using matmul and transpose
                    # Alert: Very little difference with matlab. In matlab the L_train(t)=1.5367e-04, in python 0.0002
                    if config.break_on:
                        if L_old-L_train[t]<config.tol_L:
                            print("L_break")
                            break
                    L_old = L_train[t]


            # Update U
            # print('Update U')
            U_old = U
            U = W + 1 / rho * Y1
            pg.l21(U, reg=alpha / rho) # Alert: If alpha =0 then there is no difference between Matlab (prox_l21) and Python


            # Update V
            # print('Update V')
            V_old = V
            V[0, :] = 1 / rho * Y2[0, :] + T[0, :] # bug fixed converting 1 to 0

            for i in range(1, k - 1):
                f1 = torch.matmul(
                    (torch.matmul(beta * V[i - 1, :], L) + Y2[i, :] + rho * (T[i, :] + V[i - 1, :]) + Y3[i - 1, :]), f1inv)
                f2 = torch.matmul((torch.matmul(beta * V[i - 1, :], L) + Y2[i, :] + rho * T[i, :] + Y3[i - 1, :]), f2inv)
                V[i, f1 <= V[i - 1, :]] = f1[f1 <= V[i - 1, :]]
                V[i, f2 > V[i - 1, :]] = f2[f2 > V[i - 1, :]]
            # Update Y
            # print('Update Y')
            Y1 = Y1 + rho * (W - U)
            Y2 = Y2 + rho * (T - V) # Alert: T, V is matching with matlab. But somehow, Y2 is not. May be some granular level differences.
            for i in range(1, k - 1):
                Y3[i - 1, :] = torch.maximum(Y3[i - 1, :] + rho * (V[i - 1, :] - V[i, :]), torch.zeros(TASK_num))

            # Compute ADMM residuals and update rho correspondingly
            # print("Compute ADMM residuals and update rho correspondingly")
            s = rho * torch.sqrt(torch.sum(torch.sum(torch.square(U_old - U)))) + rho * torch.sqrt(
                torch.sum(torch.sum(torch.square(V_old - V)))) # bug warning: 1/10 smaller than matlab output
            r = torch.sqrt(torch.sum(torch.sum(torch.square(U - W)))) + torch.sqrt(torch.sum(torch.sum(torch.square(V - T))))
            # Alert: second part of equation is not matching. While in matlab before sqrt is 1.23e-12, in python 1.98e-13
            if (r > config.tol_r_s * s): # bug warning: Maybe 10 is very big. Most of the time this condision is not triggered
                rho = 2 * rho
                #   update inverse matrix for computation of V-update when rho changes
                f1inv = torch.linalg.inv(beta * L + 2 * rho * E)
                f2inv = torch.linalg.inv(beta * L + rho * E)
            else:
                if (config.tol_r_s  * r < s): # bug warning: Maybe 10 is very big. Most of the time this condision is not triggered
                    rho = rho / 2
                    #       update inverse matrix for computation of V-update when rho changes
                    f1inv = torch.linalg.inv(beta * L + 2 * rho * E)
                    f2inv = torch.linalg.inv(beta * L + rho * E)

            self.T = T
            self.W = -W

            adi_toc = time.perf_counter()
            time_taken = adi_toc-adi_tic
            adi_list.append(adi)
            adi_time_list.append(time_taken)
            max_iter_list.append(t)
            r_list.append(r.item())
            s_list.append(s.item())
            rho_list.append(rho)
            if pred_too == True:
                # [mze, mae] = self.predict(X_train, Y_train, [T, -W, self.fc1.weight.data, self.fc1.bias.data])
                [mze, mae] = self.predict(X_train, Y_train)
                mze_list.append(mze)
                mae_list.append(mae)
                logger_str = 'ADMM iteration: {}, Max iteration: {}, r-residual: {:.6E}, s-residual: {:.6E}, rho: {}, mze: {}, mae: {}, time taken {}'.format(adi+1, t+1, Decimal(r.item()), Decimal(s.item()), rho, mze, mae, time_taken)

            else:
                logger_str = 'ADMM iteration: {}, Max iteration: {}, r-residual: {}, s-residual: {}, rho: {}, time taken {}'.format(adi+1, t+1, r, s, rho, time_taken)
            print(logger_str)
            # Logs
            self.logger.info(logger_str)


            # Check for stop criteria
            if config.break_on:
                if (math.isnan(r) and math.isnan(s)):
                    print("R_S_break due to NaN")
                    # print(X_hidden, V, U)
                    break
                if (r<config.tol_r and s<config.tol_s):
                    print("R_S_break")
                    break


        # Saving Results
        # dictionary of lists
        # if pred_too == True:
        #     result_dict = {'iteration': adi_list, 'max_iter': max_iter_list, 'r-residual': r_list, 's_residual': s_list, 'rho': rho_list, 'mze': mze_list, 'mae': mae_list, 'time': adi_time_list}
        # else:
        #     result_dict = {'iteration': adi_list, 'max_iter': max_iter_list, 'r-residual': r_list, 's_residual': s_list,
        #                    'rho': rho_list, 'time': adi_time_list}
        #
        # result_df = pd.DataFrame(result_dict)

        # saving the dataframe
        # result_df.to_csv(config.training_progress_log, index=False)
        # The final learned model parameter set
        # model_param = torch.cat((T, -W), 0)
        # if pred_too==True:
        #     [mze, mae] =self.predict(X_train, Y_train, [T, -W, self.fc1.weight.data, self.fc1.bias.data])

        self.logger.disabled = self.no_init_logger
        if config.using_mlp:
            return [self.T, self.W, self.mlp_model.fc1.weight.data, self.mlp_model.fc1.bias.data]
        else:
            return [self.T, self.W, None, None]

    # def predict(self, X_test, Y_test, trained_param):
    def predict(self, X_test, Y_test, return_pred = False):
        if config.using_mlp: # added to 9th edition
            self.mlp_model.eval()

        # [T, W, self.fc1.weight.data, self.fc1.bias.data] = trained_param
        with torch.no_grad():
            TASK_num = Y_test.shape[0]
            samples = Y_test.shape[1]
            k = self.T.shape[0] + 1
            for i in range(TASK_num):
                if config.using_mlp:
                    X_hidden = self.mlp_model(X_test[i, :, :])
                else:
                    X_hidden = X_test[i, :, :]

                tmp = torch.zeros((k, samples))
                out = torch.matmul(X_hidden, self.W[:, i])
                # sig1 = 1 / (1 + torch.exp(-(out + self.T[:, i][0])))
                # sig2 = 1 / (1 + torch.exp(-(out + self.T[:, i][1])))
                for j in range(k):
                    if j == 0:
                        sig1 = 1 / (1 + torch.exp(-(out + self.T[:, i][j])))
                        tmp[j, :] = sig1 - 0
                    elif j == (k - 1):
                        sig2 = 1 / (1 + torch.exp(-(out + self.T[:, i][-1])))
                        tmp[j, :] = 1 - sig2
                    else:
                        sig1 = 1 / (1 + torch.exp(-(out + self.T[:, i][j - 1])))
                        sig2 = 1 / (1 + torch.exp(-(out + self.T[:, i][j])))
                        tmp[j, :] = sig2 - sig1
                pred = torch.argmax(tmp, dim=0) + 1
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

        if config.using_mlp: # added in 9th edition
            self.mlp_model.train()
        if return_pred:
            return [mze, mae, Y_pred]
        else:
            return [mze, mae]