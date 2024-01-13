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
import pickle

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2=0, hidden_dim_3=0):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim_1)
        # self.fc1.apply(utils.weights_init_identity)
        # self.dropout = torch.nn.Dropout(p=0.2)
        if hidden_dim_2!=0:
            self.hidden_dim_2 = hidden_dim_2
            self.fc2 = torch.nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
            if hidden_dim_3!=0:
                self.hidden_dim_3 = hidden_dim_3
                self.fc3 = torch.nn.Linear(self.hidden_dim_2, self.hidden_dim_3)
            else:
                self.hidden_dim_3 = None
        else:
            self.hidden_dim_2 = None
            self.hidden_dim_3 = None
    def forward(self, x):
        if self.hidden_dim_2 != None and self.hidden_dim_3 == None: # 2 layers
            if config.activation_function == 'relu':
                X_hidden = F.relu(self.fc2(self.fc1(x)))
            elif config.activation_function == 'selu':
                X_hidden = F.selu(self.fc2(self.fc1(x)))
        elif self.hidden_dim_2 != None  and self.hidden_dim_3 != None: # 3 layers
            if config.activation_function == 'relu':
                X_hidden = F.relu(self.fc3(self.fc2(self.fc1(x))))
            elif config.activation_function == 'selu':
                X_hidden = F.selu(self.fc3(self.fc2(self.fc1(x))))
        else: # 1 layer
            if config.activation_function == 'relu':
                # X_hidden = self.dropout(F.selu(self.fc1(x))) # relu
                X_hidden = torch.relu(self.fc1(x))  # relu
                # X_hidden = self.fc1(x)
            elif config.activation_function == 'selu':
                X_hidden = torch.selu(self.fc1(x))  # selu

        return X_hidden


class Model:
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2=0, hidden_dim_3=0, logger = None):
        if logger==None:
            self.no_init_logger = True
            self.logger = utils.set_loggger(__name__, config.param_file)
        else:
            self.no_init_logger = False
            self.logger = logger
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.hidden_dim_3 = hidden_dim_3
        if config.using_mlp==True:
            self.mlp_model = MLP(self.input_dim, self.hidden_dim_1, self.hidden_dim_2, self.hidden_dim_3)
            # self.mlp_model.apply(utils.weights_init_uniform)
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
        self.min_mze = None


    def fit(self, X_train,Y_train,adj_data,lr,alpha, beta, adi_range, adi_split, fresh_start=True, pred_too=True):

        adi_list = []
        adi_time_list = []
        max_iter_list = []
        r_list = []
        s_list = []
        rho_list = []
        mze_list = []
        mae_list = []

        values = Y_train.unique()

        k = values.shape[0]
        TASK_num = Y_train.shape[0]
        TRAIN_size = Y_train.shape[1]
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
            # initialize learning rate lambda
            lambda_w = lr  # lr
            lambda_t = lr  # lr
            R = torch.zeros((TASK_num, TASK_num))
            for i in range(TASK_num):
                nb = sum(adj_data[:, i] == 1) # bug fixed coverting 1 to i
                if nb > 0:
                    R[:, i] = -torch.from_numpy(adj_data[:, i] / nb) # bug fixed coverting 1 to i
                    R[i, i] = 1
            L = torch.matmul(R, np.transpose(R))
            E = torch.eye((TASK_num))
            # admm param
            if fresh_start:
                # initialize parameters for training data for each task(size= d*TASK_num)
                if config.w_from_matlab:
                    W_mat = scipy.io.loadmat(config.train_weight_location)
                    self.W = torch.tensor(W_mat['W']).float()
                else:
                    self.W = torch.nn.Parameter(
                        torch.empty(d, TASK_num).normal_(mean=0, std=0.1))  # default: 0.1 # compare this with matlab
                self.U = self.W
                self.Y1 = torch.zeros(self.W.shape)
                # initialize threshold size : (k-1)*TASK_num
                pi = torch.nn.Parameter(torch.ones(k - 1, 1) * 1 / k)
                # assign each class with same probability
                cum_pi = torch.cumsum(pi, dim=0)
                T = torch.log(cum_pi / (1 - cum_pi))
                self.T = T.repeat(1, TASK_num)
                self.Y2 = torch.zeros(self.T.shape)
                self.V = self.T
                self.Y3 = torch.zeros(k - 2, self.T.shape[1])
                self.rho = config.rho
                self.f1inv = torch.linalg.inv(beta * L + 2 * self.rho * E)
                self.f2inv = torch.linalg.inv(beta * L + self.rho * E)

        for adi in range(adi_range):
            # with open(config.param_file, 'a') as f:
            #     f.write('MLP Weight Before ADI {}\n'.format(adi))
            #     np.savetxt(f,self.mlp_model.fc1.weight.cpu().detach().numpy()[0,0:10])
            #     f.write('\n----------------------------\n')
            #     f.write('ADMM Weight Before ADI {}\n'.format(adi))
            #     np.savetxt(f, self.W.cpu().detach().numpy()[0,0:10])
            #     f.write('\n----------------------------\n')
            # f.close()
            with torch.no_grad():
                adi_tic = time.perf_counter()
                L_old = float('inf') # Previous loss
                I = torch.zeros(config.max_iteration)
                L_train = torch.zeros(config.max_iteration)

            # Update self.W,self.T via gradient decent
            for t in range(config.max_iteration):
                print("ADI: {}, Iteration: {}".format(adi+adi_split, t), end='\r')
                # cross tasks overall loss
                loss = 0
                with torch.no_grad():
                    GW = torch.zeros((d, TASK_num))
                    GT = torch.zeros((k - 1, TASK_num))
                if config.using_mlp:
                    self.optimizer.zero_grad() # optimizer
                for i in range(TASK_num):
                    # mlp optimizer
                    for j in range(TRAIN_size):
                        # calculate what is the model predict is
                        if config.using_mlp:
                            X_hidden = self.mlp_model(X_train[i,j,:])
                        else:
                            X_hidden = X_train[i,j,:] # Alert: It will not match with matlab due to random_suffle in data processing
                        if Y_train[i,j]==k:
                            sig1=1
                            sig2= 1 / (1 + torch.exp(-(self.T[int(Y_train[i,j])-2,i] - torch.matmul(X_hidden,self.W[:,i]))))
                            dif=SMALL # default: SMALL
                        elif Y_train[i,j]==1:
                            sig1= 1 / (1 + torch.exp(-(self.T[int(Y_train[i,j])-1,i] - torch.matmul(X_hidden,self.W[:,i]))))
                            sig2=0
                            dif=SMALL
                        else:
                            sig1= 1 / (1 + torch.exp(-(self.T[int(Y_train[i,j])-1,i] - torch.matmul(X_hidden,self.W[:,i]))))
                            sig2= 1 / (1 + torch.exp(-(self.T[int(Y_train[i,j])-2,i] - torch.matmul(X_hidden,self.W[:,i]))))
                            dif= (self.T[int(Y_train[i,j])-2,i]-self.T[int(Y_train[i,j])-1,i])

                        # loss
                        l_cum=-torch.log(sig1-sig2)

                        if(l_cum==float('inf')):
                            loss_str = "error: inf loss, adi {}, t {}, task {}, sample {}, Y_value {}, l_cum {}".format(adi+adi_split, t, i, j, Y_train[i,j], l_cum)
                            print(loss_str)
                            # Logs
                            self.logger.info(loss_str)
                        loss=loss+l_cum

                        # we want to minimize l_cum w.r.t self.W and self.T
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
                    GW = 1 / TRAIN_size * 1 / TASK_num * GW + self.Y1 + self.rho * (self.W - self.U)
                    GT = 1 / TRAIN_size * 1 / TASK_num * GT + self.Y2 + self.rho * (self.T - self.V)
                    # update parameters
                    self.W = self.W - lambda_w * GW
                    self.T = self.T - lambda_t * GT
                    I[t] = t
                    L_train[t] = 1 / TRAIN_size * 1 / TASK_num * loss + torch.sum(torch.sum(torch.matmul(self.Y1, torch.transpose((self.W - self.U), 0, 1)))) + (self.rho / 2.) * torch.sum(
                        torch.sum(torch.square((self.W - self.U)))) + torch.sum(torch.sum(torch.matmul(self.Y2,  torch.transpose((self.T - self.V), 0, 1)))) + (self.rho / 2) * torch.sum(
                        torch.sum(torch.square(self.T - self.V)))
                    if config.break_on:
                        if L_old-L_train[t]<config.tol_L:
                            print("L_break")
                            break
                    L_old = L_train[t]


            # Update self.U
            # print('Update self.U')
            U_old = self.U
            self.U = self.W + 1 / self.rho * self.Y1
            pg.l21(self.U, reg=alpha / self.rho) # Alert: If alpha =0 then there is no difference between Matlab (prox_l21) and Python


            # Update self.V
            V_old = self.V
            self.V[0, :] = 1 / self.rho * self.Y2[0, :] + self.T[0, :] # bug fixed converting 1 to 0

            for i in range(1, k - 1):
                f1 = torch.matmul(
                    (torch.matmul(beta * self.V[i - 1, :], L) + self.Y2[i, :] + self.rho * (self.T[i, :] + self.V[i - 1, :]) + self.Y3[i - 1, :]), self.f1inv)
                f2 = torch.matmul((torch.matmul(beta * self.V[i - 1, :], L) + self.Y2[i, :] + self.rho * self.T[i, :] + self.Y3[i - 1, :]), self.f2inv)
                self.V[i, f1 <= self.V[i - 1, :]] = f1[f1 <= self.V[i - 1, :]]
                self.V[i, f2 > self.V[i - 1, :]] = f2[f2 > self.V[i - 1, :]]

            # Update Y
            self.Y1 = self.Y1 + self.rho * (self.W - self.U)
            self.Y2 = self.Y2 + self.rho * (self.T - self.V) # Alert: self.T, self.V is matching with matlab. But somehow, self.Y2 is not. May be some granular level differences.
            for i in range(1, k - 1):
                self.Y3[i - 1, :] = torch.maximum(self.Y3[i - 1, :] + self.rho * (self.V[i - 1, :] - self.V[i, :]), torch.zeros(TASK_num))

            # Compute ADMM residuals and update rho correspondingly
            s = self.rho * torch.sqrt(torch.sum(torch.sum(torch.square(U_old - self.U)))) + self.rho * torch.sqrt(
                torch.sum(torch.sum(torch.square(V_old - self.V)))) # bug warning: 1/10 smaller than matlab output
            r = torch.sqrt(torch.sum(torch.sum(torch.square(self.U - self.W)))) + torch.sqrt(torch.sum(torch.sum(torch.square(self.V - self.T))))

            if (r > config.tol_r_s * s):
                self.rho = 2 * self.rho
                #   update inverse matrix for computation of self.V-update when rho changes
                self.f1inv = torch.linalg.inv(beta * L + 2 * self.rho * E)
                self.f2inv = torch.linalg.inv(beta * L + self.rho * E)
            else:
                if (config.tol_r_s  * r < s):
                    self.rho = self.rho / 2
                    #       update inverse matrix for computation of self.V-update when rho changes
                    self.f1inv = torch.linalg.inv(beta * L + 2 * self.rho * E)
                    self.f2inv = torch.linalg.inv(beta * L + self.rho * E)


            adi_toc = time.perf_counter()
            time_taken = adi_toc-adi_tic
            adi_list.append(adi)
            adi_time_list.append(time_taken)
            max_iter_list.append(t)
            r_list.append(r.item())
            s_list.append(s.item())
            rho_list.append(self.rho)
            if pred_too:
                [mze, mae] = self.predict(X_train, Y_train)
                mze_list.append(mze)
                mae_list.append(mae)
                logger_str = 'ADMM iteration: {}, Max iteration: {}, r-residual: {:.6E}, s-residual: {:.6E}, ' \
                             'rho: {}, mze: {}, mae: {}, time taken {}'.format(adi+1+adi_split, t+1,
                                                                               Decimal(r.item()), Decimal(s.item()),
                                                                               self.rho, mze, mae, time_taken)
                if self.min_mze==None:
                    self.min_mze = mze
                elif mze<self.min_mze:
                    self.min_mze = mze
                    self.saveModel()
            else:
                logger_str = 'ADMM iteration: {}, Max iteration: {}, r-residual: {}, s-residual: {}, ' \
                             'rho: {}, time taken {}'.format(adi+1+adi_split, t+1, r, s, self.rho, time_taken)
            print(logger_str)
            # Logs
            self.logger.info(logger_str)


            # Check for stopping criteria
            if config.break_on:
                if (math.isnan(r) and math.isnan(s)):
                    print("R_S_break due to NaN")
                    # print(X_hidden, self.V, self.U)
                    break
                if (r<config.tol_r and s<config.tol_s):
                    print("R_S_break")
                    break


        # Saving Results
        # dictionary of lists
        if pred_too == True:
            result_dict = {'iteration': adi_list, 'max_iter': max_iter_list, 'r-residual': r_list, 's_residual': s_list, 'rho': rho_list, 'mze': mze_list, 'mae': mae_list, 'time': adi_time_list}
        else:
            result_dict = {'iteration': adi_list, 'max_iter': max_iter_list, 'r-residual': r_list, 's_residual': s_list,
                           'rho': rho_list, 'time': adi_time_list}

        result_df = pd.DataFrame(result_dict)

        # saving the dataframe
        result_df.to_csv(config.training_progress_log, index=False)

        logger_min = "Min MZE {}".format(self.min_mze)
        self.logger.info(logger_min)
        self.logger.disabled = self.no_init_logger
        if config.using_mlp:
            return [self.T, self.W, self.mlp_model.fc1.weight.data, self.mlp_model.fc1.bias.data]
        else:
            return [self.T, self.W, None, None]

    def saveModel(self):
        model_dump_file = open(config.minModel_file, 'wb')
        pickle.dump(self, model_dump_file)

    def predict(self, X_test, Y_test, return_pred = False):
        self.W = -self.W
        if config.using_mlp:
            self.mlp_model.eval()
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
        self.W = -self.W  # this line is the crucial bug which was missing in this version
        if config.using_mlp:
            self.mlp_model.train()
        if return_pred:
            return [mze, mae, Y_pred]
        else:
            return [mze, mae]