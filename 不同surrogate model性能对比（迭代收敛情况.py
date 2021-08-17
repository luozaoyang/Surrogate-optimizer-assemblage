# -*- coding: utf-8 -*-

"""
Global vertical illuminance optimizer 
using offline training model combined with online bayesian optimizer
Credited by Zhaoyang LUO
"""
import warnings
from time import time
import numpy as np
import os
import joblib
from skopt import gp_minimize, forest_minimize, dummy_minimize, gbrt_minimize
from skopt.plots import plot_convergence
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import rbfopt
os.environ['OMP_NUM_THREADS'] = '1'


warnings.filterwarnings('ignore')

# name=['select_angle ','DNI','DHI','GH_illu','solar_altitude','solar_azimuth','x_array','y_column','grid_distance']
f=open(r'C:\Users\Administrator\Desktop\testfile.csv')
df = pd.read_csv(f,header=None)

i=2

# select_angle = eval(df[1][1]) #先列后行，column first， array second,pandas弄出来的为字符串，不知为啥，操蛋！！
select_angle = np.arange(0,90,10).tolist()
DNI = int(df[1][i])
DHI = int(df[2][i])
GH_illu = int(df[3][i])
solar_altitude = int(df[4][i])
solar_azimuth = int(df[5][i])
x_array = int(df[6][i])
y_column = int(df[7][i])
grid_distance = eval(df[8][i])#先列后行，column first， array second,pandas弄出来的为字符串，不知为啥，操蛋！！

disk_name='f'
iteration_times= 10
saved_name= 'Optimizer Comparison'

# print(DNI,DHI,GH_illu,solar_altitude,solar_azimuth,x_array,y_column )
# print(select_angle)
# print(grid_distance)

def predict_function(blind_angle):
    matrix_list1=[]
    matrix_list2=[]
    matrix_temp_list =[]
    #1.compile data along with weather info into matrix for calculation

    for l in range(0,len(grid_distance)):
        matrix_temp_list.append([grid_distance[l]])
    for i in range(0,6):
        for k in range(0,int(x_array)):
            for j in range(0,int(y_column)):
                element1=[i,blind_angle[i],int(DNI),int(DHI),int(GH_illu),int(solar_altitude),int(solar_azimuth),int(k),int(j)]
                element2=[i,-90,int(DNI),int(DHI),int(GH_illu),int(solar_altitude),int(solar_azimuth),int(k),int(j)]
                matrix_list1.append(element1)
                matrix_list2.append(element2)
    cur_matrix = np.column_stack((np.mat(matrix_list1),np.mat(matrix_temp_list)))
    clo_matrix = np.column_stack((np.mat(matrix_list2),np.mat(matrix_temp_list)))

    #2.load z-score info from train data for preprocessing data
    #transfer = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\transfer.pkl'%disk_name)
    transfer = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\transfer.pkl'%disk_name)
    timely_input = transfer.transform(cur_matrix)
    close_input = transfer.transform(clo_matrix)

    #3. load pre-trained model for prediction
    #MLP_model = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\MLP model.pkl'%disk_name)
    MLP_model = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\MLP model.pkl'%disk_name)
    timely_pre = MLP_model.predict(timely_input).reshape(6,x_array*y_column)
    close_pre = MLP_model.predict(close_input).reshape(6,x_array*y_column)

    #4. calculate the combined result by each blind
    final_result = sum(timely_pre)-5*(sum(close_pre)/6)
    Predict_MLP_list = final_result.astype(np.int32).tolist()

    #5. shift the value into positive (normal) one
    Predict_MLP_output =[]
    for k in Predict_MLP_list:
        Predict_MLP_output.append(abs(k))

    #6. setting opt objective using ev boundary
    new_result=[]
    for i in Predict_MLP_output:
        if i >= int(2670):
            new_result.append(np.random.randint(-10,0))
    #note: there should not be zero placed here, it seems that the sequential opt can't
    # accept constant number like continuely zero
        else:
            new_result.append(int(i))
    selected_value = min(new_result)
    opt_goal = int(-1)*selected_value

    return opt_goal
def predict_function1(x):
    list = x.tolist()
    blind_angle = [i * 10 for i in list]

    matrix_list1=[]
    matrix_list2=[]
    matrix_temp_list =[]
    #1.compile data along with weather info into matrix for calculation

    for l in range(0,len(grid_distance)):
        matrix_temp_list.append([grid_distance[l]])
    for i in range(0,6):
        for k in range(0,int(x_array)):
            for j in range(0,int(y_column)):
                element1=[i,blind_angle[i],int(DNI),int(DHI),int(GH_illu),int(solar_altitude),int(solar_azimuth),int(k),int(j)]
                element2=[i,-90,int(DNI),int(DHI),int(GH_illu),int(solar_altitude),int(solar_azimuth),int(k),int(j)]
                matrix_list1.append(element1)
                matrix_list2.append(element2)
    cur_matrix = np.column_stack((np.mat(matrix_list1),np.mat(matrix_temp_list)))
    clo_matrix = np.column_stack((np.mat(matrix_list2),np.mat(matrix_temp_list)))

    #2.load z-score info from train data for preprocessing data
    #transfer = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\transfer.pkl'%disk_name)
    transfer = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\transfer.pkl'%disk_name)
    timely_input = transfer.transform(cur_matrix)
    close_input = transfer.transform(clo_matrix)

    #3. load pre-trained model for prediction
    #MLP_model = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\MLP model.pkl'%disk_name)
    MLP_model = joblib.load(r'%s:\我的坚果云\papers\sci paper\2ed paper\data\neural network\trained model(MLP)\MLP model.pkl'%disk_name)
    timely_pre = MLP_model.predict(timely_input).reshape(6,x_array*y_column)
    close_pre = MLP_model.predict(close_input).reshape(6,x_array*y_column)

    #4. calculate the combined result by each blind
    final_result = sum(timely_pre)-5*(sum(close_pre)/6)
    Predict_MLP_list = final_result.astype(np.int32).tolist()

    #5. shift the value into positive (normal) one
    Predict_MLP_output =[]
    for k in Predict_MLP_list:
        Predict_MLP_output.append(abs(k))

    #6. setting opt objective using ev boundary
    new_result=[]
    for i in Predict_MLP_output:
        if i >= int(2670):
            new_result.append(np.random.randint(-10,0))
    #note: there should not be zero placed here, it seems that the sequential opt can't
    # accept constant number like continuely zero
        else:
            new_result.append(int(i))
    selected_value = min(new_result)
    opt_goal = int(-1)*selected_value

    return opt_goal


def rbf(predict_function):
    settings = rbfopt.RbfoptSettings(rbf='gaussian',algorithm='MSRSM',max_evaluations=iteration_times,minlp_solver_path=r'D:\rbfopt\bonmin\bonmin', nlp_solver_path=r'D:\rbfopt\ipopt\ipopt',num_cpus=1)
    bb = rbfopt.RbfoptUserBlackBox(6, np.array([min(select_angle)/10] * 6), np.array([max(select_angle)/10] * 6),
                                   np.array(['I']*6), predict_function)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    func_vals, x, itercount, evalcount, fast_evalcount = alg.optimize()  # best-objective(min),the assigned variable(hen get the best-objective), iternum
    value_list = [abs(i) for i in alg.all_node_val.tolist()]  # enumerate and get intermediate value at every x
    x_iters = np.arange(len(value_list)).tolist()
    # best_variable=[i * 10 for i in x.tolist()]
    return {'x_iters':x_iters,'func_vals':func_vals}

# a=alg.all_node_val #所有goal的取值节点
# b=alg.all_node_pos #所有x的取值节点
# print(a,b)
def run(minimizer, n_iter=5):
    func = partial(predict_function)#7. launch the opt process with skopt(well, not so excellent)
    space = [select_angle]*6
    return [minimizer(func, space, n_calls=iteration_times, random_state=n)for n in range(n_iter)]

def run1(n_iter=5):
    box=[]
    for s in range(n_iter):
        box.append(rbf(predict_function1))
    return box


# Random search
dummy_res = run(dummy_minimize)

# Gaussian processes
gp_res = run(partial(gp_minimize, n_initial_points=10,initial_point_generator='lhs'))
# 对skopt的result，OptimizeResult，用字典get() 函数返回指定键的值。
# suggested_blind_angles = gp_res.get('x')
# maximal_min_illu = gp_res.get('fun') * (-1)
# goal_total = [abs(i) for i in gp_res.get('func_vals').tolist()]
# iteration_total = np.arange(0, iteration_times, 1).tolist()
print(gp_res)
# Random forest
rf_res = run(partial(forest_minimize, base_estimator="RF"))

# Extra trees
et_res = run(partial(forest_minimize, base_estimator="ET"))

# gradient boosting的贝叶斯优化
gbrt_res = run(partial(gbrt_minimize, base_estimator="GBRT"))

# rbf interpolation的优化法
rbf_res = run1()
print(rbf_res)

# plot = plot_convergence(
#                         ("Random Search", dummy_res),
#                         ("Gaussian Process", gp_res),
#                         ("Random Forest", rf_res),
#                         ("ExtraTrees", et_res),
#                         ("Gradient boosting",gbrt_res),
#                         ("radial basis function opt",rbf_res),markersize=3,min_or_max='max')
#
# plot.legend(loc="best", prop={'size': 10}, numpoints=0.01)
# plt.savefig(r'%s:\我的坚果云\papers\博士论文\实验\第四章实验\online optimizer comparison result\%s.png'%(disk_name,saved_name),dpi=300)
# plt.show()



