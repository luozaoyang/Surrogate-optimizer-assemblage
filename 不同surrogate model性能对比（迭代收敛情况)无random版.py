# -*- coding: utf-8 -*-

"""
Global vertical illuminance optimizer 
using offline training model combined with online bayesian optimizer
Credited by Zhaoyang LUO
"""
import warnings
import time
import numpy as np
import os
import joblib
from skopt import gp_minimize, forest_minimize, dummy_minimize, gbrt_minimize
from skopt.plots import plot_convergence
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import rbfopt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator, FuncFormatter  # noqa: E402

os.environ['OMP_NUM_THREADS'] = '1'


warnings.filterwarnings('ignore')


def predict_function(x):
    blind_angle = [i * 10 for i in x]

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
            # new_result.append(0)
    #note: there should not be zero placed here, it seems that the sequential opt can't
    # accept constant number like continuely zero
        else:
            new_result.append(int(i))
    selected_value = min(new_result)
    opt_goal = int(-1)*selected_value

    return opt_goal

# Radial Basis Function Interpolation
def rbf(rand_seed=1):
    settings = rbfopt.RbfoptSettings(rbf='gaussian',
                                     algorithm='MSRSM',
                                     max_evaluations=iteration_times,
                                     minlp_solver_path=r'D:\rbfopt\bonmin\bonmin',
                                     nlp_solver_path=r'D:\rbfopt\ipopt\ipopt',
                                     do_infstep=False,
                                     num_global_searches=5,
                                     num_cpus=1,
                                     rand_seed=rand_seed
                                     )
    bb = rbfopt.RbfoptUserBlackBox(6, np.array([min(select_angle)/10] * 6), np.array([max(select_angle)/10] * 6),
                                   np.array(['I']*6), predict_function)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    func_vals, x, itercount, evalcount, fast_evalcount = alg.optimize()
    # best-objective(min),the assigned variable,Total number of iterations,Total number of function evaluations,
    goal_total = [abs(i) for i in alg.all_node_val.tolist()]  # enumerate and get intermediate value at every x
    # iteration_total = np.arange(1, evalcount+1, 1).tolist()
    # final_maximal_min_illu = func_vals
    best_goal_total = [max(goal_total[:i]) for i in range(1, evalcount + 1)]
    # suggested_blind_angles=[i * 10 for i in x.tolist()]
    timecost = alg.elapsed_time
    timecost_first = (goal_total.index(max(goal_total))+1)/len(goal_total)*timecost
    result = (best_goal_total, timecost, timecost_first,goal_total)
    return result # iteration_total,final_maximal_min_illu,suggested_blind_angles

# Random search
def random_search_opt(random_state=1):
    variable_temp= [i/10 for i in list(select_angle)]
    space = [variable_temp]*6
    start = time.time()
    dummy_res=dummy_minimize(predict_function,space,
                             n_calls=iteration_times,
                             initial_point_generator="lhs",
                             random_state=random_state)
    # suggested_blind_angles = [i * 10 for i in dummy_res.get('x')]
    # final_maximal_min_illu = dummy_res.get('fun') * (-1)
    # iteration_total = np.arange(1, iteration_times+1, 1).tolist()
    goal_total = [abs(i) for i in dummy_res.get('func_vals').tolist()]
    best_goal_total = [max(goal_total[:i]) for i in range(1, iteration_times + 1)]
    timecost = time.time()-start
    timecost_first = (goal_total.index(max(goal_total))+1)/len(goal_total)*timecost
    result = (best_goal_total, timecost, timecost_first,goal_total)
    return result # iteration_total,final_maximal_min_illu,suggested_blind_angles

# Bayesian optimization
def bayesian_opt(random_state=1):
    variable_temp = [i/10 for i in list(select_angle)]
    space = [variable_temp]*6
    start = time.time()
    gaussian_res = gp_minimize(predict_function,space,
                               n_calls=iteration_times,
                               n_initial_points=10,
                               initial_point_generator='random',
                               random_state=random_state)

    # suggested_blind_angles = [i * 10 for i in gaussian_res.get('x')]
    # final_maximal_min_illu = gaussian_res.get('fun') * (-1)
    # iteration_total = np.arange(1, iteration_times+1, 1).tolist()
    goal_total = [abs(i) for i in gaussian_res.get('func_vals').tolist()]
    best_goal_total = [max(goal_total[:i]) for i in range(1, iteration_times + 1)]
    timecost = time.time() - start
    timecost_first = (goal_total.index(max(goal_total))+1)/len(goal_total)*timecost
    result = (best_goal_total, timecost, timecost_first,goal_total)
    return result # iteration_total,final_maximal_min_illu,suggested_blind_angles

# Random Forest
def random_forest_opt(random_state=1):
    variable_temp = [i/10 for i in list(select_angle)]
    space = [variable_temp]*6
    start = time.time()
    rf_res = forest_minimize(predict_function,space,
                             base_estimator="RF",
                             n_calls=iteration_times,
                             n_initial_points=10,
                             initial_point_generator='lhs',
                             random_state=random_state)

    # suggested_blind_angles = [i * 10 for i in rf_res.get('x')]
    # final_maximal_min_illu = rf_res.get('fun') * (-1)
    # iteration_total = np.arange(1, iteration_times+1, 1).tolist()
    goal_total = [abs(i) for i in rf_res.get('func_vals').tolist()]
    best_goal_total = [max(goal_total[:i]) for i in range(1, iteration_times + 1)]
    timecost = time.time() - start
    timecost_first = (goal_total.index(max(goal_total))+1)/len(goal_total)*timecost
    result = (best_goal_total, timecost, timecost_first,goal_total)
    return result # iteration_total,final_maximal_min_illu,suggested_blind_angles

# Extra trees
def extra_trees_opt(random_state=1):
    variable_temp = [i/10 for i in list(select_angle)]
    space = [variable_temp]*6
    start = time.time()
    et_res = forest_minimize(predict_function,space,
                             base_estimator="ET",
                             n_calls=iteration_times,
                             n_initial_points=10,
                             initial_point_generator='lhs',
                             random_state=random_state)

    # suggested_blind_angles = [i * 10 for i in et_res.get('x')]
    # final_maximal_min_illu = et_res.get('fun') * (-1)
    # iteration_total = np.arange(1, iteration_times+1, 1).tolist()
    goal_total = [abs(i) for i in et_res.get('func_vals').tolist()]
    best_goal_total = [max(goal_total[:i]) for i in range(1, iteration_times + 1)]
    timecost = time.time() - start
    timecost_first = (goal_total.index(max(goal_total))+1)/len(goal_total)*timecost
    result = (best_goal_total, timecost, timecost_first,goal_total)
    return result # iteration_total,final_maximal_min_illu,suggested_blind_angles

# Gradient boosting的贝叶斯优化
def gbrt_res(random_state):
    variable_temp = [i/10 for i in list(select_angle)]
    space = [variable_temp]*6
    start = time.time()
    gbrt_res = gbrt_minimize(predict_function,space,
                             base_estimator="GBRT",
                             n_calls=iteration_times,
                             n_initial_points=10,
                             initial_point_generator='lhs',
                             random_state=random_state)

    # suggested_blind_angles = [i * 10 for i in gbrt_res.get('x')]
    # final_maximal_min_illu = gbrt_res.get('fun') * (-1)
    # iteration_total = np.arange(1, iteration_times+1, 1).tolist()
    goal_total = [abs(i) for i in gbrt_res.get('func_vals').tolist()]
    best_goal_total = [max(goal_total[:i]) for i in range(1, iteration_times + 1)]
    timecost = time.time() - start
    timecost_first = (goal_total.index(max(goal_total)) + 1) / len(goal_total) * timecost
    result = (best_goal_total,timecost,timecost_first,goal_total)
    return result # iteration_total,final_maximal_min_illu,suggested_blind_angles

def repeat_for_rand(minimizer, n_iter=5):
    best_res = []
    time_cost = []
    time_cost_first = []
    goal_total_List= []
    for n in range(n_iter):
        try:
            result_comibine = minimizer(random_state=n)
            best_res.append(result_comibine[0])
            time_cost.append(result_comibine[1])
            time_cost_first.append(result_comibine[2])
            goal_total_List.append(result_comibine[3])
        except:
            result_comibine = minimizer(rand_seed=n)
            best_res.append(result_comibine[0])
            time_cost.append(result_comibine[1])
            time_cost_first.append(result_comibine[2])
            goal_total_List.append(result_comibine[3])
    return best_res,time_cost,time_cost_first, goal_total_List


def visualization_value_oscillation(goal_total_list,subjects):
    fig = plt.figure(figsize=(10,10))
    for i in range(1,len(subjects)+1):
        ax = fig.add_subplot(len(subjects),1,i)
        plt.title('Optimization Procedure',fontsize=9)
        plt.grid(axis='y',linestyle='-.')
        iterations = np.arange(1, iteration_times + 1, 1).tolist()
        print(goal_total_list[i-1])
        plt.plot(iterations,np.mean(np.array(goal_total_list[i-1]),axis=0).tolist(),label='Objective Value per iteration')
        #np.mean处理的对象是矩阵，非列表
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        ax.set_title(subjects[i-1],fontsize=9)
        # ax.set_xlabel('Iteration Times',fontsize=9)
        # ax.set_ylabel('Objective Value',fontsize=9)
        plt.legend(loc='upper left',fontsize=9)
        fig.tight_layout()
    plt.savefig(r'%s:\我的坚果云\papers\博士论文\实验\第四章实验\online optimizer comparison result\%s.png'%(disk_name,'oscillation_procedure'),dpi=300)
    plt.show()


def visualization_best_value_evolve_with_iters(*args, **kwargs):
    markersize = kwargs.get("makersize", 10)
    lw = kwargs.get('lw', 3)
    plt.figure(figsize=(10, 10))

    plt.subplot(2,1,1)
    ax = plt.gca()
    ax.set_title("Convergence plot",fontsize=12)
    ax.set_xlabel("Number of calls $n$",fontsize=12)
    ax.set_ylabel("Maximum after each iteration",fontsize=12)
    #ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()
    colors = cm.viridis(np.linspace(0.01, 1, len(args)))
    iterations = np.arange(1, iteration_times+1, 1).tolist()
    for results, color in zip(args, colors):
        name, maxs, time_cost = results
        for m in maxs:
            ax.plot(iterations, m, c=color, alpha=0.2)
        plt.plot(iterations, np.mean(np.array(maxs), axis=0).tolist(), c=color, marker=".", markersize=markersize, lw=lw, label=name)
        #注意，这里iteration不能包含0，否则自动忽略
        ax.legend(loc="lower right",fontsize=9,prop={'size': 10}, numpoints=0.1)

    plt.subplot(2, 1, 2)
    ax = plt.gca()
    ax.set_title("Convergence plot",fontsize=12)
    ax.set_xlabel("Time length",fontsize=12)
    ax.set_ylabel("Maximum after each iteration",fontsize=12)
    #ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()
    colors = cm.viridis(np.linspace(0.01, 1, len(args)))
    iterations = np.arange(1, iteration_times+1, 1)
    for results, color in zip(args, colors):
        name, maxs, time_cost = results
        # print(iterations)
        # print(time_cost)
        for m in range(0,len(maxs)):
            ax.plot(iterations*time_cost[m]/iteration_times, maxs[m], c=color, alpha=0.2)
        plt.plot(iterations*np.mean(np.array(time_cost)/iteration_times, axis=0), np.mean(np.array(maxs), axis=0), c=color, marker=".", markersize=markersize, lw=lw, label=name)
        #注意，这里iteration不能包含0，否则自动忽略
        ax.legend(loc="lower right",fontsize=9,prop={'size': 10}, numpoints=0.1)

    plt.savefig(r'%s:\我的坚果云\papers\博士论文\实验\第四章实验\online optimizer comparison result\%s.png'%(disk_name,'best_value_evolve'),dpi=300)
    plt.show()


def visualization_time_cost(time_list,label_list):
    plt.figure(figsize=[15, 5])
    width = 0.3
    plt.bar(label_list, time_list, width, label='Time', color='orange')
    for a, b in zip(label_list, time_list):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    plt.legend(fontsize=12, loc='upper right')
    plt.title('Time cost comparison after set iteration', fontsize=12)
    plt.xlabel('Optimizer category', fontsize=12)
    plt.ylabel('Time spend(s)', fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig(r'%s:\我的坚果云\papers\博士论文\实验\第四章实验\online optimizer comparison result\%s.png'%(disk_name,'time_cost_total'),dpi=300)
    plt.show()

def visualization_time_cost_at_max(time_list,label_list):
    plt.figure(figsize=[15, 5])
    width = 0.3
    plt.bar(label_list, time_list, width, label='Time', color='black')
    for a, b in zip(label_list, time_list):
        plt.text(a, b + 0.01, '%.2f' % b, ha='center', va='bottom', fontsize=9)
    plt.legend(fontsize=12, loc='upper right')
    plt.title('Time cost comparison when first get the maximum value', fontsize=12)
    plt.xlabel('Optimizer category', fontsize=12)
    plt.ylabel('Time spend(s)', fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig(r'%s:\我的坚果云\papers\博士论文\实验\第四章实验\online optimizer comparison result\%s.png'%(disk_name,'time_cost_at_max'),dpi=300)
    plt.show()

if __name__ == '__main__':
    # name=['select_angle ','DNI','DHI','GH_illu','solar_altitude','solar_azimuth','x_array','y_column','grid_distance']
    f = open(r'C:\Users\Administrator\Desktop\transicionfile.csv') #先开启四章节的folder下的data prediction for multi-blinds 然后完成临时feature数据csv的生成
    df = pd.read_csv(f, header=None)
    i = 8 #选择时刻数，8代表14点
    # select_angle = eval(df[1][1]) #先列后行，column first， array second,pandas弄出来的为字符串，不知为啥，操蛋！！
    select_angle = np.arange(0, 90, 10)
    DNI = int(df[1][i])
    DHI = int(df[2][i])
    GH_illu = int(df[3][i])
    solar_altitude = int(df[4][i])
    solar_azimuth = int(df[5][i])
    x_array = int(df[6][i])
    y_column = int(df[7][i])
    grid_distance = eval(df[8][i])  # 先列后行，column first， array second,pandas弄出来的为字符串，不知为啥，操蛋！！

    disk_name = 'f'
    iteration_times = 150
    saved_name = 'Optimizer Comparison'
    label_list=["RBF Interpolation Optimization","Random Search","Bayesian Optimization","ExtraTrees","Gradient Boosting Trees"] #"Random Forest",放第四个
    # print(DNI,DHI,GH_illu,solar_altitude,solar_azimuth,x_array,y_column )
    # print(select_angle)
    # print(grid_distance)
    # iteration_total1,best_goal_total1,final_maximal_min_illu1,suggested_blind_angles1 = rbf()
    # iteration_total2,best_goal_total2,final_maximal_min_illu2,suggested_blind_angles2 = random_search_opt()
    # iteration_total3,best_goal_total3,final_maximal_min_illu3,suggested_blind_angles3 = gaussian_opt()
    # iteration_total4,best_goal_total4,final_maximal_min_illu4,suggested_blind_angles4 = random_forest_opt()
    # iteration_total5,best_goal_total5,final_maximal_min_illu5,suggested_blind_angles5 = extra_trees_opt()
    # iteration_total6,best_goal_total6,final_maximal_min_illu6,suggested_blind_angles6 = gbrt_res()

    best_res1, time_cost1, time_cost_first1, goal_total1 = repeat_for_rand(rbf)
    best_res2, time_cost2, time_cost_first2, goal_total2 = repeat_for_rand(random_search_opt)
    best_res3, time_cost3, time_cost_first3, goal_total3 = repeat_for_rand(bayesian_opt)
    best_res5, time_cost5, time_cost_first5, goal_total5 = repeat_for_rand(extra_trees_opt)
    best_res6, time_cost6, time_cost_first6, goal_total6 = repeat_for_rand(gbrt_res)
    #    best_res4, time_cost4, time_cost_first4, goal_total4 = repeat_for_rand(random_forest_opt)

    timecost_list = [np.sum(time_cost1,axis = 0),
                     np.sum(time_cost2,axis = 0),
                     np.sum(time_cost3,axis = 0),
                     np.sum(time_cost5,axis = 0),
                     np.sum(time_cost6,axis = 0)] #np.sum(time_cost4,axis = 0),

    timecost_at_max_list = [np.mean(time_cost_first1,axis = 0),
                            np.mean(time_cost_first2,axis = 0),
                            np.mean(time_cost_first3,axis = 0),
                            np.mean(time_cost_first5,axis = 0),
                            np.mean(time_cost_first6,axis = 0)]  #np.mean(time_cost_first4,axis = 0),
    goal_total_list = [goal_total1,goal_total2,goal_total3,goal_total5,goal_total6] #goal_total4,
    #元组可以使用下标访问

    visualization_best_value_evolve_with_iters((label_list[0], best_res1, time_cost1),
                                               (label_list[1], best_res2, time_cost2),
                                               (label_list[2], best_res3, time_cost3),
                                               (label_list[3], best_res5, time_cost5),
                                               (label_list[4], best_res6, time_cost6))   #("Random Forest ",  best_res4),
    visualization_time_cost(timecost_list, label_list)
    visualization_time_cost_at_max(timecost_at_max_list, label_list)
    visualization_value_oscillation(goal_total_list, label_list)

