# -*- coding: utf-8 -*-

import rbfopt
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd

def obj_funct(x):
  return x[0]*x[1] - x[2]

bb = rbfopt.RbfoptUserBlackBox(3, np.array([0] * 3), np.array([10] * 3),
                               np.array(['R', 'I', 'R']), obj_funct) # get dimension(variables num),get variable low(array), get variable higher(array), get variable type(real,integer, categorical)

# print(np.array([0] * 3))
# print(np.array([10] * 3))
# print(np.array(['R', 'I', 'R']))

settings = rbfopt.RbfoptSettings(rbf='gaussian',algorithm='MSRSM',
    max_evaluations=50,minlp_solver_path=r'D:\rbfopt\bonmin\bonmin', nlp_solver_path=r'D:\rbfopt\ipopt\ipopt',num_cpus=1)
#algorithm='Gutmann',rbf='gaussian',init_strategy='lhd_maximin',global_search_method='solver',这个是文献的方法，特别慢
alg = rbfopt.RbfoptAlgorithm(settings, bb)

#alg.set_output_stream(r'C:\Users\Administrator\Desktop\state.csv')
objval, x, itercount, evalcount, fast_evalcount= alg.optimize() # best-objective(min),the assigned variable(hen get the best-objective), iternum\
a=alg.all_node_val #所有goal的取值节点
b=alg.all_node_pos #所有x的取值节点
print(a,b)

# print(objval)
# print(itercount)
# print(x)
# print(evalcount)