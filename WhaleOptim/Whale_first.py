# 鲸鱼优化算法
# By: Liangyong Qi, from Xidian University  2019/05/05
# ---------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import copy
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import datasets

# ---------------------------------------------------------------------------------------------------
# 1. 定义鲸鱼优化算法的类
class WhaleOptimizationAlgorithm:

    # 1.1 初始化
    def __init__(self, func, n_dim,
                 size_pop=50, max_iter=300,
                 A=2, a=1, b=1, A_strategy=1,
                 alpha=0.25, betamin=0.20, gamma=1,
                 solution_final=None, history=False):
        self.func = func  # 待优化的函数
        self.n_dim = n_dim  # 待优化的变量个数
        self.size_pop = size_pop  # 鲸鱼群体的数量
        self.max_iter = max_iter  # 迭代次数
        self.A = A  # 振幅因子
        self.a = a  # parameter in Eq. (3.3)
        self.b = b  # parameter in Eq. (3.4)
        self.alpha = alpha  # parameter in Eq. (3.5)
        self.betamin = betamin  # mininum value of beta
        self.gamma = gamma  # parameter in Eq. (3.5)
        self.A_strategy = A_strategy  # decrease of A w.r.t. iteration number

        self.solution_final = solution_final  # final solution
        self.best_fitness = float('inf')  # best fitness
        self.best_position = None  # best position

        self.whale_position_history = []  # position history
        self.whale_fitness_history = []  # fitness history
        self.history = history  # 是否记录历史信息

    # 1.2 优化计算
    def optimize(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        # 1.2.1 初始化鲸鱼群体
        whale_position = np.random.uniform(low=-100, high=100, size=(self.size_pop, self.n_dim))  # 随机初始化鲸鱼群体
        # 1.2.2 迭代寻优
        for iter_num in range(self.max_iter):