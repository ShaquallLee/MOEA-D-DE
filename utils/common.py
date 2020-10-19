#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: common.py
# @time: 2020/10/18 0018 16:27
# @desc:
import random
import math

def get_random_list(size):
    '''
    h获取长度为size随机列表
    :param size:
    :return:
    '''
    perm = [x for x in range(size)]
    random.shuffle(perm)
    return perm

def fitness_function(y, namda, idealpoint):
    '''
    使用切比雪夫分布函数计算fitness
    :param y:
    :param namda:
    :param idealpoint:
    :return:
    '''
    max_fun = -1.0e+30
    for n in range(len(y)):
        diff = math.fabs(y[n]-idealpoint[n])
        if namda[n] == 0:
            feval = 0.0001*diff
        else:
            feval = diff*namda[n]
        if feval > max_fun:
            max_fun=feval
    fitness = max_fun

    return fitness