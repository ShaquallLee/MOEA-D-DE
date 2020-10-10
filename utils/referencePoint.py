#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file:
# @time: 2020/10/9 0009 17:52
# @desc: 根据Pareto前沿计算参考点
import numpy as np

def get_referencepoint(pf):
    '''
    根据Pareto front 计算参考点
    :param pf:
    :return:
    '''
    pf = np.array(pf)
    reference_point = []
    for i in range(len(pf[0])):
        reference_point.append(max(pf[:, i]))
    return list(np.array(reference_point)*1.1)

