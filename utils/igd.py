#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: igd.py
# @time: 2020/10/8 0008 18:25
# @desc: 计算多目标算法的IGD指数

import numpy as np

def get_igd(pf, points):
    '''
    计算IGD
    :param pf: 真实前沿点
    :param points: 计算所得前沿点
    :return:
    '''
    igd_sum = 0.0
    for p in pf:
        igd_min = float('inf')
        for pi in points:
            d = sum(list((np.array(p)-np.array(pi))**2))
            if d < igd_min:
                igd_min = d
        igd_sum += igd_min
    return float(igd_sum)/len(pf)





if __name__ == '__main__':
    pass
