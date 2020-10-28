#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: test.py
# @time: 2020/10/20 0020 18:59
# @desc: 实验测试MODE/D-DE算法及其改进
from utils.benchmarks import *
from moeadde1 import MOEADDE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.fileProcess import savePareto2Txt, readPareto4Txt, saveArray2Excel, saveRes2Excel
from utils.igd import get_igd
from utils.referencePoint import get_referencepoint
from utils.common import extract_info,draw_scatter3D, draw_igd
from utils.hypervolume import HyperVolume

problems = [DTLZ1,DTLZ2,DTLZ3,DTLZ4,DTLZ5,DTLZ6,DTLZ7]
names = ['dtlz1','dtlz2','dtlz3','dtlz4','dtlz5','dtlz6','dtlz7']
max_run = 10
pid = 3

def problems_test(draw, r2f=False):
    '''
    一系列函数问题的测试
    :return:
    '''
    results = {}
    for id in range(len(problems)):
        print('DTLZ{} starting……'.format(id))
        # problem_test(problem=problems[id], draw=False)
        igds, hvs = n_run(10, problems[id], draw=draw, r2f=r2f)
        results[names[id]] = [igds, hvs]
    return results

def problem_test(problem, draw=True, s2f=False):
    '''
    单个问题测试
    :param problem:
    :param draw:时候画图
    :param s2f:是否保存帕累托前沿到文件中
    :return:
    '''
    model = MOEADDE(problem)
    distances = model.execute()
    pops, x, y, z = extract_info(model)
    reference_point = get_referencepoint(pops)
    hv = HyperVolume(reference_point=reference_point)
    hv_score = hv.compute(model.pop)
    igd = get_igd(model.pareto_front, pops)  # 计算反世代距离IGD
    print('hyper volume is {}'.format(hv_score))
    print('inverted generational distance is {}'.format(igd))
    if draw:
        draw_scatter3D(model.problem.name(), hv_score, igd, reference_point, x, y, z)
        draw_igd(distances, model)
    if s2f:
        savePareto2Txt(model.problem.name(), pops)
    return hv_score, igd


def n_run(n, problem, draw, s2f=False):
    '''
    运行n次
    :param n:
    :param problem:
    :return:
    '''
    igds = []
    hvs = []
    for i in range(n):
        hv, igd = problem_test(problem, draw, s2f=s2f)
        hvs.append(hv)
        igds.append(igd)
    print("avgIGD={},minIGD={}\navgHV={},minHV={}".format(
        sum(igds)/n, min(igds), sum(igds)/n, min(igds)
    ))
    return igds, hvs

if __name__ == '__main__':
    # igdss, hvss = problems_test(False, r2f=True)
    # problem_test(DTLZ1,s2f=True)
    # igds, hvs = n_run(10, problems[pid], True, s2f=True)
    res = problems_test(False, r2f=False)
    saveRes2Excel("./results/excels/{}.xls".format(names[pid]), res)
