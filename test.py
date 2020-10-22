#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: test.py
# @time: 2020/10/20 0020 18:59
# @desc:
from utils.benchmarks import *
from moeadde1 import MOEADDE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.igd import get_igd
from utils.referencePoint import get_referencepoint
from utils.common import extract_info,draw_scatter3D
from utils.hypervolume import HyperVolume

problems = [DTLZ1,DTLZ2,DTLZ3,DTLZ4,DTLZ5,DTLZ6,DTLZ7]
max_run = 10

def problems_test(draw):
    '''
    一系列函数问题的测试
    :return:
    '''
    for id in range(len(problems)):
        print('DTLZ{} starting……'.format(id))
        # problem_test(problem=problems[id], draw=False)
        n_run(10, problems[id], draw=draw)

def problem_test(problem, draw=True):
    '''
    单个问题测试
    :param problem:
    :param draw:
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
        plt.figure()
        plt.plot(distances[1:])
        plt.xlabel("generation")
        plt.ylabel("IGD")
        plt.savefig('./results/{}.png'.format(model.problem.name()))
        plt.show()
    return hv, igd


def n_run(n, problem, draw):
    '''
    运行n次
    :param n:
    :param problem:
    :return:
    '''
    igds = []
    hvs = []
    for i in range(n):
        hv, igd = problem_test(problem, draw)
        hvs.append(hv)
        igds.append(igd)
    print("avgIGD={},minIGD={}\navgHV={},minHV={}".format(
        sum(igds)/n, min(igds), sum(igds)/n, min(igds)
    ))

if __name__ == '__main__':
    problems_test(draw=False)

