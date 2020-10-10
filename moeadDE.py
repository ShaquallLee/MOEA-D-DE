#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: moea_d.py
# @time: 2020/9/22 0022 17:43
# @desc:MOEA/D算法实现
import math

from pymop.problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
import numpy as np
import copy
import random
from individual import Individual as ind
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.hypervolume import HyperVolume
from utils.pfget import get_pflist
from utils.igd import get_igd
from utils.referencePoint import get_referencepoint

# 总共跑多少次
total_run = 1
EPS = 1.2e-7
problems = [DTLZ1,DTLZ2,DTLZ3,DTLZ4,DTLZ5,DTLZ6,DTLZ7]
problem_id = 0

class MOEAD():
    def __init__(self,pf=None, problem=DTLZ1):
        # 每次最大迭代的次数
        self.max_generation = 100
        # 每次最大计算fitness次数
        self.max_count_fitness = 200000
        self.count_fitness = 0
        # 计算邻居个数
        self.max_neighborhood_size = 20

        # 问题变量的维度
        self.n_var = 10
        # 需要解决的问题
        self.problem = problem(self.n_var)
        # 多目标问题的个数
        self.n_obj = self.problem.n_obj
        # 目标函数变量取值上下界
        self.lbound = self.problem.xl
        self.rbound = self.problem.xu

        # 问题的种群
        self.pf = pf
        if pf is None:
            self.vector_size = 18  # 权重向量大小 双目标23，三目标99
        else:
            self.vector_size = len(pf)
        self.pop = []   # 种群个体

        # ideal point 理想点
        self.ideal_point = []
        self.ideal_fitness = []

        # 边界点
        self.boundary_point = [-1, -1, -1]

    def run(self, hv=None):
        # 初始化
        self.init_vector()
        print('权重初始化成功')
        self.init_neighbor()
        print('邻居初始化成功')
        self.init_pop()
        print('种群初始化成功')
        # print('hyper volume is ', hv.compute(self.pop))

        # 进化更新
        # for gene in range(self.max_generation):
        gene = 1
        while self.count_fitness < self.max_count_fitness: #and gene < self.max_generation:
            # print('第{}代开始'.format(gene))
            for i in range(self.vector_size):
                child, _ = self.reproduction(i)
                child = self.improvememt(child)
                # update reference
                fit = self.update_reference(child)
                #update of neighborhoods,由于当前i是i最近的元素，距离为0，故也在下面函数中更新
                self.update_problem(child, fit, i)
            # print('hyper volume is ', hv.compute(self.pop))
            gene += 1
        print('结束')

    def init_vector(self):
        '''
        初始化权重限量
        :return:
        '''
        if self.pf is None:
            for i in range(self.vector_size):
                if self.n_obj == 2:
                    pop = ind()
                    pop.vector.append(i)
                    pop.vector.append(self.vector_size-i)
                    for j in range(len(self.n_obj)):
                        pop.namda.append(1.0*pop.vector[j]/self.vector_size)
                    self.pop.append(pop)
                elif self.n_obj == 3:
                    for j in range(self.vector_size):
                        if i+j <= self.vector_size:
                            pop = ind()
                            pop.vector.append(i)
                            pop.vector.append(j)
                            pop.vector.append(self.vector_size-i-j)
                            for k in range(self.n_obj):
                                pop.namda.append(1.0*pop.vector[k]/self.vector_size)
                            self.pop.append(pop)
            self.vector_size = len(self.pop)
        else:
            for i in range(len(self.pf)):
                pop = ind()
                pop.vector = self.pf[i]
                for k in range(self.n_obj):
                    pop.namda.append(1.0 * pop.vector[k] / self.vector_size)
                self.pop.append(pop)
        # print(self.vector_size)

    def init_neighbor(self):
        distances = []
        for i in range(self.vector_size):
            idis = []
            for j in range(self.vector_size):
                idis.append((math.sqrt(sum((np.array(self.pop[i].namda)-np.array(self.pop[j].namda))**2)),j))
            res = sorted(idis, key=lambda x:x[0])
            self.pop[i].neighbor = [x[1] for x in res[1:self.max_neighborhood_size+1]]

    def init_pop(self):
        '''
        初始化种群
        :return:
        '''
        self.ideal_fitness = [float('inf'),float('inf'),float('inf')]
        self.ideal_point = [None, None, None]
        for i in range(self.vector_size):
            # initial pop and fitness
            pop = self.lbound + (self.rbound-self.lbound) * np.random.rand(self.n_var)
            self.pop[i].pop_x = pop
            self.pop[i].pop_fitness = self.problem.evaluate(pop)
            self.count_fitness += 1
            # update reference point
            for j in range(self.n_obj):
                if self.pop[i].pop_fitness[j] < self.ideal_fitness[j]:
                    self.ideal_fitness[j] = self.pop[i].pop_fitness[j]
                    self.ideal_point[j] = copy.copy(self.pop[i].pop_x)

    def reproduction(self, pop_id):
        '''
        交叉
        :param pop_id:
        :return:
        '''
        neighbors = self.pop[pop_id].neighbor
        x1, x2 = random.sample(list(neighbors), 2)
        parent1 = self.pop[x1].pop_x
        parent2 = self.pop[x2].pop_x
        child1 = [float('inf') for i in range(self.n_var)]
        child2 = [float('inf') for i in range(self.n_var)]
        if random.random() <= 1.0:
            for i in range(self.n_var):
                if random.random() < 0.5:
                    if math.fabs(parent1[i]-parent2[i]) > EPS:
                        if parent1[i] < parent2[i]:
                            y1 = parent1[i]
                            y2 = parent2[i]
                        else:
                            y1 = parent2[i]
                            y2 = parent1[i]
                        yl = self.lbound[i]
                        yu = self.rbound[i]
                        rand = random.random()
                        beta = 1.0 + (2.0 * (y1-yl) / (y2-y1))
                        alpha = 2.0 - pow(beta, -(20+1.0))
                        if rand <= (1.0/alpha):
                            betaq = math.pow((rand * alpha), (1.0 / (20+1.0)))
                        else:
                            betaq = math.pow((1.0 / (2.0 - rand * alpha)), (1.0 / (20+1.0)))
                        c1 = 0.5 * ((y1+y2)-betaq * (y2-y1))
                        beta = 1.0 + (2.0 * (yu-y2) / (y2-y1))
                        alpha = 2.0 - pow(beta, -(20+1.0))
                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (20+1.0)))
                        else:
                            betaq = pow((1.0 / (2.0 - rand * alpha)), (1.0 / (20+1.0)))
                        c2 = 0.5 * ((y1+y2)+betaq * (y2-y1))
                        if c1 < yl:
                            c1=yl
                        if c2 < yl:
                            c2=yl
                        if c1 > yu:
                            c1=yu
                        if c2 > yu:
                            c2=yu
                        if random.random()<0.5:
                            child1[i] = c2
                            child2[i] = c1
                        else:
                            child1[i] = c1
                            child2[i] = c2
                    else:
                        child1[i] = parent1[i]
                        child2[i] = parent2[i]
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
        else:
            for i in range(self.n_var):
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        return child1, child2

    def improvememt(self, ind):
        '''
        做进一步的提升
        :param ind:
        :return:
        '''
        rate = 1.0/self.n_var
        for j in range(self.n_var):
            if random.random() < rate:
                y = ind[j]
                yl = self.lbound[j]
                yu = self.rbound[j]
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                rnd = random.random()
                mut_pow = 1.0 / (20 + 1.0)
                if rnd <= 0.5:
                    xy = 1.0-delta1
                    val = 2.0 * rnd+(1.0-2.0 * rnd) * (math.pow(xy, (20+1.0)))
                    deltaq = math.pow(val, mut_pow) - 1.0
                else:
                    xy = 1.0-delta2
                    val = 2.0 * (1.0-rnd)+2.0 * (rnd-0.5) * (pow(xy, (20+1.0)))
                    deltaq = 1.0 - (pow(val, mut_pow))
                y = y + deltaq * (yu-yl)
                if y < yl:
                    y = yl
                if y > yu:
                    y = yu
                ind[j] = y
        return ind

    def update_reference(self, child):
        '''
        update reference point
        :param child:
        :return:
        '''
        p = self.problem.evaluate(child)
        self.count_fitness+=1
        for i in range(self.n_obj):
            if p[i] < self.ideal_fitness[i]:
                self.ideal_fitness[i] = p[i]
                self.ideal_point[i] = child
        return p

    def update_problem(self,child, fit, id):
        for i in range(self.max_neighborhood_size):
            k = self.pop[id].neighbor[i]
            f1 = self.scalar_func(self.pop[k].pop_fitness, self.pop[k].namda)
            f2 = self.scalar_func(fit, self.pop[k].namda)
            if f2 < f1:
                self.pop[k].pop_x = child
                self.pop[k].pop_fitness = fit

    def scalar_func(self, y_obj, namda):
        '''
        分布函数
        :param y_obj:
        :param namda:
        :param idealpoint:
        :return:
        '''
        max_fun = -1.0e+30
        for n in range(self.n_obj):
            diff = math.fabs(y_obj[n] - self.ideal_fitness[n])
            if namda[n]==0:
                feval = 0.00001*diff
            else:
                feval = diff*namda[n]
            if feval>max_fun:
                max_fun = feval
        return max_fun

def problems_test():
    '''
    一系列函数的测试
    :return:
    '''
    for id in range(len(problems)):
        print('DTLZ{} starting……'.format(id))
        problem_test(problem=problems[id])

def problem_test(problem):
    '''
    使用单个问题测试MOEA/D
    :param problem:
    :return:
    '''
    model = MOEAD(problem=problem)
    model.run()
    pops, x, y, z = extract_info(model)
    pf = get_pflist('./pf_files/n10000/{}.txt'.format(model.problem.name()))
    reference_point = get_referencepoint(pops)  # 获取参考点
    hv = HyperVolume(reference_point=reference_point)   #计算超体积HV
    hv_score = hv.compute(model.pop)
    igd = get_igd(pf, pops) # 计算反世代距离IGD
    print('hyper volume is {}'.format(hv_score))
    print('inverted generational distance is {}'.format(igd))
    draw_scatter3D(model.problem.name(), hv_score, igd, reference_point, x, y, z)


def draw_scatter3D(pname, hv_score, igd, reference_point, x, y, z):
    '''
    画3D散点图
    :param hv_score:
    :param igd:
    :param x:
    :param y:
    :param z:
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    x_r, y_r, z_r = reference_point
    ax.scatter(x_r, y_r, z_r, c='r')
    ax.set_xlabel('func1')
    ax.set_ylabel('func2')
    ax.set_zlabel('func3')
    title = '{} pareto front\nHV:{}\nIGD:{}'.format(pname, hv_score, igd)
    ax.set_title(title)
    plt.show()


def extract_info(model):
    '''
    从model运行结果中拿出后面要用的信息
    :param model:
    :return:
    '''
    x = []
    y = []
    z = []
    pops = []
    for p in model.pop:
        xx, yy, zz = p.pop_fitness
        pops.append(p.pop_fitness)
        x.append(xx)
        y.append(yy)
        z.append(zz)
    return pops, x, y, z


if __name__ == '__main__':
    problem_test(DTLZ7)
