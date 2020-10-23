#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: fileProcess.py
# @time: 2020/10/22 0022 17:03
# @desc:

def savePareto2Txt(name,pareto):
    '''
    保存Pareto前沿到txt文件中
    :param name:
    :param pareto:
    :return:
    '''
    with open('./results/{}_res.txt'.format(name), 'w+', encoding='utf-8') as f:
        for items in pareto:
            if items!=[]:
                f.write('\t'.join(list(map(str, list(items))))+'\n')
            else:
                print('EORRO:88888888')
                print(pareto)
    print('保存至‘./results/{}_res.txt’成功'.format(name))

def readPareto4Txt(name):
    '''
    从txt文件中读取pareto前沿
    :param name:
    :return:
    '''
    pareto = []
    with open('./results/{}.txt'.format(name),'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            pareto.append(list(map(float, line.strip().split('\t'))))
    print("读取‘./results/{}_res.txt'成功".format(name))
    return pareto

