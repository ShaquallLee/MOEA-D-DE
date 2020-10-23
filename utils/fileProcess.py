#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: fileProcess.py
# @time: 2020/10/22 0022 17:03
# @desc:

from xlwt import *

def saveArray2Excel(name, data):
    '''
    保存数据到excel文件中
    :param name:
    :param data:
    :return:
    '''
    f = Workbook(encoding='utf-8')
    table = f.add_sheet('sheet1')
    for i in range(len(name)):
        for j in range(len(name[i])):
            table.write(j, i, data[i][j])
    f.save(name)
    print("保存数据到{}文件成功".format(name))

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

