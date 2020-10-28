#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: fileProcess.py
# @time: 2020/10/22 0022 17:03
# @desc:

from xlwt import *
import time

def saveArray2Excel(name, data):
    '''
    保存数据到excel文件中
    :param name:
    :param data:
    :return:
    '''
    f = Workbook(encoding='utf-8')
    table = f.add_sheet('sheet1')
    for i in range(len(data[0])):
        table.write(0, i+1, "第{}次".format(i+1))
    table.write(0, len(data[0])+1, '均值')
    table.write(0, len(data[0])+2, '最小值')
    table.write(0, len(data[0])+3, '最大值')
    table.write(1,0, "IGD")
    table.write(2,0, "HV")
    for i in range(len(data)):
        for j in range(len(data[i])):
            table.write(i+1,j+1, data[i][j])
        table.write(i+1,len(data[i])+1, sum(data[i])/len(data[i]))
        table.write(i+1,len(data[i])+2, min(data[i]))
        table.write(i+1,len(data[i])+3, max(data[i]))
    f.save(name)
    print("保存数据到{}文件成功".format(name))

def saveRes2Excel(fname, res):
    '''
    保存数据到excel文件中
    :param name:
    :param data:
    :return:
    '''
    f = Workbook(encoding='utf-8')
    for name, data in res.items():
        table = f.add_sheet(name)
        for i in range(len(data[0])):
            table.write(0, i+1, "第{}次".format(i+1))
        table.write(0, len(data[0])+1, '均值')
        table.write(0, len(data[0])+2, '最小值')
        table.write(0, len(data[0])+3, '最大值')
        table.write(1,0, "IGD")
        table.write(2,0, "HV")
        for i in range(len(data)):
            for j in range(len(data[i])):
                table.write(i+1,j+1, data[i][j])
            table.write(i+1,len(data[i])+1, sum(data[i])/len(data[i]))
            table.write(i+1,len(data[i])+2, min(data[i]))
            table.write(i+1,len(data[i])+3, max(data[i]))
    f.save(fname)
    print("保存数据到{}文件成功".format(fname))

def savePareto2Txt(name,pareto):
    '''
    保存Pareto前沿到txt文件中
    :param name:
    :param pareto:
    :return:
    '''
    tname = './results/{}_res_{}.txt'.format(name, time.time())
    with open(tname, 'w+', encoding='utf-8') as f:
        for items in pareto:
            if items!=[]:
                f.write('\t'.join(list(map(str, list(items))))+'\n')
            else:
                print('EORRO:88888888')
                print(pareto)
    print("保存至‘{}'成功".format(tname))

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

