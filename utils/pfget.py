#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: pfget.py
# @time: 2020/10/5 0005 18:46
# @desc:
import re


def get_pflist(file_name):
    '''
    从文件中读取帕累托前沿数据
    :param file_name:
    :return:
    '''
    pf_l = []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.readlines()
        for line in content:
            pf_l.append(list(map(float, re.split('\s+', line.strip()))))
    return pf_l

if __name__ == '__main__':
    pass
