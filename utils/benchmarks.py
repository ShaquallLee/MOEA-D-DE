#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: benchmarks.py
# @time: 2020/10/18 0018 14:21
# @desc: 为建立benchmark函数设的类，其中包含了要用的benchmark
from pymop.problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7

class PROBLEM():
    def __init__(self,n_var,n_obj, lbound, rbound):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = lbound
        self.xu = rbound

class F1(PROBLEM):
    def __init__(self):
        PROBLEM.__init__(self, 30, 2, 0, 1)
        self.name = "F1"

    def evaluate(self, point):
        return 1+1
