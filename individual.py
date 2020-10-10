#!/usr/bin/env python
# encoding: utf-8
# @author: lishaogang
# @file: individual.py
# @time: 2020/9/27 0027 20:01
# @desc:
import numpy as np

class Individual():
    def __init__(self):
        self.vector = []
        self.neighbor = []
        self.pop_x = []
        self.pop_fitness = float('inf')
        self.namda = []

