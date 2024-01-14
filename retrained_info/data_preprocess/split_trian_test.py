#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import os
import numpy as np
import shutil
import math
import sys


SRC_DIR = './filtered_traces/'
TRAIN_DIR = './cooked_train_traces/'
TEST_DIR = './cooked_test_traces/'

# TRACE_CLASSES = ['norway3g', 'belgium4g', 'lumos5g', 'soliswifi']
TRACE_CLASSES = ['norway3g', 'lumos4g', 'lumos5g', 'soliswifi']

TRAIN_FRAC = 0.8


def mkdir(dir):
    is_exists = os.path.exists(dir)
    if not is_exists:
        os.makedirs(dir)


def read_and_split(src_dir=SRC_DIR, train_dir=TRAIN_DIR, test_dir=TEST_DIR):

    mkdir(train_dir)
    mkdir(test_dir)

    trace_class_list = os.listdir(src_dir)
    for trace_class in trace_class_list:
        if trace_class.strip() not in TRACE_CLASSES and not os.path.isdir(trace_class):
            continue

        trace_list = os.listdir(os.path.join(src_dir, trace_class))
        trace_class_idx = TRACE_CLASSES.index(trace_class)

        trace_cnt = {}
        for trace in trace_list:
            if trace_class_idx == 0: # 3G
                type = trace.split('_')[1]
                if '.' in type:
                    type = type.split('.')[0]
            elif trace_class_idx == 1: # 4G
                # type = trace.split('_')[1]     # belgium4g
                type = trace.split('_')[2]       # lumos4g
            elif trace_class_idx == 2:  # 5G
                type = trace.split('_')[3]
            elif trace_class_idx == 3:  # WiFi
                type = trace.split('_')[1]
            
            if type not in trace_cnt:
                trace_cnt[type] = [trace]
            else:
                trace_cnt[type].append(trace)

        # traverse trace_cnt to randomly split train and test set at a ratio of (TRAIN_FRAC: (1-TRAIN_FRAC))
        # each type of trace should be split separately
        # copy train traces to train_dir, and test traces to test_dir
        for type, traces in trace_cnt.items():
            train_cnt = math.ceil(len(traces) * TRAIN_FRAC)
            test_cnt = len(traces) - train_cnt

            # randomly select train_cnt traces from traces
            train_traces = np.random.choice(a=traces, size=train_cnt, replace=False)
            for train_trace in train_traces:
                src_path = os.path.join(src_dir, trace_class, train_trace)
                dst_path = os.path.join(train_dir, train_trace)
                shutil.copy(src_path, dst_path)

            # copy the rest traces to test_dir
            for trace in traces:
                if trace not in train_traces:
                    src_path = os.path.join(src_dir, trace_class, trace)
                    dst_path = os.path.join(test_dir, trace)
                    shutil.copy(src_path, dst_path)


if __name__ == '__main__':

    read_and_split()
