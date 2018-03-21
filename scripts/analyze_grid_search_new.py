#!/usr/bin/env python

import sys
import re
import os.path
import glob
import numpy as np


def main(args):
    home = os.environ.get('HOME')
    workdir=home+"/work/soft_patterns/logs/"
    type = 0
    if len(args) < 2:
        print("Usage:",args[0],"<prefix> <type (0 for accuracy [default], 1 for loss)> <work_dir = "+workdir+">")
        return -1
    elif len(args) > 2:
        type = int(args[2])
        if len(args) > 3:
             workdir = args[3]

    prefix = args[1]

    s=workdir + prefix + '*'

    files = glob.glob(s)

    if len(files) == 0:
        print("No files found for {} with regexp {}".format(prefix, s))
        return -2

    params = dict()

    global_best = None
    global_best_val = -1 if type == 0 else 1000

    n_files = 0
    all_vals = []
    for f in files:
        best = get_top(f, type)

        if best != -1:
            n_files += 1
            all_vals.append(best)
            get_local_params(params, f, best)
            print(best, f)

            if best > global_best_val and type == 0 or (best < global_best_val and type == 1):
                global_best = f
                global_best_val = best

    analyze(params, type)

    if n_files > 0:
         print("Overall best across {} files: {} ({}). Mean value is {}".format(n_files, global_best_val, global_best, round(np.mean(all_vals), 3)))

    return 0

def get_params(param_file):
    with open(param_file) as ifh:
        params = [x.split() for x in ifh]

    filtered_params = dict()
    for p in params:
        if len(p) > 2:
            p[0] = p[0][2:]
            filtered_params[p[0]] = dict([(x,[]) for x in p[1:]])

    return filtered_params

def get_local_params(params, f, v):
    with open(f) as ifh:
        l = ifh.readline()

    vs = l.rstrip()[10:].split()

    for x in vs:
        e = x[:-1].split('=')
        if e[1][0] == "'":
            e[1] = e[1][1:-1]

        if e[0] not in params:
            if e[0] == 'model_save_dir':
                continue

            params[e[0]] = dict()

        if e[1] not in params[e[0]]:
            params[e[0]][e[1]] = []
    
        params[e[0]][e[1]].append(v)

def analyze(local_params, type):
    for name in local_params:
        if (len(local_params[name]) == 1):
            continue

        print(name+":")

        for k,v in local_params[name].items():
            if len(v):
                print("\t{}: {}: {:,.3f}, Mean: {:,.3f} {}".format(k, "Max" if type == 0 else "Min", np.max(v) if type == 0 else np.min(v), round(np.mean(v),3), len(v)))
            else:
                print("\t",k,"No files")

def get_top(f, type):
    maxv = -1 if type == 0 else 1000

    with open(f) as ifh:
        for l in ifh:
            if l.find('dev loss:') != -1:
                e = l.rstrip().split()

                if type == 0:
                    acc = float(e[-1][:-1])
                    if acc > maxv:
                        maxv = acc
                else:
                    loss = float(e[-3])
                    if loss < maxv:
                        maxv = loss


    return maxv


if __name__ == '__main__':
    sys.exit(main(sys.argv))
