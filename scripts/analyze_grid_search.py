#!/usr/bin/env python

import sys
import os.path
import glob
import numpy as np

home = os.environ.get('HOME')
workdir=home+"/work/soft_patterns/"

def main(args):
    if len(args) < 2:
        print("Usage:",args[0],"<dataset name>")
        return -1

    dataset = args[1]

    p_keys = ['6-20_5-20_4-10_3-10_2-10', '6-10_5-10_4-10_3-10_2-10', '7-10_6-10_5-10_4-10_3-10_2-10']
    l_keys = ['0.01', '0.005', '0.001']
    t_keys = ['0.05', '0.1', '0.2']
    d_keys = ['10', '25', '50', '100']
    s_keys = ['25', '50', '100']

    ps = dict([(k, []) for k in p_keys])
    ls = dict([(k, []) for k in l_keys])
    ts = dict([(k, []) for k in t_keys])
    ds = dict([(k, []) for k in d_keys])
    ss = dict([(k, []) for k in s_keys])

    for p in p_keys:
        for l in l_keys:
                for t in t_keys:
                        for d in d_keys:
                            for s in s_keys:
                                f='{}output_p{}_d{}_l{}_t{}_r_mt_b150_clip0_840B.300d_w0.1_{}_seed{}_*/output.dat'.format(
                                        workdir, p, d, l, t, dataset, s)
                                best = get_top(f)

                                if best != -1:
                                    ps[p].append(best)
                                    ls[l].append(best)
                                    ts[t].append(best)
                                    ds[d].append(best)
                                    ss[s].append(best)

    analyze("patterns", ps)
    analyze("learning rate", ls)
    analyze("dropout", ts)
    analyze("dimension", ds)
    analyze("seed", ss)

    return 0

def analyze(str, kv):
    print(str+":")

    for k,v in kv.items():
        if len(v):
            print("\t{}: Max: {:,.3f}, Mean: {:,.3f} {}".format(k, np.max(v), np.mean(v), len(v)))
        else:
            print("\t",k,"No files")

def get_top(f):
    fs = glob.glob(f)
    if not len(fs):
        return -1
    
    maxv = -1

    with open(fs[0]) as ifh:
        for l in ifh:
            if l.find('dev loss:') != -1:
                e = l.rstrip().split()

                acc = float(e[-1][:-1])

                if acc > maxv:
                    maxv = acc
    return maxv


if __name__ == '__main__':
    sys.exit(main(sys.argv))
