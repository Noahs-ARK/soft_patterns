#!/usr/bin/env python

import sys
import re
import os.path
import glob
import numpy as np


def add_val(res, index, odict):
    v = res.group(index)
    #print(v,'for',index)

    if v not in odict:
        odict[v] = []

    return v

def main(args):
    home = os.environ.get('HOME')
    workdir=home+"/work/soft_patterns/"
    type = 0
    if len(args) < 2:
        print("Usage:",args[0],"<dataset name> <type (0 for accuracy [default], 1 for loss)> <workdir ="+workdir+">")
        return -1
    elif len(args) > 2:
        type = int(args[2])
        if len(args) > 3:
            workdir = args[3]

    dataset = args[1]

#    s=workdir+'/'+'output_p*_d*_l*_t*_r_mt_b150_clip0_840B.300d_w*_'+dataset+'_seed*_*/output.dat'
    s=workdir+'/'+'*'+dataset+'*'
    files = glob.glob(s)

    if len(files) == 0:
        print("No files found for", dataset)
        return -2

    ls = dict()
    ts = dict()
    ds = dict()
    hs = dict()
    zs = dict()
    cs = dict()
    reg = re.compile(".*_l([0-9\.]+)_t([0-9\.]+)_d([0-9]+)_h([0-9]+)_z([0-9]+)_c([0-9\.])_"+dataset+'.+')
    #reg = re.compile(".*_l([0-9\\.]+)")

    global_best = None
    global_best_val = -1 if type == 0 else 1000
    all_bests = []

    for f in files:
        fname = f.split("/")[-1]
        res = reg.match(fname)

        if res is None:
            print(fname,"doesn't match regexp",reg)
            return

        best = get_top(f, type)

        if best != -1:
            print(best, f)
            all_bests.append(best)
            l = add_val(res, 1, ls)
            t = add_val(res, 2, ts)
            d = add_val(res, 3, ds)
            h = add_val(res, 4, hs)
            z = add_val(res, 5, zs)
            c = add_val(res, 6, cs)

            if c is None:
                continue

            if best > global_best_val and type == 0 or (best < global_best_val and type == 1):
                global_best = f
                global_best_val = best

            ls[l].append(best)
            ts[t].append(best)
            ds[d].append(best)
            hs[h].append(best)
            zs[z].append(best)
            cs[c].append(best)

    analyze("learning rate", ls, type)
    analyze("dropout", ts, type)
    analyze("dimension", ds, type)
    analyze("hidden dimensions", hs, type)
    analyze("window size", zs, type)
    analyze("gradient clip", cs, type)

    print("Overall best across {} runs: {} ({}). Mean: {}".format(len(all_bests), global_best_val, global_best, round(np.mean(all_bests), 3)))
    return 0

def analyze(str, kv, type):
    print(str+":")

    for k,v in kv.items():
        if len(v):
            print("\t{}: {}: {:,.3f}, Mean: {:,.3f} {}".format(k, "Max" if type == 0 else "Min", np.max(v) if type == 0 else np.min(v), np.mean(v), len(v)))
        else:
            print("\t",k,"No files")

def get_top(f, type):
    fs = glob.glob(f)
    if not len(fs):
        return -1
  
    ind = 0 
    if len(fs) > 1:
        print(f,"has more than one")
        for i, f in enumerate(fs):
            from subprocess import call
            import sys
            print(i)
            call(["ls", "-l", f])

        res = int(input("Do you want to remove 0, 1 or -1 (none)?\n"))
        while res != 0 and res != 1 and res != -1:
            res = int(input("Got "+str(res)+". Please Enter 0, 1 or -1\n"))

        if res != -1:
            fn = "/".join(fs[res].split("/")[:-1])
            print(fn)
            call(["/bin/rm", "-rf", fn])
            ind = 1 - res
 
    maxv = -1 if type == 0 else 1000

    with open(fs[ind]) as ifh:
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
