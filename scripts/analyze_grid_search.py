#!/usr/bin/env python

import sys
import re
import os.path
import glob
import numpy as np

home = os.environ.get('HOME')
workdir=home+"/work/soft_patterns/"

def add_val(res, index, odict):
    v = res.group(index)
    #print(v,'for',index)

    if v not in odict:
        odict[v] = []

    return v

def main(args):
    type = 0
    if len(args) < 2:
        print("Usage:",args[0],"<dataset name> <type (0 for accuracy [default], 1 for loss)>")
        return -1
    elif len(args) > 2:
        type = int(args[2])

    dataset = args[1]

    s=workdir+'/'+'output_p*_d*_l*_t*_r_mt_b150_clip0_840B.300d_w*_'+dataset+'_seed*_*/output.dat'
    files = glob.glob(s)

    if len(files) == 0:
        print("No files found for", dataset)
        return -2

    ps = dict()
    ls = dict()
    ts = dict()
    ds = dict()
    ss = dict()
    ws = dict()
    bs = dict()
    reg = re.compile("output_p(.*)_d([0-9]+)_l([0-9\.]+)_t([0-9\.]+)_r_mt_b150_clip0_840B\.300d_w([0-9\.]+)_"+dataset+'_seed(\d+)(:?_bh([0-9]+))?_\w+')

    global_best = None
    global_best_val = -1 if type == 0 else 1000

    for f in files:
        fname = f.split("/")[-2]
        res = reg.match(fname)

        if res is None:
            print(fname,"doesn't match regexp")
            return

        best = get_top(f, type)

        if best != -1:
            p = add_val(res, 1, ps)
            d = add_val(res, 2, ds)
            l = add_val(res, 3, ls)
            t = add_val(res, 4, ts)
            w = add_val(res, 5, ws)
            s = add_val(res, 6, ss)
            b = add_val(res, 7, bs)

            if b is None:
                continue

            if best > global_best_val and type == 0 or (best < global_best_val and type == 1):
                global_best = f
                global_best_val = best

            ps[p].append(best)
            ds[d].append(best)
            ls[l].append(best)
            ts[t].append(best)
            ws[w].append(best)
            ss[s].append(best)
            bs[b].append(best)

    analyze("patterns", ps, type)
    analyze("learning rate", ls, type)
    analyze("dropout", ts, type)
    analyze("dimension", ds, type)
    analyze("seed", ss, type)
    analyze("word dropout", ws, type)
    analyze("rnn hidden dim", bs, type)

    print("Overall best: {} ({})".format(global_best_val, global_best))
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
