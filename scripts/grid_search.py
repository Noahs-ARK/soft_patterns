#!/usr/bin/env  python

import sys
import os
import os.path
import copy
import subprocess
from operator import mul
from random import shuffle
from operator import mul
from functools import reduce


dirs=['stanford_sentiment_binary', 'amazon_reviews', 'ROC_stories', 'stanford_sentiment_binary_100',
	  'stanford_sentiment_binary_500', 'stanford_sentiment_binary_1000', 'stanford_sentiment_binary_2500',
	  'amazon_reviews_100', 'amazon_reviews_500', 'amazon_reviews_1000', 'amazon_reviews_2500',
	  'amazon_reviews_5000', 'amazon_reviews_10000']

n_dirs=len(dirs)

WORK = os.environ['HOME']
model_dir = WORK + "/work/soft_patterns/"
resource_dir = WORK + "/resources/"


def main(args):
	gpu = None
	starting_point = 0
	indices_to_run = None
	if len(args) < 4:
		print("Usage:", args[0], "<dataset> <file name> <n instsances> <gpu (optional)> <starting point = 0> <specific instances to run>")
		print("Dirs are:")
		for i in range(n_dirs):
			print("{}: {}".format(i, dirs[i]))
		return -1
	elif len(args) > 4:
		gpu = args[4]
		if len(args) > 5:
			starting_point = int(args[5])
			if len(args) > 6:
				indices_to_run = set([int(x) for x in args[6].split(",")])

	ddir = dirs[int(args[1])]
	data_dir = resource_dir + "/text_cat/" + ddir
	n_instances = int(args[3])
	file_name = args[2]

	name = ddir+'_'+".".join(file_name.split("/")[-1].split(".")[:-1])
	with open(file_name) as ifh:
		all_args = [l.rstrip().split() for l in ifh]

	all_args = [l if len(l) > 1 else [l[0], ''] for l in all_args]

	n = reduce(mul, [len(x)-1 for x in all_args], 1)
	print(all_args)

	print("Got {} different configurations".format(n))

	if indices_to_run is None:
		indices_to_run = x = [i for i in range(starting_point, n)]
		shuffle(indices_to_run)
		indices_to_run = set(indices_to_run[:n_instances])
	# print("In2run:", indices_to_run)
	recursive_run_code(all_args, 0, 0, [], data_dir, indices_to_run, name, gpu)

def recursive_run_code(all_args, curr_param_index, curr_index, curr_values, data_dir, indices_to_run, name, gpu):
	if curr_param_index == len(all_args):
		curr_index += 1
		if curr_index in indices_to_run:
			run_code(all_args, curr_values, data_dir, name, curr_index, gpu)
		# else:
		# 	print(curr_index, "failed")
	else:
		for j in all_args[curr_param_index][1:]:
			curr_values_tmp = copy.deepcopy(curr_values)
			curr_values_tmp.append(j)
			curr_index = recursive_run_code(all_args, curr_param_index + 1, curr_index, curr_values_tmp, data_dir,
											indices_to_run, name, gpu)

	return curr_index

def run_code(all_args, curr_values, data_dir, name, curr_index, gpu):
	# print("Running", name, "with args", curr_values)
	git_tag = os.popen('git log | head -n 1 | cut -d " " -f2 | cut -b -7').read().rstrip()

	s = name + "." + str(curr_index)
	odir =  model_dir + "/output_"+s

	args = ['python', '-u', 'soft_patterns.py', "--td",  data_dir + "/train.data", "--tl",  data_dir + "/train.labels",
			"--vd", data_dir + "/dev.data", "--vl", data_dir + "/dev.labels",
			"--model_save_dir", odir]

	params = [[all_args[i][0], curr_values[i]] for i in range(len(all_args))]
	# print("p is "+str(params))
	params = [item for sublist in params for item in sublist]
	args += params

	# print(args)

	HOSTNAME = os.environ['HOSTNAME'] if 'HOSTNAME' in os.environ else ''

	cmd = " ".join(args)
	if HOSTNAME.endswith('.stampede2.tacc.utexas.edu'):
		f = gen_cluster_file(s, cmd)

		os.system('sbatch' + f)
	else:
		if gpu != None:
			os.environ['CUDA_VISIBLE_DEVICES'] = gpu
			cmd += ' -g'

		print(cmd)
		of=model_dir+'logs/'+s + '_' + str(git_tag) + ".out" 

		if os.path.isfile(of):
			print("Output file "+of+" found. Continuing")
		else:
			os.system(cmd+" |& tee "+of)


def	gen_cluster_file(s, com):
	f = model_dir + "/runs/"+s

	print("Writing", f)
	with open(f, 'w') as ofh:
		ofh.write("#!/usr/bin/env bash\n")
		ofh.write("#SBATCH -J "+s+"\n")
		ofh.write("#SBATCH -o " + model_dir + "/logs/" + s + '_' + str(git_tag) + ".out\n")
		ofh.write("#SBATCH -p normal\n")  # specify queue
		ofh.write("#SBATCH -N 1\n")  # Number of nodes, not cores (16 cores/node)
		ofh.write("#SBATCH -n 1\n")
		ofh.write("#SBATCH -t 48:00:00\n")  # max time
		ofh.write("#SBATCH --mail-user=roysch@cs.washington.edu\n")
		ofh.write("#SBATCH --mail-type=ALL\n")
		ofh.write("#SBATCH -A TG-DBS110003       # project/allocation number;\n")
		ofh.write("source activate torch3\n")
		ofh.write("mpirun " + com + "\n")

	return f


if __name__ == "__main__":
	sys.exit(main(sys.argv))
