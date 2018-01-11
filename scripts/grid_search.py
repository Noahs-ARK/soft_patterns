#!/usr/bin/env  python

import sys
import os
import copy
from subprocess import call
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
	if len(args) < 3:
		print("Usage:", args[0], "<dataset> <file name> <n instsances>")
		print("Dirs are:")
		for i in range(n_dirs):
			print("{}: {}".format(i, dirs[i]))
		return -1

	dir = dirs[int(args[1])]
	data_dir = resource_dir + "/text_cat/" + dir
	n_instances = int(args[3])
	file_name = args[2]

	name = ".".join(file_name.split("/")[-1].split(".")[:-1])
	with open(file_name) as ifh:
		all_args = [l.rstrip().split() for l in ifh]

	all_args = [l if len(l) > 1 else [l[0], ''] for l in all_args]

	n = reduce(mul, [len(x)-1 for x in all_args], 1)
	print(all_args)


# ps=(6:20,5:20,4:10,3:10,2:10 6:10,5:10,4:10,3:10,2:10 7:10,6:10,5:10,4:10,3:10,2:10 6:10,5:10,4:10 5:10,4:10,3:10,2:10)
# ls=(0.01 0.05 0.01 0.005)
# ws=(0 0.05 0.1 0.15)
# ts=(0 0.05 0.1 0.2 0.3)
# ds=(0 10 25 50)

	print("Got {} different configurations".format(n))

	indices_to_run = x = [i for i in range(n)]
	shuffle(indices_to_run)
	indices_to_run = set(indices_to_run[:n_instances])
	# print("In2run:", indices_to_run)
	recursive_run_code(all_args, 0, 0, [], data_dir, indices_to_run, name)

def recursive_run_code(all_args, curr_param_index, curr_index, curr_values, data_dir, indices_to_run, name):
	if curr_param_index == len(all_args):
		curr_index += 1
		if curr_index in indices_to_run:
			run_code(all_args, curr_values, data_dir, name, curr_index)
		# else:
		# 	print(curr_index, "failed")
	else:
		for j in all_args[curr_param_index][1:]:
			curr_values_tmp = copy.deepcopy(curr_values)
			curr_values_tmp.append(j)
			curr_index = recursive_run_code(all_args, curr_param_index + 1, curr_index, curr_values_tmp, data_dir,
											indices_to_run, name)

	return curr_index

def run_code(all_args, curr_values, data_dir, name, curr_index):
	# print("Running", name, "with args", curr_values)


	s = name + "." + str(curr_index)
	odir =  model_dir + "/output_"+s

	args = ["--td",  data_dir + "/train.data", "--tl",  data_dir + "/train.labels",
			"--vd", data_dir + "/dev.data", "--vl", data_dir + "/dev.labels",
			"--model_save_dir", odir]

	params = [[all_args[i][0], curr_values[i]] for i in range(len(all_args))]
	# print("p is "+str(params))
	params = [item for sublist in params for item in sublist]
	args += params


	# print(args)

	HOSTNAME = os.environ['HOSTNAME']

	if HOSTNAME.endswith('.stampede2.tacc.utexas.edu'):
		f = gen_cluster_file(s, " ".join(args))

		call(['sbatch', f])
	else:
		call(['python', '-u', 'soft_patterns.py']+  args)



def	gen_cluster_file(s, com):
	f = model_dir + "/runs/"+s

	print("Writing", f)
	with open(f, 'w') as ofh:
		ofh.write("#!/usr/bin/env bash\n")
		ofh.write("#SBATCH -J "+s+"\n")
		ofh.write("#SBATCH -o " + model_dir + "/logs/" + s + ".out\n")
		ofh.write("#SBATCH -p normal\n")  # specify queue
		ofh.write("#SBATCH -N 1\n")  # Number of nodes, not cores (16 cores/node)
		ofh.write("#SBATCH -n 1\n")
		ofh.write("#SBATCH -t 48:00:00\n")  # max time
		ofh.write("#SBATCH --mail-user=roysch@cs.washington.edu\n")
		ofh.write("#SBATCH --mail-type=ALL\n")
		ofh.write("#SBATCH -A TG-DBS110003       # project/allocation number;\n")
		ofh.write("source activate torch3\n")
		ofh.write("mpirun python -u soft_patterns.py " + com + "\n")

	return f


if __name__ == "__main__":
	sys.exit(main(sys.argv))
