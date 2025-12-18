This repo includes the code to run MTE before training

We use a cluster to run the MTE in parallel for all timestamps. Please note we are using SLURM to do this.

MTE_read.py constructs the results from MTE analysis as a 4D tensor

[time, time, 52, 52]

52 is number of regions: states in the US, we use 0-51 as the index for regions; each state is a process

For example, [8, 12, 0, 1] = 0.1 represents 0[8] has an influence on 1[12] with a weight of magnitude of 0.1
