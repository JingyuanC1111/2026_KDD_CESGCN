This repo includes the code to run MTE before training

We use a cluster to run the MTE in parallel for all timestamps. Please note we are using SLURM to do this.

MTE_read.py constructs the results from MTE analysis as a 4D tensor

[time, time, 52, 52]

52 is number of regions: states in the US, we use 0-51 as the index for regions; each state is a process

For example, [8, 12, 0, 1] = 0.1 represents 0[8] has an influence on 1[12] with a weight of magnitude of 0.1

To run the MTE analysis, each week, we only need to update the dataset and the timestamp in job.sbatch, currently, for the most recent forecasting, we have set it to 200, as we have 200 weeks of data till now. For the next forecasting week, we change it to 201 etc. Every new week, we only need to run MTE for the latest timestamp. Meanwhile, remember to expand the time dimension in the MTE_read.py:

MTE_static_matrices = torch.zeros((200, 200, 52, 52))
    for time in range(12, 201):

change it to:

MTE_static_matrices = torch.zeros((201, 201, 52, 52))
    for time in range(12, 202):

The MTE_matrices_flu_hosp.npy we get after we run MTE_read.py is the latest 4D tensor we constructed from MTE analysis.

