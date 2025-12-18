## Environment

The code is implemented in Python 3.9.0. All required packages to run the code are listed in `requirements.txt`.

`Unified_graph_variant.py` is the ablation variant designed to validate the mechanism-parallel design of graph construction.

`write_cfg.py` specifies the target date for forecasting.

`layer.py` and `net.py` define the model architecture.

`trainer.py` and `util.py` provide the functional modules used during model training and execution.

`eval_func.py` describes the design of the evaluation metrics.

`WIS_RE.py` evaluates the WIS score of CESGCN, compared with the FluSight_baseline and FluSight_ensemble models. Users can alter the 'starting_date' in the code and check the WIS performance up to 4 timestamps ahead, remember using the correct ground truth


---

## Running Strategy

First, we download new data from https://github.com/cdcepi/FluSight-forecast-hub/blob/main/target-data/target-hospital-admissions.csv

Then, we run `data_processing.py` to convert data to a proper format for further MTE analysis.

Third step, we feed the output dataset to replace the dataset in src/MTE matrix, and run the corresponding sbatch file, remember replacing the using the latest time index.

Fourth, for each latest forecasting week, update MTE result.

Then, run `MTE_read.py` to get the latest constructed CESG, remember updating the dimension of the 4D tensor.

Finally, run `CESGCN_Train_From_Scratch.py` to train the model from scratch, a set of tuning parameter is available (learning_rate, node_dim, dropout rate).

The current best model has node_dim=16, lr=0.02, dropout=0.2, and metric is set to MSE (MAE is also available, but MSE shows better performance compared to MAE)

Or we can use `CESGCN_Using_Prev_Best_Model.py` to directly predict using the previous trained best model.
