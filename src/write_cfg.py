import json
import pandas as pd
import sys
master_date=sys.argv[1]
horizon=4
data = {}
cfg_name='cfg'

data['input_path']='https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/target-data/target-hospital' \
                   '-admissions.csv'
data['temp_output_path']='../output/'
data['output_path']='/home/jc2wv/FluX-forecasting/FluX-forecasing/output/CESGCN/format_std/'
data['reference_date'] = master_date
f_dates=[]
for d in range(-1,horizon):
    f_dates.append((pd.to_datetime(master_date)+pd.Timedelta(weeks=d)).strftime("%Y-%m-%d"))
data['fct_date'] = f_dates


with open('../cfg/'+cfg_name, 'w') as outfile:
    json.dump(data, outfile)
