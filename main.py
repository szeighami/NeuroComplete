import time
import sys
import json
import  os
import numpy as np
import subprocess
from data_utils import generate_training_queries

def run_RAQ(exp_base_path):
    print('******* Running Exp ' + exp_base_path + ' ***************')
    os.chdir(exp_base_path)
    with open('conf.json', 'r') as f:
        config = json.load(f) 

    start = time.time()

    print("Generating Training Data")

    queries, test_queries, res, test_res =  generate_training_queries(config['agg_type'], config['data_loc'],config['sel_id_path'], config['sel_cols'], config['cat_cols'], config['agg_col'],config['table_cols_new'], config['table_cols_id'], config['id_col'], config['predicates_list'], config)

    end = time.time()
    print("Training data generation took {:.2f}s".format(end-start))
    start = time.time()


    np.save('queries.npy', queries);
    np.save('res.npy', res);
    np.save('test_queries.npy', test_queries);
    np.save('test_res.npy', test_res);

    print("training model  --------------")
    os.environ["XLA_FLAGS"]  = "--xla_gpu_force_compilation_parallelism=1"
    p = subprocess.Popen(["python", config["path_to_neurocomplete"]+"/train_neurocomplete.py"])  

    end = time.time()
    print("Model Training took {:.2f}s".format(end-start))

    return

if __name__=='__main__':
    run_RAQ(".")
