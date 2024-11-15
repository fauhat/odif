import utils
import os
import argparse
import pandas as pd
import numpy as np
from config import get_algo_config, get_algo_class
from parser_utils import parser_add_model_argument, update_model_configs

dataset_root = 'data'

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=int, default=1,
                    help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--input_dir", type=str, default='tabular', help="the path of the data sets")
parser.add_argument("--output_dir", type=str, default='&tabular_record/',
                    help="the output file path")
parser.add_argument("--dataset", type=str, default='FULL',
                    help="FULL represents all the csv file in the folder, or a list of data set names split by comma")
parser.add_argument("--model", type=str, default='dif')
parser.add_argument('--contamination', type=float, default=-1,
                    help='this is used to estimate robustness w.r.t. anomaly contamination')
parser.add_argument('--silent_header', action='store_true')
parser.add_argument('--save_rep', action='store_true')
parser.add_argument('--save_score', action='store_true')
parser.add_argument("--flag", type=str, default='')

parser = parser_add_model_argument(parser)
args = parser.parse_args()

print("args.input_dir: ", args.input_dir)
print("args.dataset: ", args.dataset)

data_lst = utils.get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)

print("data_lst: ", data_lst)

model_class = get_algo_class(args.model)
model_configs = get_algo_config(args.model)
model_configs = update_model_configs(args, model_configs)
print('model configs:', model_configs)
print('model_configs.device -->',model_configs["device"])
model_configs['device'] = 'cpu'
print('model_configs.device -->',model_configs["device"])
print('model configs:', model_configs)
model_configs['n_ensemble'] = 18
dif_object = model_class(**model_configs, random_state=42)
print('dif_object:', dif_object)

for f in data_lst:
    if f.endswith('pkl'):
        df = pd.read_pickle(f)
    elif f.endswith('csv'):
        df = pd.read_csv(f)
    else:
        continue
    dataset_name = os.path.splitext(os.path.split(f)[1])[0]
    x, y = utils.data_preprocessing(df)

    # Iterate through each record and print
    for index, row in df.iterrows():
        # print("Dataset Name:", dataset_name)
        # print("Record:")
        # print(row)
        dif_object.n_features = x.shape[-1]
        print("dif_object.n_features: ", dif_object.n_features)
        ensemble_seeds = np.random.randint(0, 1e+5, dif_object.n_ensemble)
        if dif_object.verbose >= 2:
            net = dif_object.Net(n_features=dif_object.n_features, **dif_object.network_args)
            print("net: ",net)
        x_single = np.array([row]).reshape(1,-1)
        print("x_single.shape: ", x_single.shape)        
        #row_reduced = dif_object._inference_transfer([row])

        #reduce the row here using the same principle used in the DIF class
        # if dif_object.data_type == 'tabular' and x_single.shape[0] == dif_object.x_reduced_lst[0].shape[0]:
        #     reduced_row = dif_object.x_reduced_lst[0][0]  # Assuming self.x_reduced_lst[0] contains the representations for the first network
        # elif dif_object.new_ensemble_method:
        #     reduced_row = dif_object.deep_transfer_batch_ensemble(x_single, dif_object.net_lst[0])[0]
        # else:
        #     reduced_row = dif_object.deep_transfer(x_single, dif_object.net_lst[0])[0]

        #reduced_row = dif_object.deep_transfer_batch_ensemble(x_single, dif_object.net_lst[0])[0]
        #reduced_row = dif_object.deep_transfer_batch_ensemble(x_single, dif_object.net_init(dif_object.Net(n_features=len(row), **dif_object.network_args).to(dif_object.device)))[0]

        net = dif_object.Net(n_features=len(row), **dif_object.network_args).to(dif_object.device)
        dif_object.net_init(net)

        reduced_row = dif_object.deep_transfer_batch_ensemble(x_single, net)[0]


        print("Dataset Name:", dataset_name)
        print("Record:")
        print("row: ", row)        
        print("reduced_row:: ", reduced_row[0][0])  
        # for idx, value in enumerate(reduced_row):
        #     print(f"Column {idx + 1}: {value}")   

        #print(row.values)
        #print("2nd Field Value:", row.iloc[1])  # Assuming the 2nd field is at index 1
        break

    # # Experiment: robustness w.r.t. different anomaly contamination rate
    if args.contamination != -1:
        x_train, y_train = utils.adjust_contamination(x, y,
                                                      contamination_r=args.contamination,
                                                      swap_ratio=0.5,
                                                      random_state=2021)
    else:
        x_train, y_train = x, y
    
    print("x_train: ", x_train)
    print("y_train: ", y_train)