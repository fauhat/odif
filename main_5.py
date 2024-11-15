import os
import pickle
import argparse
import time
import pandas as pd
import numpy as np
import utils
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

os.makedirs(args.output_dir, exist_ok=True)
data_lst = utils.get_data_lst(os.path.join(dataset_root, args.input_dir), args.dataset)
print(os.path.join(dataset_root, args.input_dir))
print("data_lst: ",data_lst)

model_class = get_algo_class(args.model)
model_configs = get_algo_config(args.model)
model_configs = update_model_configs(args, model_configs)
# print('model configs:', model_configs)
# print('model_configs.device -->',model_configs["device"])
model_configs['device'] = 'cpu'
# print('model_configs.device -->',model_configs["device"])
# print('model configs:', model_configs)

#---------------------ONLY FOR TESTING(MUST BE REMOVED OR COMMENTED OUT)----------------------------------
# model_configs['n_ensemble'] = 2
# model_configs['n_estimators'] = 3
#----------------------------------------------------------------------------------------------------------

cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
result_file = os.path.join(args.output_dir, f'{args.model}.{args.input_dir}.{args.flag}.csv')
print(result_file)

if not args.silent_header:
    f = open(result_file, 'a')
    print('\n---------------------------------------------------------', file=f)
    print(f'model: {args.model}, data dir: {args.input_dir}, dataset: {args.dataset}, '
          f'contamination: {args.contamination}, {args.runs}runs, ', file=f)
    for k in model_configs.keys():
        print(f'Parameters,\t [{k}], \t\t  {model_configs[k]}', file=f)
    print('---------------------------------------------------------', file=f) 
    print('data, auc-roc, std, auc-pr, std, time', file=f)
    f.close()

for f in data_lst:
    if f.endswith('pkl'):
        df = pd.read_pickle(f)
    elif f.endswith('csv'):
        df = pd.read_csv(f)
    else:
        continue
    dataset_name = os.path.splitext(os.path.split(f)[1])[0]
    x, y = utils.data_preprocessing(df)

    # # Experiment: robustness w.r.t. different anomaly contamination rate
    if args.contamination != -1:
        x_train, y_train = utils.adjust_contamination(x, y,
                                                      contamination_r=args.contamination,
                                                      swap_ratio=0.5,
                                                      random_state=2021)
    else:
        x_train, y_train = x, y
    
    # random_row_indices = np.random.choice(test_reduced_lst[0].shape[0], size=50)
    # online_train_reduced_ndarray = test_reduced_lst[0][random_row_indices]

    # random_row_indices = np.random.choice(x[0].shape[0], size=100)
    # x = x[random_row_indices]
    # x_train = x

    print("type(x): ", type(x))
    print("x.shape: ", x.shape)
    print("type(x_train): ", type(x_train))
    print("x_train.shape: ", x_train.shape)
    print("x_train: ", x_train)
    print("y_train: ", y_train)
    # args.runs = 3   #newly added by fauhat on 1st September Friday
    args.runs = 1   #newly added by fauhat on 17th March Friday
    auc_lst, ap_lst = np.zeros(args.runs), np.zeros(args.runs),
    t1_lst = np.zeros(args.runs)
    print("args.run: ",args.runs)
    print("t1_lst initialized: ", t1_lst)
    for i in range(args.runs):
        start_time = time.time()
        print(f'\nRunning [{i+1}/{args.runs}] of [{args.model}] on Dataset [{dataset_name}]')
          
        clf = model_class(**model_configs, random_state=42+i)
        #clf.n_ensemble = 50 #newly added by fauhat on 1st September Friday
        #clf.batch_size = None
        clf.fit(x_train)
        # print("x_reduced: ", clf.x_reduced_lst)
        #print("clf_lst: ", clf.clf_lst)
        t1 = time.time()
        #scores = clf.decision_function(x)
        scores = clf.decision_function_custom(x)
        #scores = clf.decision_function(x, x)
        #print("scores:: ",scores)

        # # ------ significance of synergy: replacing the random representation ensemble ------ # #
        if args.save_rep and hasattr(clf, "x_reduced_lst"):
            anom_idx, norm_idx = np.where(y == 1)[0], np.where(y == 0)[0]
            if len(norm_idx) > 1000:
                norm_idx = norm_idx[np.random.RandomState(42).choice(len(norm_idx), 1000, replace=False)]

            new_rep_lst = []
            for x_rep in clf.x_reduced_lst:
                anom, norm = x_rep[anom_idx], x_rep[norm_idx]
                x_rep = np.vstack([anom, norm])
                new_rep_lst.append(x_rep)

            save_dir = '&results_rep_quality_2208/'
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(
                new_rep_lst,
                open(save_dir + f'{dataset_name}_{args.model}_reduced_lst_full_anom.pkl', 'wb')
            )

        # # ------ significance of synergy: replacing the isolation-based anomaly scoring  ------ # #
        if args.save_score and hasattr(clf, "score_lst"):
            save_dir = '&results_score_quality_2208/'
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(
                clf.score_lst,
                open(save_dir + f'{dataset_name}_{args.model}_score_lst.pkl', 'wb')
            )

        auc, ap = utils.evaluate(y, scores)
        auc_lst[i], ap_lst[i] = auc, ap
        t1_lst[i] = t1 - start_time

        print(f'{dataset_name}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {t1_lst[i]:.1f}, {args.model}')

    # print("auc_lst::: ",auc_lst)
    print("dataset: ", dataset_name)
    # print("model-configs: ",model_configs)
    print("clf_lst: ", clf.clf_lst)
    print("clf_lst[0]: ", clf.clf_lst[0])
    # print("clf.clf_lst[0].estimators_: ", clf.clf_lst[0].estimators_)
    print("len(clf.clf_lst): ", len(clf.clf_lst))
    print("len(clf.clf_lst[0].estimators_): ", len(clf.clf_lst[0].estimators_))
    # print("clf_lst[0].estimators[0]: ", clf.clf_lst[0].estimators_[0])
    # print("clf_lst[0].estimators[0].tree: ", clf.clf_lst[0].estimators_[0].tree_)
    # print("clf_lst[0].estimators[0].tree :: node_count: ", clf.clf_lst[0].estimators_[0].tree_.node_count)
    # print("clf_lst[0].estimators[0].tree :: feature: ", clf.clf_lst[0].estimators_[0].tree_.feature)
    # print("clf_lst[0].estimators[0].tree :: threshold: ", clf.clf_lst[0].estimators_[0].tree_.threshold)
    #print("clf_lst[0].x_reduced_list shape ", clf.x_reduced_lst[0].shape)
    #print("clf_lst[0].x_reduced_list num cols ", clf.x_reduced_lst.shape[1])
    print("clf.max_samples_: ", clf.clf_lst[0].max_samples)
    """
    print("clf_lst[0].x_reduced_list: ", clf.x_reduced_lst)
    print("t1_lst.shape: ", t1_lst.shape)
    print("t1_lst: ", t1_lst)
    print("t1_lst[0]: ", t1_lst[0])
    #print("t1_lst[1]: ", t1_lst[1])
    #print("t1_lst[2]: ", t1_lst[2])
    """
    print("clf.x_reduced_list[0]: ", clf.x_reduced_lst[0])
    print("clf.x_reduced_list[0].shape[0]: ", clf.x_reduced_lst[0].shape[0])

    avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
    std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
    avg_time = np.average(t1_lst)

    f = open(result_file, 'a')
    txt = f'{dataset_name}, {avg_auc:.4f}, {std_auc:.4f}, ' \
          f'{avg_ap:.4f}, {std_ap:.4f}, ' \
          f'{avg_time:.1f}, cont, {args.contamination}'
    print(txt, file=f)
    print(txt)
    f.close()
