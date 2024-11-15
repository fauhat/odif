import os
import pickle
import argparse
import time
import pandas as pd
import numpy as np
import requests
import utils
from config import get_algo_config, get_algo_class
from parser_utils import parser_add_model_argument, update_model_configs
from custom_dif import CustomDIF

class CustomDifInvoker:
    def __init__(self, n_ensemble, n_estimators, n_instances=None):
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        #self.n_instances = n_instances
        self.initialize()
        self.create_output_file()

    def initialize(self):
        dataset_root = 'data'
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--runs", type=int, default=1,
                                 help="how many times we repeat the experiments to obtain the average performance")
        self.parser.add_argument(
            "--input_dir", type=str, default='tabular', help="the path of the data sets")
        self.parser.add_argument("--output_dir", type=str, default='&tabular_record/',
                                 help="the output file path")
        self.parser.add_argument("--dataset", type=str, default='FULL',
                                 help="FULL represents all the csv file in the folder, or a list of data set names split by comma")
        self.parser.add_argument("--model", type=str, default='dif')
        self.parser.add_argument('--contamination', type=float, default=-1,
                                 help='this is used to estimate robustness w.r.t. anomaly contamination')
        self.parser.add_argument('--silent_header', action='store_true')
        self.parser.add_argument('--save_rep', action='store_true')
        self.parser.add_argument('--save_score', action='store_true')
        self.parser.add_argument("--flag", type=str, default='')

        self.parser = parser_add_model_argument(self.parser)
        self.args = self.parser.parse_args()

        os.makedirs(self.args.output_dir, exist_ok=True)
        self.data_lst = utils.get_data_lst(os.path.join(
            dataset_root, self.args.input_dir), self.args.dataset)
        print(os.path.join(dataset_root, self.args.input_dir))
        print(self.data_lst)

        self.model_class = get_algo_class(self.args.model)
        #self.model_class = CustomDIF
        self.model_configs = get_algo_config(self.args.model)
        self.model_configs = update_model_configs(self.args, self.model_configs)
        print('model configs:', self.model_configs)
        print('model_configs.device -->', self.model_configs["device"])
        self.model_configs['device'] = 'cpu'
        print('model_configs.device -->', self.model_configs["device"])
        print('model configs:', self.model_configs)
        # model_configs['n_ensemble'] = 25
        self.model_configs['n_ensemble'] = self.n_ensemble
        self.model_configs['n_estimators'] = self.n_estimators

        cur_time = time.strftime("%m-%d %H.%M.%S", time.localtime())
        self.result_file = os.path.join(
            self.args.output_dir, f'{self.args.model}.{self.args.input_dir}.{self.args.flag}.csv')
        print(self.result_file)

    def create_output_file(self):
        if not self.args.silent_header:
            f = open(self.result_file, 'a')
            print('\n---------------------------------------------------------', file=f)
            print(f'model: {self.args.model}, data dir: {self.args.input_dir}, dataset: {self.args.dataset}, '
                  f'contamination: {self.args.contamination}, {self.args.runs}runs, ', file=f)
            for k in self.model_configs.keys():
                print(
                    f'Parameters,\t [{k}], \t\t  {self.model_configs[k]}', file=f)
            print('---------------------------------------------------------', file=f)
            print('data, auc-roc, std, auc-pr, std, time', file=f)
            f.close()
            print("model: "+self.args.model)

    def run(self):
        for f in self.data_lst:
            if f.endswith('pkl'):
                df = pd.read_pickle(f)
            elif f.endswith('csv'):
                df = pd.read_csv(f)
            else:
                continue
            dataset_name = os.path.splitext(os.path.split(f)[1])[0]
            x, y = utils.data_preprocessing(df)

            # # Experiment: robustness w.r.t. different anomaly contamination rate
            if self.args.contamination != -1:
                x_train, y_train = utils.adjust_contamination(x, y,
                                                            contamination_r=self.args.contamination,
                                                            swap_ratio=0.5,
                                                            random_state=2021)
            else:
                x_train, y_train = x, y

            self.args.runs = 1   #newly added by fauhat on 1st September Friday
            auc_lst, ap_lst = np.zeros(self.args.runs), np.zeros(self.args.runs),
            clf_lst = []
            t1_lst = np.zeros(self.args.runs)
            print("args.run: ",self.args.runs)
            print("t1_lst initialized: ", t1_lst)
            for i in range(self.args.runs):
                start_time = time.time()
                print(f'\nRunning [{i+1}/{self.args.runs}] of [{self.args.model}] on Dataset [{dataset_name}]')
                auc_lst[i], ap_lst[i], clf = self.fit_and_compute_score(i, x, y, x_train, dataset_name)           
                t1 = time.time()
                t1_lst[i] = t1 - start_time
                clf_lst.append(clf)
                print(f'{dataset_name}, {auc_lst[i]:.4f}, {ap_lst[i]:.4f}, {t1_lst[i]:.1f}, {self.args.model}')
        
        print("dataset: ", dataset_name)
        print("model-configs: ",self.model_configs)
        # print("clf_lst: ", clf.clf_lst)
        #print("clf_lst[0]: ", clf.clf_lst[0])
        #print("clf_lst[0].estimators: ", clf.clf_lst[0].estimators_)
        #print("clf_lst[0].estimators[0]: ", clf.clf_lst[0].estimators_[0])
        #print("clf_lst[0].estimators[0].tree: ", clf.clf_lst[0].estimators_[0].tree_)
        #print("clf_lst[0].estimators[0].tree :: node_count: ", clf.clf_lst[0].estimators_[0].tree_.node_count)
        # print("clf_lst[0].estimators[0].tree :: feature: ", clf.clf_lst[0].estimators_[0].tree_.feature)
        # print("clf_lst[0].estimators[0].tree :: threshold: ", clf.clf_lst[0].estimators_[0].tree_.threshold)
        #print("clf_lst[0].x_reduced_list shape ", clf.x_reduced_lst[0].shape)
        #print("clf_lst[0].x_reduced_list num cols ", clf.x_reduced_lst.shape[1])
        print("t1_lst.shape: ", t1_lst.shape)
        print("t1_lst: ", t1_lst)
        print("t1_lst[0]: ", t1_lst[0])
        #print("t1_lst[1]: ", t1_lst[1])
        #print("t1_lst[2]: ", t1_lst[2])

        avg_auc, avg_ap = np.average(auc_lst), np.average(ap_lst)
        std_auc, std_ap = np.std(auc_lst), np.std(ap_lst)
        avg_time = np.average(t1_lst)

        f = open(self.result_file, 'a')
        txt = f'{dataset_name}, {avg_auc:.4f}, {std_auc:.4f}, ' \
            f'{avg_ap:.4f}, {std_ap:.4f}, ' \
            f'{avg_time:.1f}, cont, {self.args.contamination}'
        print(txt, file=f)
        print(txt)
        f.close()
        return auc_lst, ap_lst, clf_lst

    def fit_and_compute_score(self, i, x, y, x_train, dataset_name):
        clf = self.model_class(**self.model_configs, random_state=42+i)
        # clf.n_ensemble = 50 #newly added by fauhat on 1st September Friday
        # clf.batch_size = None
        clf.fit(x_train)
        # print("clf_lst: ", clf.clf_lst)
        # t1 = time.time()
        scores = clf.decision_function(x)

        # # ------ significance of synergy: replacing the random representation ensemble ------ # #
        if self.args.save_rep and hasattr(clf, "x_reduced_lst"):
            anom_idx, norm_idx = np.where(y == 1)[0], np.where(y == 0)[0]
            if len(norm_idx) > 1000:
                norm_idx = norm_idx[np.random.RandomState(
                    42).choice(len(norm_idx), 1000, replace=False)]

            new_rep_lst = []
            for x_rep in clf.x_reduced_lst:
                anom, norm = x_rep[anom_idx], x_rep[norm_idx]
                x_rep = np.vstack([anom, norm])
                new_rep_lst.append(x_rep)

            save_dir = '&results_rep_quality_2208/'
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(
                new_rep_lst,
                open(
                    save_dir + f'{dataset_name}_{self.args.model}_reduced_lst_full_anom.pkl', 'wb')
            )

        # # ------ significance of synergy: replacing the isolation-based anomaly scoring  ------ # #
        if self.args.save_score and hasattr(clf, "score_lst"):
            save_dir = '&results_score_quality_2208/'
            os.makedirs(save_dir, exist_ok=True)
            pickle.dump(
                clf.score_lst,
                open(save_dir +
                     f'{dataset_name}_{self.args.model}_score_lst.pkl', 'wb')
            )

        auc, ap = utils.evaluate(y, scores)
        return auc, ap, clf
        #auc_lst[i], ap_lst[i] = auc, ap

        # response = requests.get(instance1_url)
        # print("remote api response: ", response.content)
