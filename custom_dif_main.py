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
from dif_invoker_2 import DifInvoker
from custom_dif_invoker import CustomDifInvoker

#print("------------In main_4----------------------")
# #dif_inovoker = DifInvoker(6, 3)
# # dist_auc_lst = []
# # dist_ap_lst = []
# dist_auc_lst = np.zeros(2)
# dist_ap_lst = np.zeros(2)
# dif_inovoker_1 = DifInvoker(3, 3)
# auc_lst, ap_lst, clf_lst = dif_inovoker_1.run()
# # dist_auc_lst.append(np.average(auc_lst))
# # dist_ap_lst.append(np.average(ap_lst))
# dist_auc_lst[0] = np.average(auc_lst)
# dist_ap_lst[0] = np.average(ap_lst)
# dif_inovoker_2 = DifInvoker(3, 3)
# auc_lst, ap_lst, clf_lst = dif_inovoker_2.run()
# #print("auc_lst, ap_lst, clf_lst: ", auc_lst, ap_lst, clf_lst)
# # dist_auc_lst.append(np.average(auc_lst))
# # dist_ap_lst.append(np.average(ap_lst))
# dist_auc_lst[1] = np.average(auc_lst)
# dist_ap_lst[1] = np.average(ap_lst)
# #auc_lst, ap_list, clf_lst = dif_inovoker.run()
# #print("auc_lst, ap_lst, clf_lst: ", auc_lst, ap_list, clf_lst)
# print("dist_auc_lst, dist_ap_lst: ", dist_auc_lst, dist_ap_lst)

ensemble_num_lst = [3,5]
n_estimators = 3
dist_auc_lst = np.zeros(2)
dist_ap_lst = np.zeros(2)
for i in range(len(ensemble_num_lst)):
    #dif_inovoker = DifInvoker(ensemble_num_lst[i], n_estimators)
    custom_dif_inovoker = CustomDifInvoker(ensemble_num_lst[i], n_estimators)
    #auc_lst, ap_lst, clf_lst = dif_inovoker.run()
    auc_lst, ap_lst, clf_lst = custom_dif_inovoker.run()
    dist_auc_lst[i] = np.average(auc_lst)
    dist_ap_lst[i] = np.average(ap_lst)

print("dist_auc_lst, dist_ap_lst: ", dist_auc_lst, dist_ap_lst)
avg_auc, avg_ap = np.average(dist_auc_lst), np.average(dist_ap_lst)
std_auc, std_ap = np.std(dist_auc_lst), np.std(dist_ap_lst)
print("avg_auc, avg_ap, std_auc, std_ap", avg_auc,avg_ap,std_auc,std_ap)