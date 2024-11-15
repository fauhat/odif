import numpy as np
import torch
import random
import time
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyGDataLoader
from algorithms import net_torch
from algorithms.dif import DIF
import numpy as np
import dif_test_dataset_2 as dif_util

class CustomDIF(DIF):
    def __init__(self, network_name='mlp', network_class=None,
                n_ensemble=50, n_estimators=6, max_samples=256,
                hidden_dim=[500,100], rep_dim=20, skip_connection=None, dropout=None, activation='tanh',
                data_type='tabular', batch_size=64,
                new_score_func=True, new_ensemble_method=True,
                random_state=42, device='cuda', n_processes=1,
                verbose=0, **network_args):
        super().__init__(network_name, network_class,
                n_ensemble, n_estimators, max_samples,
                hidden_dim, rep_dim, skip_connection, dropout, activation,
                data_type, batch_size,
                new_score_func, new_ensemble_method,
                random_state, device, n_processes,
                verbose, **network_args)

    # def fit(self, X, y=None):
    #     """
    #     Fit detector. y is ignored in unsupervised methods.
    #     Parameters
    #     ----------
    #     X : numpy array of shape (n_samples, n_features)
    #         The input samples.
    #     y : Ignored
    #         Not used, present for API consistency by convention.
    #     Returns
    #     -------
    #     self : object
    #         Fitted estimator.
    #     """
    #     print("--------------------inside CustomDIF------------------------------------------")
    #     start_time = time.time()
    #     self.n_features = X.shape[-1] if self.data_type != 'graph' else max(X.num_features, 1)
    #     ensemble_seeds = np.random.randint(0, 1e+5, self.n_ensemble)

    #     if self.verbose >= 2:
    #         net = self.Net(n_features=self.n_features, **self.network_args)
    #         print(net)

    #     self._training_transfer(X, ensemble_seeds)

    #     if self.verbose >= 2:
    #         it = tqdm(range(self.n_ensemble), desc='clf fitting', ncols=80)
    #     else:
    #         it = range(self.n_ensemble)

    #     for i in it:
    #         self.clf_lst.append(
    #             IsolationForest(n_estimators=self.n_estimators,
    #                             max_samples=self.max_samples,
    #                             random_state=ensemble_seeds[i])
    #         )
    #         self.clf_lst[i].fit(self.x_reduced_lst[i])

    #     if self.verbose >= 1:
    #         print(f'training done, time: {time.time()-start_time:.1f}')
    #     return self

    # def decision_function(self, X):
    #     #OVERRIDDEN
    #     """Predict raw anomaly score of X using the fitted detector.
    #     The anomaly score of an input sample is computed based on different
    #     detector algorithms. For consistency, outliers are assigned with
    #     larger anomaly scores.
    #     Parameters
    #     ----------
    #     X : numpy array of shape (n_samples, n_features)
    #         The training input samples. Sparse matrices are accepted only
    #         if they are supported by the base estimator.
    #     Returns
    #     -------
    #     anomaly_scores : numpy array of shape (n_samples,)
    #         The anomaly score of the input samples.
    #     """

    #     test_reduced_lst = self._inference_transfer(X)
    #     final_scores = self._inference_scoring(test_reduced_lst, n_processes=self.n_processes)
    #     return final_scores

    
    def _cal_score(xx, clf):
        print("------------------In CustomDIF _cal_score method--------------------------")
        depths = np.zeros((xx.shape[0], len(clf.estimators_)))
        depth_sum = np.zeros(xx.shape[0])
        deviations = np.zeros((xx.shape[0], len(clf.estimators_)))
        leaf_samples = np.zeros((xx.shape[0], len(clf.estimators_)))

        max_samples = 256
        for ii, estimator_tree in enumerate(clf.estimators_):
            # estimator_population_ind = sample_without_replacement(n_population=xx.shape[0], n_samples=256,
            #                                                       random_state=estimator_tree.random_state)
            # estimator_population = xx[estimator_population_ind]

            tree = estimator_tree.tree_
            n_node = tree.node_count

            #-----------------part-of-custom-implementation-------------------------------------------------
            root_node = dif_util.TreeNode()
            root_node.node_id = 0
            root_node.feature = clf.estimators_[0].tree_.feature[0]
            root_node.threshold = clf.estimators_[0].tree_.threshold[0]
            count = dif_util.build_tree_2(root_node, tree.children_left, tree.children_right, tree.feature, tree.threshold)
            n_node = count
            #------------------------------------------------------------------------------------------------

            if n_node == 1:
                continue

            # get feature and threshold of each node in the iTree
            # in feature_lst, -2 indicates the leaf node
            feature_lst, threshold_lst = tree.feature.copy(), tree.threshold.copy()

            #----------------------part-of-custom-implementation--------------------------------------------
            feature_lst, threshold_lst = dif_util.get_feature_and_threshold(root_node)    
            #------------------------------------------------------------------------------------------------

            #     feature_lst = np.zeros(n_node, dtype=int)
            #     threshold_lst = np.zeros(n_node)
            #     for j in range(n_node):
            #         feature, threshold = tree.feature[j], tree.threshold[j]
            #         feature_lst[j] = feature
            #         threshold_lst[j] = threshold
            #         # print(j, feature, threshold)
            #         if tree.children_left[j] == -1:
            #             leaf_node_list.append(j)

            # compute depth and score
            leaves_index = estimator_tree.apply(xx)
            node_indicator = estimator_tree.decision_path(xx)

            #----------------------part-of-custom-implementation--------------------------------------------
            leaves_index = dif_util.apply(root_node, xx)
            node_indicator = dif_util.decision_path(root_node, n_node, xx)
            #-----------------------------------------------------------------------------------------------

            # The number of training samples in each test sample leaf
            n_node_samples = estimator_tree.tree_.n_node_samples

            # node_indicator is a sparse matrix with shape (n_samples, n_nodes), indicating the path of input data samples
            # each layer would result in a non-zero element in this matrix,
            # and then the row-wise summation is the depth of data sample
            n_samples_leaf = estimator_tree.tree_.n_node_samples[leaves_index]
            #----------------------part-of-custom-implementation--------------------------------------------
            n_node_samples = dif_util.get_n_node_samples(root_node, n_node, xx)
            n_samples_leaf = n_node_samples[leaves_index]
            #-----------------------------------------------------------------------------------------------
            d = (np.ravel(node_indicator.sum(axis=1)) + super._average_path_length(n_samples_leaf) - 1.0)
            depths[:, ii] = d
            depth_sum += d

            # decision path of data matrix XX
            node_indicator = np.array(node_indicator.todense())

            # set a matrix with shape [n_sample, n_node], representing the feature value of each sample on each node
            # set the leaf node as -2
            value_mat = np.array([xx[i][feature_lst] for i in range(xx.shape[0])])
            value_mat[:, np.where(feature_lst == -2)[0]] = -2
            th_mat = np.array([threshold_lst for _ in range(xx.shape[0])])

            mat = np.abs(value_mat - th_mat) * node_indicator

            # dev_mat = np.abs(value_mat - th_mat)
            # m = np.mean(dev_mat, axis=0)
            # s = np.std(dev_mat, axis=0)
            # dev_mat_mean = np.array([m for _ in range(xx.shape[0])])
            # dev_mat_std = np.array([s for _ in range(xx.shape[0])])
            # dev_mat_zscore = np.maximum((dev_mat - dev_mat_mean) / (dev_mat_std+1e-6), 0)
            # mat = dev_mat_zscore * node_indicator

            exist = (mat != 0)
            dev = mat.sum(axis=1)/(exist.sum(axis=1)+1e-6)
            deviations[:, ii] = dev

            # # slow implementation of deviation calculation
            # t1 = time.time()
            # # calculate deviation in each node of the path
            # # node_deviation_matrix = np.full([xx.shape[0], node_indicator.shape[1]], np.nan)
            # for j in range(xx.shape[0]):
            #     node = np.where(node_indicator[j] == 1)[0]
            #     this_feature_lst = feature_lst[node]
            #     this_threshold_lst = threshold_lst[node]
            #     n_samples_lst = n_node_samples[node]
            #     leaf_samples[j][ii] = n_samples_lst[-1]
            #
            #     deviation = np.abs(xx[j][this_feature_lst[:-1]] - this_threshold_lst[:-1])
            #     if deviation.shape[0] == 0:
            #         print(this_feature_lst[:-1]);print(feature_lst, n_node)
            #
            #     # # directly use mean
            #     deviation = np.mean(deviation)
            #     deviations[j][ii] = deviation
            # print(2, time.time() - t1)

            # # padding node deviation matrix, and use node mean
            # node_deviation_matrix = pd.DataFrame(node_deviation_matrix)
            # for c in node_deviation_matrix.columns:
            #     node_deviation_matrix[c] = node_deviation_matrix[c].fillna(node_deviation_matrix[c].mean())
            #     if pd.isna(node_deviation_matrix[c].mean()):
            #         node_deviation_matrix.drop(c, axis=1, inplace=True)
            #         # node_deviation_matrix[c] = 0
            # node_deviation_matrix = node_deviation_matrix.values
            # deviations[:, ii] = np.mean(node_deviation_matrix, axis=1)

        scores = 2 ** (-depth_sum / (len(clf.estimators_) * super._average_path_length([clf.max_samples_])))
        #----------------------part-of-custom-implementation-------------------------------------------------
        scores = 2 ** (-depth_sum / (len(clf.estimators_) * super._average_path_length([max_samples])))
        #----------------------------------------------------------------------------------------------------
        deviation = np.mean(deviations, axis=1)
        leaf_sample = (clf.max_samples_ - np.mean(leaf_samples, axis=1)) / clf.max_samples_

        # print()
        # print('s', scores)
        # print(deviation)
        # print(leaf_sample)

        scores = scores * deviation
        return scores
        
