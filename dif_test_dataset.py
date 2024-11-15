import os
import pickle
import argparse
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import tree
from sklearn.tree._tree import Tree
#from sklearn import tree
import utils
from config import get_algo_config, get_algo_class
from parser_utils import parser_add_model_argument, update_model_configs
from algorithms import *
from online_dif import OnlineDIF
from online_dif import TreeNode
# class TreeNode:
#     def __init__(self, node_id:int=0,  feature:int=0, threshold:float=0, right_child=None, left_child=None, parent=None) -> None:
#         self.node_id = node_id
#         self.feature = feature
#         self.threshold = threshold
#         self.right_child = right_child
#         self.left_child = left_child
#         self.parent = parent
#         self.visited:bool = False

def data_preprocessing_local(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    x = df.values[:, :-1]
    y = np.array(df.values[:, -1], dtype=int)
    """
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(x)
    x = minmax_scaler.transform(x)
    """
    return x, y

def insert_into_numpy_array(feature_array:np.ndarray, array_size:int, index: int, value: float):
    temp = feature_array[i]
    for i in range(index+1, array_size):
        feature_array[i] = temp
        temp = feature_array[i]

def left_subtree(root:TreeNode, children_left_array, index:int):
    pass

def right_subtree(root:TreeNode, children_right_array, index:int):
    pass

def build_tree(root:TreeNode, left_subtree:np.ndarray, left_subtree_index:int, right_subtree:np.ndarray, right_subtree_index:int, feature:np.ndarray, threshold:np.ndarray):
    if left_subtree[left_subtree_index] != -1:
        #print("yes they are equal")
        left_node = TreeNode()
        left_node.node_id = left_subtree[left_subtree_index]
        left_node.feature = feature[left_node.node_id]
        left_node.threshold = threshold[left_node.node_id]
        root.left_child = left_node
        left_node.parent = root
        build_tree(left_node, left_subtree, left_node.node_id, right_subtree, left_node.node_id, feature, threshold)
    else:
        #print("no they are not equal")
        root.left_child = None
    
    if right_subtree[right_subtree_index] != -1:
        #print("yes they are equal")
        right_node = TreeNode()
        right_node.node_id = right_subtree[right_subtree_index]
        right_node.feature = feature[right_node.node_id]
        right_node.threshold = threshold[right_node.node_id]
        root.right_child = right_node
        right_node.parent = root
        build_tree(right_node, left_subtree, right_node.node_id, right_subtree, right_node.node_id, feature, threshold)
    else:
        #print("no they are not equal")
        root.right_child = None
    
def build_tree_2(root:TreeNode, left_subtree:np.ndarray, right_subtree:np.ndarray, feature:np.ndarray, threshold:np.ndarray) -> int:
    count = 1
    temp = root
    while(True):
        """
        if count == left_subtree.shape[0] or count == right_subtree.shape[0]:
            break   
        """           
        if temp.left_child == None and count < left_subtree.shape[0] and left_subtree[temp.node_id] != -1:        
            left_node = TreeNode()
            left_node.node_id = left_subtree[temp.node_id]
            left_node.feature = feature[temp.node_id]
            left_node.threshold = threshold[temp.node_id]
            left_node.left_child = None
            left_node.right_child = None
            temp.left_child = left_node
            left_node.parent = temp     
            temp = temp.left_child
            count = count + 1
        elif temp.right_child == None and count < right_subtree.shape[0] and right_subtree[temp.node_id] != -1:        
            right_node = TreeNode()
            right_node.node_id = right_subtree[temp.node_id]
            right_node.feature = feature[temp.node_id]
            right_node.threshold = threshold[temp.node_id]
            right_node.left_child = None
            right_node.right_child = None
            temp.right_child = right_node
            right_node.parent = temp      
            temp = temp.right_child
            count = count + 1
        elif temp.parent != None:
            if temp.parent == root:
                if temp == root.left_child:
                    temp = temp.parent
                elif temp == root.right_child:
                    break
            else:
                temp = temp.parent    
        """    
        elif temp.parent == root:
            if temp == root.left_child:
                temp = temp.parent
            elif temp == root.right_child:
                break
        """        
    return count

def traverse_tree(root : TreeNode) -> int:
    count = 1
    temp = root
    while(True):
        if temp.visited == False:
            temp.visited = True
            count = count+1
        if temp.left_child != None and temp.visited == False:
            #count = count + 1
            temp = temp.left_child
        elif temp.right_child != None and temp.visited == False:
            #count = count + 1
            temp = temp.right_child
        elif  temp.parent != None:
            temp = temp.parent
        elif temp.parent == root:
            if temp == root.left_child:
                temp = temp.parent
            elif temp == root.right_child:
                break
        else:
            break
    return count

def traverse_tree_2(root : TreeNode) -> int:
    count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)
    while(True):
        print("node: ", temp.node_id)
        if temp.left_child != None and temp.left_child not in visited_nodes:
            count = count + 1
            temp = temp.left_child
            visited_nodes.append(temp)
            print("node-left: ", temp.node_id)
        elif temp.right_child != None and temp.right_child not in visited_nodes:
            count = count + 1
            temp = temp.right_child
            visited_nodes.append(temp)
            print("node-right: ", temp.node_id)
        elif temp.parent != None:
            if temp.parent == root:
                if temp == root.left_child:
                    temp = temp.parent
                elif temp == root.right_child:
                    break
            else:
                temp = temp.parent
    return (count, visited_nodes)
    #return count

def apply_single_2(root : TreeNode, data: np.ndarray) -> int:
    count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)
    leaf:TreeNode = None
    while(True):
        # print("node: ", temp.node_id)
        # if temp.left_child != None and temp.left_child not in visited_nodes:
        #     count = count + 1
        #     temp = temp.left_child
        #     visited_nodes.append(temp)
        #     print("node-left: ", temp.node_id)
        # elif temp.right_child != None and temp.right_child not in visited_nodes:
        #     count = count + 1
        #     temp = temp.right_child
        #     visited_nodes.append(temp)
        #     print("node-right: ", temp.node_id)
        # elif temp.parent != None:
        #     if temp.parent == root:
        #         if temp == root.left_child:
        #             temp = temp.parent
        #         elif temp == root.right_child:
        #             break
        #     else:
        #         temp = temp.parent
        #if temp.feature
        if temp.feature == -2:
            leaf = temp
            break
        if data[temp.feature] <= temp.threshold:
            if temp.left_child != None:
                temp = temp.left_child
            else:
                leaf = temp
                break
        elif data[temp.feature] > temp.threshold:
            if temp.right_child != None:
                temp = temp.right_child
            else:
                leaf = temp
                break
    #return (count, visited_nodes)
    #return count    
    return leaf.node_id

def apply(root : TreeNode, data: np.ndarray):
    leaves_index = []
    leaf: int
    print("data.shape: ",data.shape[0])
    for i in range(data.shape[0]):
        print(i)
        leaf = apply_single_2(root, data[i])
        leaves_index.append(leaf)
    return leaves_index

def traverse(clf: tree.ExtraTreeRegressor):
    """
    print("---------------<rebuild_tree()-----------------------")
    print("estimator: ", estimator)    
    print("estimator.tree_: ", estimator.tree_)    
    print("estimator.tree_.feature: ", estimator.tree_.feature)    
    print("---------------<rebuild_tree()-----------------------")
    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    t = node_depth, is_leaves
    return t

def apply_single(clf: tree.ExtraTreeRegressor, x: np.ndarray):
    """
    print("---------------<rebuild_tree()-----------------------")
    print("estimator: ", estimator)    
    print("estimator.tree_: ", estimator.tree_)    
    print("estimator.tree_.feature: ", estimator.tree_.feature)    
    print("---------------<rebuild_tree()-----------------------")
    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True
    t = node_depth, is_leaves
    return t

#clf = DIF(device='cpu', max_samples=20, n_ensemble=2)
df = pd.read_csv("data\\tabular\\cricket.csv")
x, y = data_preprocessing_local(df)
clf = IsolationForest(n_estimators=3,
                                max_samples=16,
                                random_state=49)
clf.fit(x)
"""
print("x: ",x)
print("Estimators: ",clf.estimators_)
print("clf.estimator[0]: ",clf.estimators_[0])
print("clf.base_estimator_: ",clf.base_estimator_)
print("clf.estimator[0].tree_: ",clf.estimators_[0].tree_)
print("clf.estimator[0].tree_.node_count: ",clf.estimators_[0].tree_.node_count)
print("clf.clf_lst[0].estimators_[0].tree_.feature: ", clf.estimators_[0].tree_.feature)
print("clf.clf_lst[0].estimators_[0].tree_.threshold: ", clf.estimators_[0].tree_.threshold)
print("clf.clf_lst[0].estimators_[0].tree_.children_left: ", clf.estimators_[0].tree_.children_left)
print("clf.clf_lst[0].estimators_[0].tree_.children_right: ", clf.estimators_[0].tree_.children_right)
"""

"""
tree.plot_tree(clf.estimators_[0])
plt.show()
"""

node_depth, is_leaves = traverse(clf.estimators_[0])
print("node_depth: ", node_depth)
print("is_leaves: ", is_leaves)
leaves_index = clf.estimators_[0].apply(x)
print("leaves_index", leaves_index)
print("leaves_index.shape", leaves_index.shape)
print("x[0]", x[0])
print("x[0][0]", x[0][1])
print("clf.clf_lst[0].estimators_[0].tree_.children_left: ", clf.estimators_[0].tree_.children_left)
print("clf.clf_lst[0].estimators_[0].tree_.children_right: ", clf.estimators_[0].tree_.children_right)
print("clf.clf_lst[0].estimators_[0].tree_.children_left.shape[0]: ", clf.estimators_[0].tree_.children_left.shape[0])
print("clf.clf_lst[0].estimators_[0].tree_.children_left[0]: ", clf.estimators_[0].tree_.children_left[0])
print("type of clf.clf_lst[0].estimators_[0].tree_.children_left: ", type(clf.estimators_[0].tree_.children_left))
print("type of clf.clf_lst[0].estimators_[0].tree_.feature: ", type(clf.estimators_[0].tree_.feature))
print("clf.clf_lst[0].estimators_[0].tree_.feature: ", clf.estimators_[0].tree_.feature)
print("clf.clf_lst[0].estimators_[0].tree_.threshold: ", clf.estimators_[0].tree_.threshold)

root_node = TreeNode()
root_node.node_id = 0
root_node.feature = clf.estimators_[0].tree_.feature[0]
root_node.threshold = clf.estimators_[0].tree_.threshold[0]

#build_tree(root_node, clf.estimators_[0].tree_.children_left, 0, clf.estimators_[0].tree_.children_right, 0, clf.estimators_[0].tree_.feature, clf.estimators_[0].tree_.threshold)
count = build_tree_2(root_node, clf.estimators_[0].tree_.children_left, clf.estimators_[0].tree_.children_right, clf.estimators_[0].tree_.feature, clf.estimators_[0].tree_.threshold)

#count = traverse_tree(root_node)
print("count: ", count)

count, visited_nodes = traverse_tree_2(root_node)
print("count (after traversal):: ", count)
print("visited_nodes:: ", visited_nodes)
print("visited_nodes.len:: ", len(visited_nodes))

print("x: ",x)
print("x[0]: ",x[0])

print("clf_estimators: ", clf.estimators_)
for ii, estimator_tree in enumerate(clf.estimators_):
    print("ii: ", ii)
    print("estimator_tree: ", estimator_tree)

leaf = apply_single_2(root_node, x[15])
print("leaf: ",leaf)
# leaf_scikit = clf.estimators_[0].apply(x[0])
# print("leaf-scikit: ",leaf_scikit)
#leaves_index = clf.estimator_[0].apply(x[0])
#print("leaves_index:", leaves_index)

leaves_index_2 = apply(root_node, x)
print("leaves_index_2: ",leaves_index_2)

o_dif = OnlineDIF(16)
data = np.array([33,0,0])
print("new data::", data)
o_dif.insert_node(root_node, data)
print("-----insert node method executed--------")
o_dif.print_tree(root_node)
"""
node_list = []
for i in range(16):
    node_dict = {'feature': clf.estimators_[0].tree_.feature[i], 'threshold': clf.estimators_[0].tree_.threshold[i]}
    node_list.append(node_dict)

print("node_list: ", node_list)


clf.estimators_[0].tree_.node_count = clf.estimators_[0].tree_.node_count + 1 
modified_features = np.insert(clf.estimators_[0].tree_.feature, 3, 0)
"""

#clf.estimators_[0].tree_.feature[0] = -1 
#clf.estimators_[0].tree_.feature = modified_features
#feature_list = np.ndarray.tolist(clf.estimators_[0].tree_.feature)
#clf.estimators_[0].tree_.feature = np.array(feature_list) 
#feature_list.insert(3, 0)
#feature_array = np.array(feature_list) 

#clf.estimators_[0].tree_.threshold.insert(3, 15)
"""
print("clf.estimator[0].tree_.node_count: ",clf.estimators_[0].tree_.node_count)
print("clf.clf_lst[0].estimators_[0].tree_.feature: ", clf.estimators_[0].tree_.feature)
print("clf.clf_lst[0].estimators_[0].tree_.feature.size: ", clf.estimators_[0].tree_.feature.size)
np.concatenate( (clf.estimators_[0].tree_.feature, np.array([0])), axis=1)
print("clf.clf_lst[0].estimators_[0].tree_.feature.size: ", clf.estimators_[0].tree_.feature.size)
print("clf.clf_lst[0].estimators_[0].tree_.feature: ", clf.estimators_[0].tree_.feature)
#print("feature_array: ", feature_array)
print("clf.clf_lst[0].estimators_[0].tree_.threshold: ", clf.estimators_[0].tree_.threshold)
"""
"""
tree_obj = Tree(0,0,0)
print("custom tree: ",tree_obj)
"""
