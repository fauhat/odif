import os
import pickle
import argparse
import time
from typing import Union
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import tree
from sklearn.tree._tree import Tree
#from sklearn import tree
import utils
# from config import get_algo_config, get_algo_class
# from parser_utils import parser_add_model_argument, update_model_configs
#from algorithms import *
#from online_dif import OnlineDIF
#from online_dif import TreeNode
from sqlalchemy import create_engine, ForeignKey, Column, String, Integer, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
import sqlalchemy_test
from scipy.sparse import csr_array
#from scipy.sparse import csr_matrix
from scipy.sparse import spmatrix
# class TreeNode:
#     def __init__(self, node_id:int=0,  feature:int=0, threshold:float=0, right_child=None, left_child=None, parent=None) -> None:
#         self.node_id = node_id
#         self.feature = feature
#         self.threshold = threshold
#         self.right_child = right_child
#         self.left_child = left_child
#         self.parent = parent
#         self.visited:bool = False

class TreeNode:
    def __init__(self, node_id:int=0,  feature:int=0, threshold:float=0, right_child=None, left_child=None, parent=None) -> None:
        self.node_id:int = node_id
        self.feature:int = feature
        self.threshold:int = threshold
        self.right_child:TreeNode = right_child
        self.left_child:TreeNode = left_child
        self.parent:TreeNode = parent
        self.visited:bool = False

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
    # Base = declarative_base()
    # engine = create_engine("mysql://root:@localhost/dif", echo=True)
    # Session = sessionmaker(bind=sqlalchemy_test.engine)
    # session = Session()
    while(True):
        """
        if count == left_subtree.shape[0] or count == right_subtree.shape[0]:
            break   
        """    
        temp.feature = feature[temp.node_id]
        temp.threshold = threshold[temp.node_id]       
        if temp.left_child == None and count < left_subtree.shape[0] and left_subtree[temp.node_id] != -1:        
            left_node = TreeNode()
            left_node.node_id = left_subtree[temp.node_id]
            # left_node.feature = feature[temp.node_id]
            # left_node.threshold = threshold[temp.node_id]
            left_node.left_child = None
            left_node.right_child = None
            temp.left_child = left_node
            left_node.parent = temp   
            # record = sqlalchemy_test.TreeNode(1,1,temp.node_id,temp.left_child.node_id,temp.right_child.node_id,temp.parent.node_id,temp.feature,temp.threshold) 
            # session.add(record)
            temp = temp.left_child
            count = count + 1
        elif temp.right_child == None and count < right_subtree.shape[0] and right_subtree[temp.node_id] != -1:        
            right_node = TreeNode()
            right_node.node_id = right_subtree[temp.node_id]
            # right_node.feature = feature[temp.node_id]
            # right_node.threshold = threshold[temp.node_id]
            right_node.left_child = None
            right_node.right_child = None
            temp.right_child = right_node
            right_node.parent = temp     
            # record = sqlalchemy_test.TreeNode(1,1,temp.node_id,temp.left_child,temp.right_child,temp.parent,temp.feature,temp.threshold) 
            # session.add(record) 
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
    #session.commit()      
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
    temp:TreeNode = root
    visited_nodes = []
    visited_nodes.append(temp)
    db_inserted_nodes = []
    Session = sessionmaker(bind=sqlalchemy_test.engine)
    session = Session()
    while(True):
        # if temp.node_id not in visited_nodes:
        #     if temp.left_child not
        #     record = sqlalchemy_test.TreeNode(1,1,temp.node_id,temp.left_child,temp.right_child,temp.parent,temp.feature,temp.threshold) 
        #     session.add(record)
        print("node: ", temp.node_id)
        print("feature: ", temp.feature)
        print("threshold: ", temp.threshold)
        if temp not in db_inserted_nodes:
            print("node (to be inserted into db): ", temp.node_id)
            if temp.right_child != None:
                right_child_node_id = temp.right_child.node_id
            else:
                right_child_node_id = -1
            if temp.left_child != None:
                left_child_node_id = temp.left_child.node_id
            else:
                left_child_node_id = -1
            if temp.parent != None:
                parent_node_id = temp.parent.node_id
            else:
                parent_node_id = -1
            record = sqlalchemy_test.TreeNode(1,1,temp.node_id,left_child_node_id,right_child_node_id,parent_node_id,temp.feature,temp.threshold) 
            session.add(record)
            db_inserted_nodes.append(temp)
        if temp.left_child != None and temp.left_child not in visited_nodes:
            # if temp.right_child != None:
            #     right_child_node_id = temp.right_child.node_id
            # else:
            #     right_child_node_id = -1
            # if temp.parent != None:
            #     parent_node_id = temp.parent.node_id
            # else:
            #     parent_node_id = -1
            # record = sqlalchemy_test.TreeNode(1,1,temp.node_id,temp.left_child.node_id,right_child_node_id,parent_node_id,temp.feature,temp.threshold) 
            # session.add(record)            
            count = count + 1
            temp = temp.left_child
            visited_nodes.append(temp)
            print("node-left: ", temp.node_id)
        elif temp.right_child != None and temp.right_child not in visited_nodes:            
            # if temp.left_child != None:
            #     left_child_node_id = temp.left_child.node_id
            # else:
            #     left_child_node_id = -1
            # if temp.parent != None:
            #     parent_node_id = temp.parent.node_id
            # else:
            #     parent_node_id = -1
            # record = sqlalchemy_test.TreeNode(1,1,temp.node_id,left_child_node_id,temp.right_child.node_id,parent_node_id,temp.feature,temp.threshold) 
            # session.add(record)    
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
    session.commit()
    session.close()
    print("db_inserted_nodes: ", db_inserted_nodes)
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

def get_n_node_samples_single(root : Union[TreeNode, None], data: np.ndarray) -> list[int]:
    count = 1
    #temp:TreeNode or None = root
    temp: Union[TreeNode, None] = root
    #visited_nodes = []
    #visited_nodes.append(temp)
    #leaf:TreeNode or None = None
    leaf: Union[TreeNode, None] = None
    n_node_samples_single = []
    n_node_samples_single.append(root_node.node_id)
    # data_point = [] #list of the current data point; i.e. [0,0,0,0...]
    # tree_nodes = [] #list of the visited nodes in tree
    # is_visited = [] #list of 1 or 0s; 1 basically i.e. [1,1,....]
    # node_indicator_single.append(temp.node_id)
    # tree_nodes.append(temp.node_id)
    # data_point.append(data_id)
    # is_visited.append(1)
    while(True):
        # if temp.feature == -2:
        #     leaf = temp
        #     node_indicator_single.append(leaf.node_id)
        #     break
        if data[temp.feature] <= temp.threshold:
            if temp.left_child != None:
                temp = temp.left_child
                n_node_samples_single.append(temp.node_id)
                # tree_nodes.append(temp.node_id)
                # data_point.append(data_id)
                # is_visited.append(1)
            else:
                leaf = temp
                #node_indicator_single.append(leaf.node_id)
                break
        elif data[temp.feature] > temp.threshold:
            if temp.right_child != None:
                temp = temp.right_child
                n_node_samples_single.append(temp.node_id)
                # tree_nodes.append(temp.node_id)
                # data_point.append(data_id)
                # is_visited.append(1)
            else:
                leaf = temp
                #node_indicator_single.append(leaf.node_id)
                break
    #return (count, visited_nodes)
    #return count    
    #node_indicator_single_2 = tree_nodes,data_point,is_visited
    #return node_indicator_single
    #return node_indicator_single_2
    return n_node_samples_single

def get_n_node_samples(root : TreeNode, node_count:int, data: np.ndarray):
    #leaves_index = []
    #node_indicator = []
    n_node_samples_list = [0]*node_count
    visited_nodes = []
    leaf: int
    #print("data.shape: ",data.shape[0])
    #print("------------Printing leaves index for each data point----------")
    # arg_row = [] #list of the current data point; i.e. [0,0,0,0...]
    # arg_column = [] #list of the visited nodes in tree
    # arg_data = [] #list of 1 or 0s; 1 basically i.e. [1,1,....]
    # row_dimension = data.shape[0]
    # column_dimension = node_count
    for i in range(data.shape[0]):
        #print(i)
        #print("data point: ",data[i])
        #visited_nodes = decision_path_single(root, data[i], i)
        visited_nodes = get_n_node_samples_single(root, data[i])
        # if i in visited_nodes:
        #     n_node_samples_list[i] = n_node_samples_list[i] + 1
        for j in range(len(visited_nodes)):
            n_node_samples_list[visited_nodes[j]] = n_node_samples_list[visited_nodes[j]] + 1
    # print("row_dimension: ", row_dimension)
    # print("column_dimension: ", column_dimension)
    # print("arg_row: ", arg_row)
    # print("arg_column: ", arg_column)
    # print("arg_data: ", arg_data)
    # node_indicator = csr_array((arg_data, (arg_row, arg_column)), shape=(row_dimension,column_dimension))
    #return node_indicator
    return np.ndarray((node_count,), buffer=np.array(n_node_samples_list),
           #offset=np.int_().itemsize,
           dtype=int)

def decision_path_single(root : Union[TreeNode, None], data: np.ndarray, data_id:int) -> []:
    count = 1
    #temp:TreeNode or None = root
    temp: Union[TreeNode, None] = root
    #visited_nodes = []
    #visited_nodes.append(temp)
    #leaf:TreeNode or None = None
    leaf: Union[TreeNode, None] = None
    node_indicator_single = []
    data_point = [] #list of the current data point; i.e. [0,0,0,0...]
    tree_nodes = [] #list of the visited nodes in tree
    is_visited = [] #list of 1 or 0s; 1 basically i.e. [1,1,....]
    node_indicator_single.append(temp.node_id)
    tree_nodes.append(temp.node_id)
    data_point.append(data_id)
    is_visited.append(1)
    while(True):
        # if temp.feature == -2:
        #     leaf = temp
        #     node_indicator_single.append(leaf.node_id)
        #     break
        if data[temp.feature] <= temp.threshold:
            if temp.left_child != None:
                temp = temp.left_child
                node_indicator_single.append(temp.node_id)
                tree_nodes.append(temp.node_id)
                data_point.append(data_id)
                is_visited.append(1)
            else:
                leaf = temp
                #node_indicator_single.append(leaf.node_id)
                break
        elif data[temp.feature] > temp.threshold:
            if temp.right_child != None:
                temp = temp.right_child
                node_indicator_single.append(temp.node_id)
                tree_nodes.append(temp.node_id)
                data_point.append(data_id)
                is_visited.append(1)
            else:
                leaf = temp
                #node_indicator_single.append(leaf.node_id)
                break
    #return (count, visited_nodes)
    #return count    
    node_indicator_single_2 = tree_nodes,data_point,is_visited
    #return node_indicator_single
    #return node_indicator_single_2
    return tree_nodes

def decision_path(root : TreeNode, node_count:int, data: np.ndarray):
    #leaves_index = []
    node_indicator = []
    visited_nodes = []
    leaf: int
    #print("data.shape: ",data.shape[0])
    #print("------------Printing leaves index for each data point----------")
    arg_row = [] #list of the current data point; i.e. [0,0,0,0...]
    arg_column = [] #list of the visited nodes in tree
    arg_data = [] #list of 1 or 0s; 1 basically i.e. [1,1,....]
    row_dimension = data.shape[0]
    column_dimension = node_count
    for i in range(data.shape[0]):
        #print(i)
        #print("data point: ",data[i])
        visited_nodes = decision_path_single(root, data[i], i)
        # if len(visited_nodes) > column_dimension:
        #     column_dimension  = len(visited_nodes)
        for j in range(len(visited_nodes)):
           arg_row.append(i)
           arg_column.append(visited_nodes[j])
           arg_data.append(1)
        #print("leaf index: ", leaf)
        #leaves_index.append(leaf)
    # print("row_dimension: ", row_dimension)
    # print("column_dimension: ", column_dimension)
    # print("arg_row: ", arg_row)
    # print("arg_column: ", arg_column)
    # print("arg_data: ", arg_data)
    node_indicator = csr_array((arg_data, (arg_row, arg_column)), shape=(row_dimension,column_dimension))
    return node_indicator

def apply(root : TreeNode, data: np.ndarray):
    leaves_index = []
    leaf: int
    #print("data.shape: ",data.shape[0])
    #print("------------Printing leaves index for each data point----------")
    for i in range(data.shape[0]):
        #print(i)
        #print("data point: ",data[i])
        leaf = apply_single_2(root, data[i])
        #print("leaf index: ", leaf)
        leaves_index.append(leaf)
    leaves_index_ndarray = np.array(leaves_index)
    #return leaves_index
    return leaves_index_ndarray

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

def print_tree(root : TreeNode) -> int:
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

def get_feature_and_threshold(root : TreeNode) -> ([],[]):
    count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)
    feature = []
    feature.insert(0, root.feature)
    threshold = []
    threshold.insert(0, root.threshold)
    while(True):
        #print("node: ", temp.node_id)
        if temp.left_child != None and temp.left_child not in visited_nodes:
            count = count + 1
            temp = temp.left_child
            visited_nodes.append(temp)
            #print("node-left: ", temp.node_id)
            feature.insert(temp.node_id, temp.feature)
            threshold.insert(temp.node_id, temp.threshold)
        elif temp.right_child != None and temp.right_child not in visited_nodes:
            count = count + 1
            temp = temp.right_child
            visited_nodes.append(temp)
            #print("node-right: ", temp.node_id)
            feature.insert(temp.node_id, temp.feature)
            threshold.insert(temp.node_id, temp.threshold)
        elif temp.parent != None:
            if temp.parent == root:
                if temp == root.left_child:
                    temp = temp.parent
                elif temp == root.right_child:
                    break
            else:
                temp = temp.parent
    #feature_ndarray = np.ndarray((len(feature),), buffer=np.array(feature), dtype=int)
    feature_ndarray = np.array(feature)
    #feature_ndarray = np.ndarray(buffer=np.array(feature), dtype=int)
    #threshold_ndarray = np.ndarray((len(threshold),), buffer=np.array(threshold), dtype=int)
    threshold_ndarray = np.array(threshold)
    #threshold_ndarray = np.ndarray(buffer=np.array(threshold), dtype=int)
    #return (feature, threshold)
    return (feature_ndarray, threshold_ndarray)

# def load_tree_from_db():
#     Session = sessionmaker(bind=sqlalchemy_test.engine)
#     session = Session()
#     results = session.query(sqlalchemy_test.TreeNode).all()
#     print("------inside load_tree_from_db()-------")
#     print("results: ", results)
    
#     for node in results:
#         if node.node_id == 0:

#         print("node: ", node.node_id)


#clf = DIF(device='cpu', max_samples=20, n_ensemble=2)
#df = pd.read_csv("data\\tabular\\cricket.csv")
df = pd.read_csv("data-2\\cricket.csv")
x, y = data_preprocessing_local(df)
clf = IsolationForest(n_estimators=3,
                                max_samples=16,
                                random_state=49)
clf.fit(x)

print("x: ",x)
print("Estimators: ",clf.estimators_)
print("clf.estimator[0]: ",clf.estimators_[0])
print("clf.base_estimator_: ",clf.base_estimator_)
print("clf.estimator[0].tree_: ",clf.estimators_[0].tree_)
print("clf.estimator[0].tree_.node_count: ",clf.estimators_[0].tree_.node_count)
print("clf.clf_lst[0].estimators_[0].tree_.feature: ", clf.estimators_[0].tree_.feature)
print("clf.estimators_[0].tree_.threshold: ", clf.estimators_[0].tree_.threshold)
print("type(clf.estimators_[0].tree_.feature): ", type(clf.estimators_[0].tree_.feature))
print("type(clf.estimators_[0].tree_.threshold): ", type(clf.estimators_[0].tree_.threshold))
print("clf.clf_lst[0].estimators_[0].tree_.children_left: ", clf.estimators_[0].tree_.children_left)
print("clf.clf_lst[0].estimators_[0].tree_.children_right: ", clf.estimators_[0].tree_.children_right)
print("clf.max_samples_: ", clf.max_samples_)

"""
tree.plot_tree(clf.estimators_[0])
plt.show()
"""

# node_depth, is_leaves = traverse(clf.estimators_[0])
# print("node_depth: ", node_depth)
# print("is_leaves: ", is_leaves)
# leaves_index = clf.estimators_[0].apply(x)
# print("leaves_index", leaves_index)
# print("leaves_index.shape", leaves_index.shape)
# print("x: ",x)
# node_indicator = clf.estimators_[0].decision_path(x)
# print("clf.estimators_[0].decision_path(x) (aka node_indicator):", node_indicator)
# print("type(node_indicator):", type(node_indicator))
# n_node_samples = clf.estimators_[0].tree_.n_node_samples
# print("n_node_samples:", n_node_samples)
# print("type(n_node_samples):", type(n_node_samples))
# print("n_node_samples.shape[0]:", n_node_samples.shape[0])
# print("n_node_samples.shape:", n_node_samples.shape)

# print("type of node_indicator", type(node_indicator))
# sample_id = 0
# # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
# node_index = node_indicator.indices[
#     node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
# ]
# print("node_index: ", node_index)
# n_node_samples = clf.estimators_[0].tree_.n_node_samples
# print("n_node_samples", n_node_samples)
# print("type of n_node_samples", type(n_node_samples))
# n_samples_leaf = clf.estimators_[0].tree_.n_node_samples[leaves_index]
# print("n_samples_leaf", n_samples_leaf)
# n = node_indicator.sum(axis=1)
# print("node_indicator.sum(axis=1)", n)
# print("x[0]", x[0])
# print("x[0][0]", x[0][1])
# print("clf.clf_lst[0].estimators_[0].tree_.children_left: ", clf.estimators_[0].tree_.children_left)
# print("clf.clf_lst[0].estimators_[0].tree_.children_right: ", clf.estimators_[0].tree_.children_right)
# print("clf.clf_lst[0].estimators_[0].tree_.children_left.shape[0]: ", clf.estimators_[0].tree_.children_left.shape[0])
# print("clf.clf_lst[0].estimators_[0].tree_.children_left[0]: ", clf.estimators_[0].tree_.children_left[0])
# print("type of clf.clf_lst[0].estimators_[0].tree_.children_left: ", type(clf.estimators_[0].tree_.children_left))
# print("type of clf.clf_lst[0].estimators_[0].tree_.feature: ", type(clf.estimators_[0].tree_.feature))
# print("clf.clf_lst[0].estimators_[0].tree_.feature: ", clf.estimators_[0].tree_.feature)
# print("clf.clf_lst[0].estimators_[0].tree_.threshold: ", clf.estimators_[0].tree_.threshold)

root_node = TreeNode()
root_node.node_id = 0
root_node.feature = clf.estimators_[0].tree_.feature[0]
root_node.threshold = clf.estimators_[0].tree_.threshold[0]

#build_tree(root_node, clf.estimators_[0].tree_.children_left, 0, clf.estimators_[0].tree_.children_right, 0, clf.estimators_[0].tree_.feature, clf.estimators_[0].tree_.threshold)
count = build_tree_2(root_node, clf.estimators_[0].tree_.children_left, clf.estimators_[0].tree_.children_right, clf.estimators_[0].tree_.feature, clf.estimators_[0].tree_.threshold)

#count = traverse_tree(root_node)
# print("count after traverse_tree(): ", count)

#------------------------------------TRAVERSE TREE 2--------------------------------------------------
# count, visited_nodes = traverse_tree_2(root_node)
# print("count (after traversal):: ", count)
# print("visited_nodes:: ", visited_nodes)
# print("visited_nodes.len:: ", len(visited_nodes))
#------------------------------------TRAVERSE TREE 2--------------------------------------------------

print("x: ",x)
print("x[0]: ",x[0])

# print("clf_estimators: ", clf.estimators_)
# for ii, estimator_tree in enumerate(clf.estimators_):
#     print("ii: ", ii)
#     print("estimator_tree: ", estimator_tree)

leaf = apply_single_2(root_node, x[15])
print("leaf: ",leaf)



leaves_index_2 = apply(root_node, x)
print("leaves_index_2: ",leaves_index_2)


node_indicator_single = decision_path_single(root_node, x[3], 3)
print("node_indicator_single: ",node_indicator_single)
node_indicator = decision_path(root_node, count, x)
print("node_indicator: ",node_indicator)

n_node_samples_single_list = get_n_node_samples_single(root_node, x[3])
print("n_node_samples_single_list: ",n_node_samples_single_list)
print("len(n_node_samples_single_list): ", len(n_node_samples_single_list))
n_node_samples = get_n_node_samples(root_node, count, x)
print("n_node_samples: ",n_node_samples)

print("--------------Printing Tree----------------")
print_tree(root_node)
#---------------ONLINE DIF--------------------------
# o_dif = OnlineDIF(16)
# data = np.array([33,0,0])
# print("new data::", data)
# o_dif.insert_node(root_node, data)
# print("-----insert node method executed--------")
# o_dif.print_tree(root_node)
#---------------------------------------------------

#load_tree_from_db()

# feature,threshold = get_feature_and_threshold(root_node)
# print("custom feature: ", feature)
# print("custom threshold: ", threshold)

