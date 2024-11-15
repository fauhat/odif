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
import random

#--------------------------------This py file is Only for testing--------------------------------------------

# class TreeNode:
#     def __init__(self, node_id:int=0,  feature:int=0, threshold:float=0, right_child=None, left_child=None, parent=None) -> None:
#         self.node_id = node_id
#         self.feature = feature
#         self.threshold = threshold
#         self.right_child = right_child
#         self.left_child = left_child
#         self.parent = parent
#         self.visited:bool = False

MAX_NODE_ALLOWED = 300

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

#written in 26th February; experimental still...
def build_tree_3(root:TreeNode, left_subtree:np.ndarray, right_subtree:np.ndarray, feature:np.ndarray, threshold:np.ndarray) -> int:
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
        print("-----------inside build_tree_3--------")   
        print("temp.feature: ", temp.feature)   
        if temp.left_child == None and count < left_subtree.shape[0] and left_subtree[temp.node_id] != -1:        
            if temp.feature != -2:
                print("temp.feature inside if: ", temp.feature)
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
                count = count + 1
            temp = temp.left_child
            # count = count + 1
        elif temp.right_child == None and count < right_subtree.shape[0] and right_subtree[temp.node_id] != -1:        
            if temp.feature != -2:
                print("temp.feature inside if: ", temp.feature)
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
                count = count + 1
            temp = temp.right_child
            # count = count + 1
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


def get_n_node_samples_2(root : TreeNode, new_data_lst: list[np.ndarray], n_node_samples:np.ndarray):
    for data_array in new_data_lst:
        n_node_samples = np.append(n_node_samples,0)

    for data_array in new_data_lst:
        visited_nodes = get_n_node_samples_single(root, data_array)
        for j in range(len(visited_nodes)):
            n_node_samples[visited_nodes[j]] = n_node_samples[visited_nodes[j]] + 1
    return n_node_samples

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

def print_tree_with_feature(root : TreeNode) -> int:
    count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)
    while(True):
        if temp.left_child != None and temp.right_child != None:
            print("node: ", temp.node_id , " (feature:",temp.feature,")"," (threshold:",temp.threshold,")", " (left_child:",temp.left_child.node_id ,")", " (right_child:",temp.right_child.node_id ,")", " (parent:",temp.parent ,")")
        else:    
            print("node: ", temp.node_id , " (feature:",temp.feature,")"," (threshold:",temp.threshold,")", " (left_child:",temp.left_child ,")", " (right_child:",temp.right_child ,")", " (parent:",temp.parent ,")")
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

def insert_node(root: TreeNode, new_node: TreeNode):
    #count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)
    is_inserted : bool = False
    while(True):
        print("node: ", temp.node_id)
        if temp.left_child != None and temp.left_child not in visited_nodes:
            if temp.node_id == new_node.parent.node_id:
                if new_node.threshold < temp.threshold:
                    new_node.left_child = temp.left_child
                    temp.left_child = new_node
                    is_inserted = True
                elif new_node.threshold > temp.threshold:
                    new_node.right_child = temp.right_child
                    temp.right_child = new_node
                    is_inserted = True
                else:
                    is_inserted = False
                    break
            else:                            
                #count = count + 1
                temp = temp.left_child
                visited_nodes.append(temp)
                print("node-left: ", temp.node_id)
        elif temp.right_child != None and temp.right_child not in visited_nodes:
            if temp.node_id == new_node.parent.node_id:
                if new_node.threshold < temp.threshold:
                    new_node.left_child = temp.left_child
                    temp.left_child = new_node
                    is_inserted = True
                elif new_node.threshold > temp.threshold:
                    new_node.right_child = temp.right_child
                    temp.right_child = new_node
                    is_inserted = True
                else:
                    is_inserted = False
                    break
            else:
                #count = count + 1
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
    #return (count, visited_nodes)
    return is_inserted

#not used
def insert_node_2(root: TreeNode, new_node: TreeNode):
    #count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)
    is_inserted : bool = False 
    previous_threshold_lesser = -1
    previous_threshold_greater = -1
    while(True):
        print("node: ", temp.node_id)
        if temp.left_child != None and temp.left_child not in visited_nodes:
            if new_node.feature == temp.feature:
                if new_node.threshold < temp.threshold:
                    new_node.left_child = temp.left_child
                    temp.left_child = new_node
                    is_inserted = True
                elif new_node.threshold > temp.threshold:
                    new_node.right_child = temp.right_child
                    temp.right_child = new_node
                    is_inserted = True
                else:
                    is_inserted = False
                    break
        elif temp.right_child != None and temp.right_child not in visited_nodes:
            if temp.node_id == new_node.parent.node_id:
                if new_node.threshold < temp.threshold:
                    new_node.left_child = temp.left_child
                    temp.left_child = new_node
                    is_inserted = True
                elif new_node.threshold > temp.threshold:
                    new_node.right_child = temp.right_child
                    temp.right_child = new_node
                    is_inserted = True
                else:
                    is_inserted = False
                    break
            else:
                #count = count + 1
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
    #return (count, visited_nodes)
    return is_inserted


def insert_node_3(root: TreeNode, data: np.ndarray):
    #count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)
    #is_inserted : bool = False 
    # previous_threshold_lesser = -1
    # previous_threshold_greater = -1
    leaf = None
    eligible_nodes = {}
    prev_temp = None
    while(True):
        # if temp.feature == -2:
        #     break
        if data[temp.feature] <= temp.threshold:
            if temp.left_child != None:
                temp = temp.left_child
                if temp.feature == -2:
                    break
                if temp.feature not in eligible_nodes:
                    eligible_nodes[temp.feature] = {'leaf_parent':None, 'leaf': None}
                # prev_temp = eligible_nodes[temp.feature]['leaf_parent']
                # eligible_nodes[temp.feature]['leaf_parent'] = temp
                # eligible_nodes[temp.feature]['leaf'] = prev_temp
                if eligible_nodes[temp.feature]['leaf_parent'] != None:
                    if eligible_nodes[temp.feature]['leaf'] != None:
                        eligible_nodes[temp.feature]['leaf_parent'] = eligible_nodes[temp.feature]['leaf']
                        eligible_nodes[temp.feature]['leaf'] = temp
                    else:
                        eligible_nodes[temp.feature]['leaf'] = temp
                else:
                    eligible_nodes[temp.feature]['leaf_parent'] = temp
            else:
                # leaf = temp
                # if temp.feature not in eligible_nodes:
                #     eligible_nodes[temp.feature] = {'leaf_parent':None, 'leaf': None}
                # eligible_nodes[temp.feature]['leaf'] = temp
                break
        elif data[temp.feature] > temp.threshold:
            if temp.right_child != None:
                temp = temp.right_child
                if temp.feature == -2:
                    break
                if temp.feature not in eligible_nodes:
                    eligible_nodes[temp.feature] = {'leaf_parent':None, 'leaf': None}
                # prev_temp = eligible_nodes[temp.feature]['leaf_parent']
                # eligible_nodes[temp.feature]['leaf_parent'] = temp
                # eligible_nodes[temp.feature]['leaf'] = prev_temp
                if eligible_nodes[temp.feature]['leaf_parent'] != None:
                    if eligible_nodes[temp.feature]['leaf'] != None:
                        eligible_nodes[temp.feature]['leaf_parent'] = eligible_nodes[temp.feature]['leaf']
                        eligible_nodes[temp.feature]['leaf'] = temp
                    else:
                        eligible_nodes[temp.feature]['leaf'] = temp
                else:
                    eligible_nodes[temp.feature]['leaf_parent'] = temp
            else:
                # leaf = temp
                # if temp.feature not in eligible_nodes:
                #     eligible_nodes[temp.feature] = {'leaf_parent':None, 'leaf': None}
                # eligible_nodes[temp.feature]['leaf'] = temp
                break

    return eligible_nodes

#--------------written in 18th March 2024----------------------------------
def get_deviations(root : TreeNode, feature_count: int):
    count = 1
    temp = root
    visited_nodes = []
    #visited_nodes.append(temp)
    threshold_lst = [0]*feature_count
    deviation_lst = [0]*feature_count
    while(True):
        #print("node: ", temp.node_id)
        #print("current-node: ", temp.feature, temp.threshold)
        # if temp not in visited_nodes:
        #     if temp.feature >= 0:
        #         deviation = abs(threshold_lst[temp.feature]-temp.threshold)
        #         if (threshold_lst[temp.feature] == 0 and deviation_lst[temp.feature] == 0) or deviation>deviation_lst[temp.feature] or deviation_lst[temp.feature]==threshold_lst[temp.feature]:
        #             deviation_lst[temp.feature] = deviation
        #         threshold_lst[temp.feature] = temp.threshold    
        #         #visited_nodes.append(temp)
        # else:
        #     if temp == root:
        #         threshold_lst[:] = [0 for _ in threshold_lst]
        if temp.feature >= 0:
                if threshold_lst[temp.feature] != 0:
                    deviation = abs(threshold_lst[temp.feature]-temp.threshold)
                    if deviation>deviation_lst[temp.feature]:
                        deviation_lst[temp.feature] = deviation
                threshold_lst[temp.feature] = temp.threshold    
                #visited_nodes.append(temp)
        if temp == root:
                threshold_lst[:] = [0 for _ in threshold_lst]
        visited_nodes.append(temp)
        if temp.left_child != None and temp.left_child not in visited_nodes:
            count = count + 1
            temp = temp.left_child
            #visited_nodes.append(temp)
            #print("node-left: ", temp.node_id)
        elif temp.right_child != None and temp.right_child not in visited_nodes:
            count = count + 1
            temp = temp.right_child
            #visited_nodes.append(temp)
            #print("node-right: ", temp.node_id)
        elif temp.parent != None:
            if temp.parent == root:
                if temp == root.left_child:
                    temp = temp.parent
                elif temp == root.right_child:
                    break
            else:
                temp = temp.parent
    return deviation_lst
    #return (count, visited_nodes)

def get_insertable_nodes(root: TreeNode, data: np.ndarray):
    #count = 1
    temp = root
    visited_nodes = []
    visited_nodes.append(temp)    
    leaf = None
    insertable_nodes = {}
    prev_temp = None
    while(True):
        if temp.feature == -2:
            break
        if temp.feature not in insertable_nodes:
            insertable_nodes[temp.feature] = {'leaf_parent':None, 'leaf': None}
            # prev_temp = eligible_nodes[temp.feature]['leaf_parent']
            # eligible_nodes[temp.feature]['leaf_parent'] = temp
            # eligible_nodes[temp.feature]['leaf'] = prev_temp
        if insertable_nodes[temp.feature]['leaf_parent'] != None:
            if insertable_nodes[temp.feature]['leaf'] != None:
                insertable_nodes[temp.feature]['leaf_parent'] = insertable_nodes[temp.feature]['leaf']
                insertable_nodes[temp.feature]['leaf'] = temp
            else:
                insertable_nodes[temp.feature]['leaf'] = temp
        else:
            insertable_nodes[temp.feature]['leaf_parent'] = temp

        if data[temp.feature] <= temp.threshold:
            if data[temp.feature] == temp.threshold:
                return None
            if temp.left_child != None:
                temp = temp.left_child
            else:
                break
        elif data[temp.feature] > temp.threshold:
            if temp.right_child != None:
                temp = temp.right_child                
            else:                
                break

    return insertable_nodes

#deprecated
def prune_insertable_nodes(insertable_nodes:dict, input_data:np.ndarray, feature_count:int, diff_threshold:list):
    for i in range(feature_count):
        print(i)
        if i in insertable_nodes and insertable_nodes[i]['leaf'] != None:
            print("input_val: ", input_data[insertable_nodes[i]['leaf'].feature])
            print("diff: ",abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold))
            print("diff_leaf: ", abs(insertable_nodes[i]['leaf'].threshold - input_data[insertable_nodes[i]['leaf'].feature]))
            if abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold) < diff_threshold[i]:
                if abs(insertable_nodes[i]['leaf'].threshold - input_data[insertable_nodes[i]['leaf'].feature]) < diff_threshold[i]:    
                    del insertable_nodes[i]
                else:
                    del insertable_nodes[i]['leaf_parent']
        elif i in insertable_nodes and insertable_nodes[i]['leaf_parent'] != None:
            print("diff_leaf-parent: ", abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]))
            if abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]) < diff_threshold[i]:
                del insertable_nodes[i]

#this one is used
def prune_insertable_nodes_2(insertable_nodes:dict, input_data:np.ndarray, feature_count:int, diff_threshold:list):
    #---------inside prune_insertable_nodes_2 -----------------
    #print the insertable nodes at the beginning
    print("printing the insertable nodes at the beginning:")
    for i in range(feature_count):
        if i in insertable_nodes:
            print(i,"=>", insertable_nodes[i])

    #first prune the leaf_parent node or leaf node based on the following:
    #if data > leaf_parent and data > leaf then prune leaf_parent
    #if data < leaf_parent and data < leaf then prune leaf_parent
    #if  data == leaf_parent node then prune leaf_parent
    #if data == leaf_node then prune leaf_node
    for i in range(feature_count):
        if i in insertable_nodes:
            if (insertable_nodes[i]['leaf_parent'] != None and input_data[i]>insertable_nodes[i]['leaf_parent'].threshold) and (insertable_nodes[i]['leaf'] != None and input_data[i]>insertable_nodes[i]['leaf'].threshold):
                insertable_nodes[i]['leaf_parent'] = None
            elif (insertable_nodes[i]['leaf_parent'] != None and input_data[i]<insertable_nodes[i]['leaf_parent'].threshold) and (insertable_nodes[i]['leaf'] != None and input_data[i]<insertable_nodes[i]['leaf'].threshold):
                insertable_nodes[i]['leaf_parent'] = None
            elif insertable_nodes[i]['leaf_parent'] != None and input_data[i]==insertable_nodes[i]['leaf_parent'].threshold:
                #insertable_nodes[i]['leaf_parent'] = None
                del insertable_nodes[i]
            elif insertable_nodes[i]['leaf'] != None and input_data[i]==insertable_nodes[i]['leaf_parent'].threshold:
                #insertable_nodes[i]['leaf'] = None
                del insertable_nodes[i]

    #print the insertable nodes after the first stage prune
    print("printing the insertable nodes after the first stage prune:")
    for i in range(feature_count):
        if i in insertable_nodes:
            print(i,"=>", insertable_nodes[i])

    for i in range(feature_count):
        print(i)
        if i in insertable_nodes and insertable_nodes[i]['leaf_parent'] != None and insertable_nodes[i]['leaf'] != None:
            print("input_val: ", input_data[insertable_nodes[i]['leaf'].feature])
            print("diff: ",abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold))
            print("diff_leaf: ", abs(insertable_nodes[i]['leaf'].threshold - input_data[i]))
            if abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold) < diff_threshold[i]:
                if abs(insertable_nodes[i]['leaf'].threshold - input_data[i]) < diff_threshold[i]:    
                    del insertable_nodes[i]
                else:
                    #del insertable_nodes[i]['leaf_parent']
                    insertable_nodes[i]['leaf_parent'] = None
        elif i in insertable_nodes and insertable_nodes[i]['leaf_parent'] != None:
            print("diff_leaf-parent: ", abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]))
            if abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]) < diff_threshold[i]:
                del insertable_nodes[i]
        elif i in insertable_nodes and insertable_nodes[i]['leaf'] != None:
            print("diff_leaf-parent: ", abs(insertable_nodes[i]['leaf'].threshold - input_data[i]))
            if abs(insertable_nodes[i]['leaf'].threshold - input_data[i]) < diff_threshold[i]:
                del insertable_nodes[i]

    #print the insertable nodes after the second stage prune
    print("printing the insertable nodes after the second stage prune:")
    for i in range(feature_count):
        if i in insertable_nodes:
            print(i,"=>", insertable_nodes[i])
    #---------prune_insertable_nodes_2 method ends-----------------

def prune_insertable_nodes_3(insertable_nodes:dict, input_data:np.ndarray, feature_count:int, diff_threshold:list, confidence:list):
    #---------inside prune_insertable_nodes_2 -----------------
    
    # confidence = parameters['confidence']
    # eligibility = parameters['eligibility']
    # deviation = parameters['deviation']

    #print the insertable nodes at the beginning
    # print("printing the insertable nodes at the beginning:")
    # for i in range(feature_count):
    #     if i in insertable_nodes:
    #         print(i,"=>", insertable_nodes[i])

    #first prune the leaf_parent node or leaf node based on the following:
    #if data > leaf_parent and data > leaf then prune leaf_parent
    #if data < leaf_parent and data < leaf then prune leaf_parent
    #if  data == leaf_parent node then prune leaf_parent
    #if data == leaf_node then prune leaf_node
    for i in range(feature_count):
        if i in insertable_nodes:
            if (insertable_nodes[i]['leaf_parent'] != None and input_data[i]>insertable_nodes[i]['leaf_parent'].threshold) and (insertable_nodes[i]['leaf'] != None and input_data[i]>insertable_nodes[i]['leaf'].threshold):
                insertable_nodes[i]['leaf_parent'] = None
                node_id = insertable_nodes[i]['leaf'].node_id
                if node_id in confidence:
                    confidence[node_id] = confidence[node_id] + 1
                else:
                    confidence[node_id] = 1
            elif (insertable_nodes[i]['leaf_parent'] != None and input_data[i]<insertable_nodes[i]['leaf_parent'].threshold) and (insertable_nodes[i]['leaf'] != None and input_data[i]<insertable_nodes[i]['leaf'].threshold):
                insertable_nodes[i]['leaf_parent'] = None
                node_id = insertable_nodes[i]['leaf'].node_id
                if node_id in confidence:
                    confidence[node_id] = confidence[node_id] + 1
                else:
                    confidence[node_id] = 1
            elif insertable_nodes[i]['leaf_parent'] != None and input_data[i]==insertable_nodes[i]['leaf_parent'].threshold:
                #insertable_nodes[i]['leaf_parent'] = None
                node_id = insertable_nodes[i]['leaf_parent'].node_id
                if node_id in confidence:
                    confidence[node_id] = confidence[node_id] + 1
                else:
                    confidence[node_id] = 1
                del insertable_nodes[i]
            elif insertable_nodes[i]['leaf'] != None and input_data[i]==insertable_nodes[i]['leaf_parent'].threshold:
                #insertable_nodes[i]['leaf'] = None
                node_id = insertable_nodes[i]['leaf'].node_id
                if node_id in confidence:
                    confidence[node_id] = confidence[node_id] + 1
                else:
                    confidence[node_id] = 1
                del insertable_nodes[i]

    #print the insertable nodes after the first stage prune
    # print("printing the insertable nodes after the first stage prune:")
    # for i in range(feature_count):
    #     if i in insertable_nodes:
    #         print(i,"=>", insertable_nodes[i])

    for i in range(feature_count):
        #print(i)
        if i in insertable_nodes and insertable_nodes[i]['leaf_parent'] != None and insertable_nodes[i]['leaf'] != None:
            # print("input_val: ", input_data[insertable_nodes[i]['leaf'].feature])
            # print("diff: ",abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold))
            # print("diff_leaf: ", abs(insertable_nodes[i]['leaf'].threshold - input_data[i]))
            if abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold) < diff_threshold[i]:
                if abs(insertable_nodes[i]['leaf'].threshold - input_data[i]) < diff_threshold[i]:    
                    del insertable_nodes[i]
                else:
                    #del insertable_nodes[i]['leaf_parent']
                    insertable_nodes[i]['leaf_parent'] = None
                    node_id = insertable_nodes[i]['leaf'].node_id 
        elif i in insertable_nodes and insertable_nodes[i]['leaf_parent'] != None:
            #print("diff_leaf-parent: ", abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]))
            if abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]) < diff_threshold[i]:
                del insertable_nodes[i]
        elif i in insertable_nodes and insertable_nodes[i]['leaf'] != None:
            #print("diff_leaf-parent: ", abs(insertable_nodes[i]['leaf'].threshold - input_data[i]))
            if abs(insertable_nodes[i]['leaf'].threshold - input_data[i]) < diff_threshold[i]:
                del insertable_nodes[i]

    #print the insertable nodes after the second stage prune
    # print("printing the insertable nodes after the second stage prune:")
    # for i in range(feature_count):
    #     if i in insertable_nodes:
    #         print(i,"=>", insertable_nodes[i])
    #---------prune_insertable_nodes_2 method ends-----------------

def get_random_pruned_insertable_node(pruned_insertable_node:dict):
    key_list = list(pruned_insertable_node.keys())
    print("list of keys: ", key_list)   
    key = random.choice(key_list)
    random_node_branch = pruned_insertable_node[key]
    print("random_node_branch: ", random_node_branch)
    print("random_node_branch.keys(): ", list(random_node_branch.keys()))
    keys = list(random_node_branch.keys())
    is_null_node_present = False
    non_null_node_key = None
    for i in keys:
        if random_node_branch[i] == None:
            is_null_node_present = True
        else:
            non_null_node_key = i
    if is_null_node_present:
        print("non_null_node_key: ", non_null_node_key)
        return (random_node_branch, non_null_node_key)
    else:        
        random_key = random.choice(list(random_node_branch.keys()))
        print("random_key: ", random_key)
        return (random_node_branch, random_key)
    
def insert_node_if_applicable(root: TreeNode, data:np.ndarray, feature_count:int, diff_threshold:list):
    print("------------INSIDE METHOD:insert_node_if_applicable-------------------------------------")
    insertable_nodes = get_insertable_nodes(root, data)
    print("data: ",data)
    print("insertable nodes: ", insertable_nodes)
    for i in range(feature_count):
        if i in insertable_nodes:
            if insertable_nodes[i]['leaf_parent'] != None:
                print_node_detail(insertable_nodes[i]['leaf_parent'])
            else:
                print(i," leaf_parent: None")
            print("--------------------------------")
            if insertable_nodes[i]['leaf'] != None:
                print_node_detail(insertable_nodes[i]['leaf'])
            else:
                print(i," leaf: None")
    #prune_insertable_nodes(insertable_nodes, data, feature_count, diff_threshold)
    prune_insertable_nodes_2(insertable_nodes, data, feature_count, diff_threshold)
    random_node_branch, random_key = get_random_pruned_insertable_node(insertable_nodes)
    print("random_node_branch, random_key", random_node_branch, random_key)
    node_to_split = random_node_branch[random_key]
    print_node_detail(node_to_split)
    # if data[node_to_split.feature] <= node_to_split.threshold and node_to_split.left_child != None:
    #     new_node = TreeNode()
    #     new_node.node_id = 17 #to be calculated
    #     new_node.feature = node_to_split.feature
    #     new_node.threshold = data[node_to_split.feature]
    #     prev_left_child = node_to_split.left_child
    #     node_to_split.left_child = new_node

#older version...
def insert_node_if_applicable_2(root: TreeNode, data:np.ndarray, feature_count:int, diff_threshold:list):
    print("------------INSIDE METHOD:insert_node_if_applicable_2-------------------------------------")
    insertable_nodes = get_insertable_nodes(root, data)
    print("data: ",data)
    print("insertable nodes: ", insertable_nodes)
    for i in range(feature_count):
        if i in insertable_nodes:
            if insertable_nodes[i]['leaf_parent'] != None:
                print_node_detail(insertable_nodes[i]['leaf_parent'])
            else:
                print(i," leaf_parent: None")
            print("--------------------------------")
            if insertable_nodes[i]['leaf'] != None:
                print_node_detail(insertable_nodes[i]['leaf'])
            else:
                print(i," leaf: None")
    #prune_insertable_nodes(insertable_nodes, data, feature_count, diff_threshold)
    prune_insertable_nodes_2(insertable_nodes, data, feature_count, diff_threshold)
    print("insertable_nodes after calling prune_insertable_nodes_2: ", insertable_nodes)
    max_deviation = 0
    max_feature = -1
    for i in range(feature_count):
        if i in insertable_nodes:
            if insertable_nodes[i]['leaf_parent']!=None and insertable_nodes[i]['leaf']!=None:
                deviation = abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold)
                if deviation > max_deviation:
                    max_deviation = deviation 
                    max_feature = i
            elif insertable_nodes[i]['leaf_parent']!=None and insertable_nodes[i]['leaf']==None:
                deviation = abs(insertable_nodes[i]['leaf_parent'].threshold - data[i])
                if deviation > max_deviation:
                    max_deviation = deviation
                    max_feature = i
            elif insertable_nodes[i]['leaf_parent']==None and insertable_nodes[i]['leaf']!=None:
                deviation = abs(insertable_nodes[i]['leaf'].threshold - input_data[i])
                if deviation > max_deviation:
                    max_deviation = deviation
                    max_feature = i
    print("max_feature: ",max_feature,"max_deviation: ", max_deviation)
    #insert node to the branch of interest
    branch_of_interest = insertable_nodes[max_feature]
    new_node = TreeNode()
    new_node.node_id = 17 #to be calculated
    new_node.feature = max_feature
    new_node.threshold = data[max_feature]
    # prev_left_child = branch_of_interest.left_child
    # branch_of_interest.left_child = new_node  
    print("branch_of_interest: ", branch_of_interest)
    if branch_of_interest['leaf_parent']!=None and branch_of_interest['leaf']!=None:
         random_key = random.choice(list(branch_of_interest.keys()))
         print("random_key: ", random_key)
    elif branch_of_interest['leaf_parent']!=None:
        if new_node.threshold < branch_of_interest['leaf_parent'].threshold:
            left_child = branch_of_interest['leaf_parent'].left_child
            branch_of_interest['leaf_parent'].left_child = new_node
            new_node.left_child = left_child
        elif new_node.threshold > branch_of_interest['leaf_parent'].threshold:
            right_child = branch_of_interest['leaf_parent'].right_child
            branch_of_interest['leaf_parent'].right_child = new_node
            new_node.right_child = right_child
    elif branch_of_interest['leaf']!=None:
        if new_node.threshold < branch_of_interest['leaf'].threshold:
            left_child = branch_of_interest['leaf'].left_child
            branch_of_interest['leaf'].left_child = new_node
            new_node.left_child = left_child
        elif new_node.threshold > branch_of_interest['leaf'].threshold:
            right_child = branch_of_interest['leaf'].right_child
            branch_of_interest['leaf'].right_child = new_node
            new_node.right_child = right_child

#in use...
def insert_node_if_applicable_3(root: TreeNode, data:np.ndarray, feature_count:int, parameters:dict):
    print("------------INSIDE METHOD:insert_node_if_applicable_2-------------------------------------")
    insertable_nodes = get_insertable_nodes(root, data)
    print("data: ",data)
    print("insertable nodes: ", insertable_nodes)
    for i in range(feature_count):
        if i in insertable_nodes:
            if insertable_nodes[i]['leaf_parent'] != None:
                print_node_detail(insertable_nodes[i]['leaf_parent'])
            else:
                print(i," leaf_parent: None")
            print("--------------------------------")
            if insertable_nodes[i]['leaf'] != None:
                print_node_detail(insertable_nodes[i]['leaf'])
            else:
                print(i," leaf: None")
    #prune_insertable_nodes(insertable_nodes, data, feature_count, diff_threshold)
    prune_insertable_nodes_3(insertable_nodes, data, feature_count, parameters['deviation'], parameters['confidence'])
    print("insertable_nodes after calling prune_insertable_nodes_2: ", insertable_nodes)
    max_deviation = 0
    max_feature = -1
    for i in range(feature_count):
        if i in insertable_nodes:
            if insertable_nodes[i]['leaf_parent']!=None and insertable_nodes[i]['leaf']!=None:
                deviation = abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold)
                if deviation > max_deviation:
                    max_deviation = deviation 
                    max_feature = i
            elif insertable_nodes[i]['leaf_parent']!=None and insertable_nodes[i]['leaf']==None:
                deviation = abs(insertable_nodes[i]['leaf_parent'].threshold - data[i])
                if deviation > max_deviation:
                    max_deviation = deviation
                    max_feature = i
            elif insertable_nodes[i]['leaf_parent']==None and insertable_nodes[i]['leaf']!=None:
                deviation = abs(insertable_nodes[i]['leaf'].threshold - input_data[i])
                if deviation > max_deviation:
                    max_deviation = deviation
                    max_feature = i
    print("max_feature: ",max_feature,"max_deviation: ", max_deviation)
    #insert node to the branch of interest
    branch_of_interest = insertable_nodes[max_feature]
    new_node = TreeNode()
    new_node.node_id = 17 #to be calculated
    new_node.feature = max_feature
    new_node.threshold = data[max_feature]
    # prev_left_child = branch_of_interest.left_child
    # branch_of_interest.left_child = new_node  
    print("branch_of_interest: ", branch_of_interest)
    confidence = parameters['confidence']
    eligibility = parameters['eligibility']
    experience = parameters['experience']
    #deviation = parameters['deviation']
    if branch_of_interest['leaf_parent']!=None and branch_of_interest['leaf']!=None:
         random_key = random.choice(list(branch_of_interest.keys()))
         print("random_key: ", random_key)
         #This if-else block is still untested, please test it and if okay remove this comment
         if new_node.threshold < branch_of_interest[random_key].threshold:
            node_id = branch_of_interest[random_key].node_id
            if node_id in eligibility:
                eligibility[node_id].left_child = eligibility[node_id].left_child + 1
            else:
                 eligibility[node_id] = {"left_child":1, "right_child":0}
                 #eligibility[node_id].left_child = 1
            left_child = branch_of_interest[random_key].left_child
            branch_of_interest[random_key].left_child = new_node
            new_node.left_child = left_child
         elif new_node.threshold > branch_of_interest[random_key].threshold:
            node_id = branch_of_interest[random_key].node_id
            if node_id in eligibility:
                eligibility[node_id].right_child = eligibility[node_id].right_child + 1
            else:
                 eligibility[node_id] = {"left_child":0, "right_child":1}
                 #eligibility[node_id].right_child = 1
            right_child = branch_of_interest[random_key].right_child
            branch_of_interest[random_key].right_child = new_node
            new_node.right_child = right_child
    elif branch_of_interest['leaf_parent']!=None:
        if new_node.threshold < branch_of_interest['leaf_parent'].threshold:
            node_id = branch_of_interest['leaf_parent'].node_id
            if node_id in eligibility:
                eligibility[node_id].left_child = eligibility[node_id].left_child + 1
            else:
                 eligibility[node_id] = {"left_child":1, "right_child":0}
                 #eligibility[node_id].left_child = 1
            left_child = branch_of_interest['leaf_parent'].left_child
            branch_of_interest['leaf_parent'].left_child = new_node
            new_node.left_child = left_child
        elif new_node.threshold > branch_of_interest['leaf_parent'].threshold:
            node_id = branch_of_interest['leaf_parent'].node_id
            if node_id in eligibility:
                eligibility[node_id].right_child = eligibility[node_id].right_child + 1
            else:
                 eligibility[node_id] = {"left_child":0, "right_child":1}
                 #eligibility[node_id].right_child = 1
            right_child = branch_of_interest['leaf_parent'].right_child
            branch_of_interest['leaf_parent'].right_child = new_node
            new_node.right_child = right_child
    elif branch_of_interest['leaf']!=None:
        if new_node.threshold < branch_of_interest['leaf'].threshold:
            node_id = branch_of_interest['leaf'].node_id
            if node_id in eligibility:
                eligibility[node_id].left_child = eligibility[node_id].left_child + 1
            else:
                 eligibility[node_id] = {"left_child":1, "right_child":0}
                 #eligibility[node_id].left_child = 1
            left_child = branch_of_interest['leaf'].left_child
            branch_of_interest['leaf'].left_child = new_node
            new_node.left_child = left_child
        elif new_node.threshold > branch_of_interest['leaf'].threshold:
            node_id = branch_of_interest['leaf'].node_id
            if node_id in eligibility:
                eligibility[node_id].right_child = eligibility[node_id].right_child + 1
            else:
                 eligibility[node_id] = {"left_child":0, "right_child":1}
                 #eligibility[node_id].right_child = 1
            right_child = branch_of_interest['leaf'].right_child
            branch_of_interest['leaf'].right_child = new_node
            new_node.right_child = right_child

#still experimental...
def insert_node_if_applicable_4(root: TreeNode, data:np.ndarray, feature_count:int, parameters:dict, rules:dict, node_id:int) -> bool:
    # print("------------INSIDE METHOD:insert_node_if_applicable_4-------------------------------------")
    
    #newly added on 6th November 2024
    if node_id >= MAX_NODE_ALLOWED:
        return False

    insertable_nodes = get_insertable_nodes(root, data)
    # print("data: ",data)
    # print("insertable nodes: ", insertable_nodes)
    # for i in range(feature_count):
    #     if i in insertable_nodes:
    #         if insertable_nodes[i]['leaf_parent'] != None:
    #             print_node_detail(insertable_nodes[i]['leaf_parent'])
    #         else:
    #             print(i," leaf_parent: None")
    #         print("--------------------------------")
    #         if insertable_nodes[i]['leaf'] != None:
    #             print_node_detail(insertable_nodes[i]['leaf'])
    #         else:
    #             print(i," leaf: None")
    #prune_insertable_nodes(insertable_nodes, data, feature_count, diff_threshold)
    # prune_insertable_nodes_3(insertable_nodes, data, feature_count, parameters['deviation'], parameters['confidence'])
    # print("insertable_nodes after calling prune_insertable_nodes_2: ", insertable_nodes)
    if not insertable_nodes:
        return False
    max_deviation = 0
    max_feature = -1
    for i in range(feature_count):
        if i in insertable_nodes:
            if insertable_nodes[i]['leaf_parent']!=None and insertable_nodes[i]['leaf']!=None:
                deviation = abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold)
                if deviation > max_deviation:
                    max_deviation = deviation 
                    max_feature = i
            elif insertable_nodes[i]['leaf_parent']!=None and insertable_nodes[i]['leaf']==None:
                deviation = abs(insertable_nodes[i]['leaf_parent'].threshold - data[i])
                if deviation > max_deviation:
                    max_deviation = deviation
                    max_feature = i
            elif insertable_nodes[i]['leaf_parent']==None and insertable_nodes[i]['leaf']!=None:
                deviation = abs(insertable_nodes[i]['leaf'].threshold - data[i])
                if deviation > max_deviation:
                    max_deviation = deviation
                    max_feature = i
    # print("max_feature: ",max_feature,"max_deviation: ", max_deviation)
    #insert node to the branch of interest
    branch_of_interest = insertable_nodes[max_feature]
    new_node = TreeNode()
    #new_node.node_id = 17 #to be calculated
    new_node.node_id = node_id #to be calculated
    new_node.feature = max_feature
    new_node.threshold = data[max_feature]
    new_node.left_child = None
    new_node.right_child = None
    new_node.parent = None
    # prev_left_child = branch_of_interest.left_child
    # branch_of_interest.left_child = new_node  
    #------------------PRINTING BRANCH OF INTEREST--------------------------------------------------
    # if branch_of_interest['leaf_parent'] != None:
    #     print("branch_of_interest['leaf_parent']: ", branch_of_interest['leaf_parent'].node_id)
    # else:
    #     print("branch_of_interest['leaf_parent']: ", branch_of_interest['leaf_parent'])
    # if branch_of_interest['leaf'] != None:
    #     print("branch_of_interest['leaf']: ", branch_of_interest['leaf'].node_id)
    # else:
    #     print("branch_of_interest['leaf']: ", branch_of_interest['leaf'])
    #----------------------------------------------------------------------------------------------
    #parameters['experience'] = parameters['experience']+1
    experience = parameters['experience']
    eligibility = parameters['eligibility']
    #experience = parameters['experience']
    deviation = parameters['deviation']
    eligibility_ratio = 0
    isInserted = False
    # print("experience: ", experience," rules[experience_count]: ",rules['experience_count'])
    # print("eligibility_ratio: ", eligibility_ratio," rules['eligibility_ratio']: ",rules['eligibility_ratio'])
    if branch_of_interest['leaf_parent']!=None and branch_of_interest['leaf']!=None:
         random_key = random.choice(list(branch_of_interest.keys()))
        #  print("random_key: ", random_key)
         #This if-else block is still untested, please test it and if okay remove this comment
         if new_node.threshold < branch_of_interest[random_key].threshold:
            node_id = branch_of_interest[random_key].node_id
            if node_id in eligibility:
                eligibility_ratio = eligibility[node_id]['left_child']/experience
                # print("experience: ", experience," rules[experience_count]: ",rules['experience_count'])
                # print("eligibility_ratio: ", eligibility_ratio," rules['eligibility_ratio']: ",rules['eligibility_ratio'])
                if experience>=rules['experience_count'] and eligibility_ratio >= rules['eligibility_ratio']:
                    #print("branch_of_interest: ", branch_of_interest)
                    # print("printing tree starting from branch_of_interest BEFORE inserting new node")
                    # print_tree_with_feature(branch_of_interest[random_key])
                    left_child = branch_of_interest[random_key].left_child
                    branch_of_interest[random_key].left_child = new_node
                    new_node.parent = branch_of_interest[random_key]    #newly added on 23rd March 24
                    new_node.left_child = left_child
                    if left_child != None:
                        left_child.parent = new_node    #newly added on 23rd March 24
                    eligibility[node_id]['left_child'] = 0
                    isInserted = True
                    # print("printing tree starting from branch_of_interest AFTER inserting new node")
                    # print_tree_with_feature(branch_of_interest[random_key])
                else:
                    eligibility[node_id]['left_child'] = eligibility[node_id]['left_child'] + 1    
            else:
                 eligibility[node_id] = {"left_child":1, "right_child":0}
                 #eligibility[node_id].left_child = 1
            # left_child = branch_of_interest[random_key].left_child
            # branch_of_interest[random_key].left_child = new_node
            # new_node.left_child = left_child
         elif new_node.threshold > branch_of_interest[random_key].threshold:
            node_id = branch_of_interest[random_key].node_id
            if node_id in eligibility:
                eligibility_ratio = eligibility[node_id]['right_child']/experience
                # print("experience: ", experience," rules[experience_count]: ",rules['experience_count'])
                # print("eligibility_ratio: ", eligibility_ratio," rules['eligibility_ratio']: ",rules['eligibility_ratio'])
                if experience>=rules['experience_count'] and eligibility_ratio >= rules['eligibility_ratio']:
                    #print("branch_of_interest: ", branch_of_interest)
                    # print("printing tree starting from branch_of_interest BEFORE inserting new node")
                    # print_tree_with_feature(branch_of_interest[random_key])
                    right_child = branch_of_interest[random_key].right_child
                    branch_of_interest[random_key].right_child = new_node
                    new_node.parent = branch_of_interest[random_key]    #newly added on 23rd March 24
                    new_node.right_child = right_child
                    if right_child != None:
                        right_child.parent = new_node    #newly added on 23rd March 24
                    eligibility[node_id]['right_child'] = 0
                    isInserted = True
                    # print("printing tree starting from branch_of_interest AFTER inserting new node")
                    # print_tree_with_feature(branch_of_interest[random_key])
                else:
                    eligibility[node_id]['right_child'] = eligibility[node_id]['right_child'] + 1
            else:
                 eligibility[node_id] = {"left_child":0, "right_child":1}
                 #eligibility[node_id].right_child = 1
            # right_child = branch_of_interest[random_key].right_child
            # branch_of_interest[random_key].right_child = new_node
            # new_node.right_child = right_child
    elif branch_of_interest['leaf_parent']!=None:
        if new_node.threshold < branch_of_interest['leaf_parent'].threshold:
            node_id = branch_of_interest['leaf_parent'].node_id
            if node_id in eligibility:
                eligibility_ratio = eligibility[node_id]['left_child']/experience
                # print("experience: ", experience," rules[experience_count]: ",rules['experience_count'])
                # print("eligibility_ratio: ", eligibility_ratio," rules['eligibility_ratio']: ",rules['eligibility_ratio'])
                if experience>=rules['experience_count'] and eligibility_ratio >= rules['eligibility_ratio']:
                    #print("branch_of_interest: ", branch_of_interest)
                    # print("printing tree starting from branch_of_interest BEFORE inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf_parent'])
                    left_child = branch_of_interest['leaf_parent'].left_child
                    branch_of_interest['leaf_parent'].left_child = new_node
                    new_node.parent = branch_of_interest['leaf_parent']    #newly added on 23rd March 24
                    new_node.left_child = left_child
                    if left_child != None:
                        left_child.parent = new_node    #newly added on 23rd March 24
                    eligibility[node_id]['left_child'] = 0
                    isInserted = True
                    # print("printing tree starting from branch_of_interest AFTER inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf_parent'])
                else:
                    eligibility[node_id]['left_child'] = eligibility[node_id]['left_child'] + 1
            else:
                 eligibility[node_id] = {"left_child":1, "right_child":0}
                 #eligibility[node_id].left_child = 1
            # left_child = branch_of_interest['leaf_parent'].left_child
            # branch_of_interest['leaf_parent'].left_child = new_node
            # new_node.left_child = left_child
        elif new_node.threshold > branch_of_interest['leaf_parent'].threshold:
            node_id = branch_of_interest['leaf_parent'].node_id
            if node_id in eligibility:
                eligibility_ratio = eligibility[node_id]['right_child']/experience
                # print("experience: ", experience," rules[experience_count]: ",rules['experience_count'])
                # print("eligibility_ratio: ", eligibility_ratio," rules['eligibility_ratio']: ",rules['eligibility_ratio'])
                if experience>=rules['experience_count'] and eligibility_ratio >= rules['eligibility_ratio']:
                    #print("branch_of_interest: ", branch_of_interest)
                    # print("printing tree starting from branch_of_interest BEFORE inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf_parent'])
                    right_child = branch_of_interest['leaf_parent'].right_child
                    branch_of_interest['leaf_parent'].right_child = new_node
                    new_node.parent = branch_of_interest['leaf_parent']    #newly added on 23rd March 24
                    new_node.right_child = right_child
                    if right_child != None:
                        right_child.parent = new_node    #newly added on 23rd March 24
                    eligibility[node_id]['right_child'] = 0
                    isInserted = True
                    # print("printing tree starting from branch_of_interest AFTER inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf_parent'])
                else:
                    eligibility[node_id]['right_child'] = eligibility[node_id]['right_child'] + 1
            else:
                 eligibility[node_id] = {"left_child":0, "right_child":1}
                 #eligibility[node_id].right_child = 1
            # right_child = branch_of_interest['leaf_parent'].right_child
            # branch_of_interest['leaf_parent'].right_child = new_node
            # new_node.right_child = right_child
    elif branch_of_interest['leaf']!=None:
        if new_node.threshold < branch_of_interest['leaf'].threshold:
            node_id = branch_of_interest['leaf'].node_id
            if node_id in eligibility:
                eligibility_ratio = eligibility[node_id]['left_child']/experience
                # print("experience: ", experience," rules[experience_count]: ",rules['experience_count'])
                # print("eligibility_ratio: ", eligibility_ratio," rules['eligibility_ratio']: ",rules['eligibility_ratio'])
                if experience>=rules['experience_count'] and eligibility_ratio >= rules['eligibility_ratio']:
                    #print("branch_of_interest: ", branch_of_interest)
                    # print("printing tree starting from branch_of_interest BEFORE inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf'])
                    left_child = branch_of_interest['leaf'].left_child
                    branch_of_interest['leaf'].left_child = new_node
                    new_node.parent = branch_of_interest['leaf']    #newly added on 23rd March 24
                    new_node.left_child = left_child
                    if left_child != None:
                        left_child.parent = new_node    #newly added on 23rd March 24
                    eligibility[node_id]['left_child'] = 0
                    isInserted = True
                    # print("printing tree starting from branch_of_interest AFTER inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf'])
                else:
                    eligibility[node_id]['left_child'] = eligibility[node_id]['left_child'] + 1
            else:
                 eligibility[node_id] = {"left_child":1, "right_child":0}
                 #eligibility[node_id].left_child = 1
            # left_child = branch_of_interest['leaf'].left_child
            # branch_of_interest['leaf'].left_child = new_node
            # new_node.left_child = left_child
        elif new_node.threshold > branch_of_interest['leaf'].threshold:
            node_id = branch_of_interest['leaf'].node_id
            if node_id in eligibility:
                eligibility_ratio = eligibility[node_id]['right_child']/experience
                # print("experience: ", experience," rules[experience_count]: ",rules['experience_count'])
                # print("eligibility_ratio: ", eligibility_ratio," rules['eligibility_ratio']: ",rules['eligibility_ratio'])
                if experience>=rules['experience_count'] and eligibility_ratio >= rules['eligibility_ratio']:
                    #print("branch_of_interest: ", branch_of_interest)
                    # print("printing tree starting from branch_of_interest BEFORE inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf'])
                    right_child = branch_of_interest['leaf'].right_child
                    branch_of_interest['leaf'].right_child = new_node
                    new_node.parent = branch_of_interest['leaf']    #newly added on 23rd March 24
                    new_node.right_child = right_child
                    if right_child != None:
                        right_child.parent = new_node    #newly added on 23rd March 24
                    eligibility[node_id]['right_child'] = 0
                    isInserted = True
                    # print("printing tree starting from branch_of_interest AFTER inserting new node")
                    # print_tree_with_feature(branch_of_interest['leaf'])
                else:
                    eligibility[node_id]['right_child'] = eligibility[node_id]['right_child'] + 1
            else:
                 eligibility[node_id] = {"left_child":0, "right_child":1}
                 #eligibility[node_id].right_child = 1
            # right_child = branch_of_interest['leaf'].right_child
            # branch_of_interest['leaf'].right_child = new_node
            # new_node.right_child = right_child
    return isInserted

def print_node_detail(node: TreeNode):
    print("-----Printing node info------")
    print("node id: ", node.node_id)
    print("feature: ", node.feature)
    print("threshold: ", node.threshold)
    if node.left_child != None:
        print("left child: ", node.left_child.node_id)
    else:
        print("left child: None")
    if node.right_child != None:
        print("right child: ", node.right_child.node_id)
    else:
        print("right child: None")
    if node.parent != None:
        print("parent: ", node.parent.node_id)
    else:
        print("parent: None")
    print("----------------------------")

#------------------------------The online dif algorithm-----------------------------------------------
def   fit_online(i_tree: Tree, online_data:np.ndarray=None) :
    # print("-----------Inside fit_online method-------------")
    # print("online_data.shape: ", online_data.shape)
    # print("i_tree: ", i_tree)
    # print("i_tree.feature: ", i_tree.feature)
    # print("i_tree.threshold[0]: ", i_tree.threshold)
    root_node = TreeNode()
    root_node.node_id = 0
    root_node.feature = i_tree.feature[0]
    root_node.threshold = i_tree.threshold[0]

    #build_tree(root_node, clf.estimators_[0].tree_.children_left, 0, clf.estimators_[0].tree_.children_right, 0, clf.estimators_[0].tree_.feature, clf.estimators_[0].tree_.threshold)
    node_count = build_tree_2(root_node, i_tree.children_left, i_tree.children_right, i_tree.feature, i_tree.threshold)
    #count = build_tree_3(root_node, clf.estimators_[0].tree_.children_left, clf.estimators_[0].tree_.children_right, clf.estimators_[0].tree_.feature, clf.estimators_[0].tree_.threshold)
    # print("------------printing the initial tree-------------------")
    # print_count, visited_nodes = print_tree_with_feature(root_node)
    # print("print count: ", print_count)
    # print("node count:", node_count)
    node_id = node_count
    # print("count: ", node_count)
    # print("node_id: ", node_id)
    n_features = online_data.shape[1]
    #print("feature_count: ", n_features)
    deviation_lst = get_deviations(root_node, n_features)
    # print("deviation_lst: ", deviation_lst)
    # print("len(deviation_lst): ", len(deviation_lst))
    parameters = {}
    parameters['confidence'] = {}
    parameters['eligibility'] = {}
    parameters['experience'] = 0
    parameters['deviation'] = deviation_lst
    rules = {}
    rules['experience_count'] = 20
    rules['eligibility_ratio'] = 0.13
    # print("------Inside the online algo loop-------")
    n_new_inserts = 0
    
    new_data_lst = []
    #---------------------the main part; must be uncommented in case the block is commented out--------------
    for i in range(online_data.shape[0]):
        # print("----inside loop for running online algo------")
        # print("i: ",i)
        # print("iteration: ",i)
        # print("online_data[i]: ",online_data[i])
        is_inserted = insert_node_if_applicable_4(root=root_node, data=online_data[i], feature_count=n_features, parameters=parameters, rules=rules, node_id=node_id)
        parameters['experience'] = parameters['experience']+1
        if is_inserted:
            node_id = node_id + 1
            n_new_inserts = n_new_inserts + 1
            new_data_lst.append(online_data[i])
            # print("INSERTED...")
        # print("is_inserted: ", is_inserted)
        # print("parameters: ", parameters)
    # n_node_samples_original = i_tree.n_node_samples
    # print("n_node_samples_original: ", n_node_samples_original)
    #print("type(n_node_samples_original): ", type(n_node_samples_original))
    # n_node_samples_new = get_n_node_samples_2(root=root_node, new_data_lst=new_data_lst,n_node_samples=n_node_samples_original)
    # print("n_node_samples_new: ", n_node_samples_new)
    #---------------------the main part; must be uncommented in case the block is commented out--------------
    
    #------------------Uncomment for single tree experiment for node count prints------------------------- 
    print("------------printing the tree after online dif run-------------------")
    #print_count, visited_nodes = print_tree_with_feature(root_node)
    #print("print count: ", print_count)
    print("Total inserted: ", n_new_inserts)    
    print("Total node count - offline/before insertion: ", node_count)    
    print("Total node count after inserting new nodes: ", node_id)  
    feature_lst, threshold_lst = get_feature_and_threshold(root_node)  
    # print("feature_lst.shape, threshold_lst.shape (printing from fit_online): ", feature_lst.shape, threshold_lst.shape)
    #------------------------------------------------------------------------------------------------------
    return (root_node, node_id, n_new_inserts, new_data_lst)

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

#--------------------------printing data---------------------------------------------
print("x: ",x)
print("type(x): ", type(x))
print("x[0]: ",x[0])
print("type(x[0]): ", type(x[0]))
print("x[0][0]: ",x[0][0])
print("type(x[0][0]): ", type(x[0][0]))
#-------------------------------------------------------------------------------------
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
#count = build_tree_3(root_node, clf.estimators_[0].tree_.children_left, clf.estimators_[0].tree_.children_right, clf.estimators_[0].tree_.feature, clf.estimators_[0].tree_.threshold)
# print("new count: ", count)
#print_tree(root_node)
print_count,visitor_lst = print_tree_with_feature(root_node)
print("print count: ", print_count)

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

# print("--------------Printing Tree----------------")
# print_tree(root_node)
# new_feature = 0
# new_threshold = 155
# new_node = TreeNode()
# new_node.node_id = 0
# new_node.feature = new_feature
# new_node.threshold = new_threshold
# insert_node(root_node, new_node)
# print("--------------Printing Tree after inserting "+new_threshold+"----------------")
# print_tree(root_node)
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
# print("custom feature.shape: ", feature.shape)
# print("custom threshold.shape: ", threshold.shape)

#---------------------insert data code block 1-------------------------------------------------------
# input_data = x[2]
# print("data: ", input_data)
# insertable_nodes = get_insertable_nodes(root_node, input_data)
# print("eligible_nodes: ", insertable_nodes)
# #print_node_detail(insertable_nodes[0]['leaf_parent'])
# print_node_detail(insertable_nodes[1]['leaf_parent'])
# #print_node_detail(insertable_nodes[1]['leaf'])
# print_node_detail(insertable_nodes[2]['leaf_parent'])
# diff_threshold = [3,3,3]
# feature_count = 3
# print ("range: ", range(feature_count))
# print("list of items: ", list(insertable_nodes.items()))
# prune_insertable_nodes(insertable_nodes, input_data, feature_count, diff_threshold)
#-------------------------------------------------------------------------------------------------

# for i in range(feature_count):
#     print(i)
#     if i in insertable_nodes and insertable_nodes[i]['leaf'] != None:
#         print("input_val: ", input_data[insertable_nodes[i]['leaf'].feature])
#         print("diff: ",abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold))
#         print("diff_leaf: ", abs(insertable_nodes[i]['leaf'].threshold - input_data[insertable_nodes[i]['leaf'].feature]))
#         if abs(insertable_nodes[i]['leaf_parent'].threshold - insertable_nodes[i]['leaf'].threshold) < diff_threshold[i]:
#             if abs(insertable_nodes[i]['leaf'].threshold - input_data[insertable_nodes[i]['leaf'].feature]) < diff_threshold[i]:    
#                 del insertable_nodes[i]
#             else:
#                 del insertable_nodes[i]['leaf_parent']
#     elif i in insertable_nodes and insertable_nodes[i]['leaf_parent'] != None:
#         print("diff_leaf-parent: ", abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]))
#         if abs(insertable_nodes[i]['leaf_parent'].threshold - input_data[i]) < diff_threshold[i]:
#             del insertable_nodes[i]

# print("list of items: ", list(insertable_nodes.items()))
# print("eligible_nodes: ", insertable_nodes)
# random_insertable_node = random.choice(list(insertable_nodes.items()))
# print("random_insertable_node: ", random_insertable_node)
# random_node = random.choice(list(random_insertable_node.keys()))
# print("random_node: ", random_node)

#-------------------------------insert data code block 2-------------------------------------------------
# print("insertable_nodes: ", insertable_nodes)
# random_node_and_random_key_tuple = get_random_pruned_insertable_node(insertable_nodes)
# print("random_node_and_random_key_tuple", random_node_and_random_key_tuple)
#--------------------------------------------------------------------------------------------------------
# key_list = list(insertable_nodes.keys())
# print("list of keys: ", key_list)   
# random_key = random.choice(key_list)
# random_nodes = insertable_nodes[random_key]
# print("random_nodes: ", random_nodes)
# print("random_nodes.keys(): ", list(random_nodes.keys()))
# random_node = random.choice(list(random_nodes.keys()))
# print("random_node: ", random_node)

# input_data = x[0]
# print("x.shape: ", x.shape)
# print("data: ", input_data)
# diff_threshold = [3,3,3]
feature_count = 3
# #insert_node_if_applicable(root_node,input_data,feature_count,diff_threshold)
# insert_node_if_applicable_2(root_node,input_data,feature_count,diff_threshold)
# print_tree_with_feature(root_node)

# parameters = {}
# parameters['confidence'] = {}
# parameters['eligibility'] = {}
# parameters['experience'] = 1
# parameters['deviation'] = [3, 3, 3]
# insert_node_if_applicable_3(root_node,input_data,feature_count,parameters)
# print_tree_with_feature(root_node)
# print("----------printing parameters-------")
# print(parameters)

# deviation_lst = get_deviations(root_node, 3)
# print("deviation_lst", deviation_lst)
# print("deviation_lst[-2]", deviation_lst[-2])

input_data = x[0:10]
print("input_data.shape: ", input_data.shape)
print("input_data: ", input_data)
#print("range(input_data.shape[0]): ", range(input_data.shape[0]))
root_node, node_count, n_new_inserts, new_data_lst = fit_online(clf.estimators_[0].tree_, input_data)
print("node_count: ",node_count)
#print_tree_with_feature(root_node)