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
#from dif_test_dataset import TreeNode

class TreeNode:
    def __init__(self, node_id:int=0,  feature:int=0, threshold:float=0, right_child=None, left_child=None, parent=None) -> None:
        self.node_id:int = node_id
        self.feature:int = feature
        self.threshold:int = threshold
        self.right_child:TreeNode = right_child
        self.left_child:TreeNode = left_child
        self.parent:TreeNode = parent
        self.visited:bool = False

class OnlineDIF:
    def __init__(self, tree_id) -> None:
        self.node_id = 16 + 1

    def update_node_id(self):
        #self.max_node_id = max_node_id
        self.node_id = self.node_id+1

    def new_branch(self)->bool:
        confidence = True
        experience = True
        deviation = True
        split_score = True
        return (confidence and experience and deviation and split_score)
    
    def create_node(self, feature, threshold)->TreeNode:
        node = TreeNode()
        #node.node_id = 0
        node.feature = feature
        node.threshold = threshold
        node.node_id = self.node_id
        self.update_node_id()
        return node
    
    def insert_node(self, root:TreeNode, datapoint:np.ndarray)->bool:
        print("-----------inside insert_node method------------------")
        count = 1
        temp = root
        # visited_nodes = []
        # visited_nodes.append(temp)
        # leaf:TreeNode = None
        is_node_inserted:bool = False
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
                #leaf = temp
                break
            if datapoint[temp.feature] < temp.threshold:
                # if temp.left_child != None:
                #     temp = temp.left_child
                # else:
                #     leaf = temp
                #     break
                if datapoint[temp.left_child.feature] > temp.threshold:
                    insertNode = self.new_branch()
                    if insertNode:
                        new_node = self.create_node(temp.feature, datapoint[temp.feature])
                        new_node.left_child = temp.left_child
                        new_node.left_child.parent = new_node
                        new_node.right_child = None
                        temp.left_child = new_node
                        new_node.parent = temp
                        is_node_inserted = True
                    break
                else:
                    temp =temp.left_child
            elif datapoint[temp.feature] > temp.threshold and datapoint[temp.right_child.feature] < temp.threshold:
                # if temp.right_child != None:
                #     temp = temp.right_child
                # else:
                #     leaf = temp
                #     break
                insertNode = self.new_branch()
                if insertNode:
                    new_node = self.create_node(temp.feature, datapoint[temp.feature])
                    new_node.right_child = temp.right_child
                    new_node.right_child.parent = new_node
                    new_node.left_child = None
                    temp.right_child = new_node
                    new_node.parent = temp
                    is_node_inserted = True
                    break
                else:
                    temp = temp.right_child
            else:
                break
        #return (count, visited_nodes)
        #return count    
        return is_node_inserted
    
    def print_tree(self, root : TreeNode) -> int:
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
    

