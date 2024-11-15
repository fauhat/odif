import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import tree
from sklearn.tree._tree import Tree
from sqlalchemy import create_engine, ForeignKey, Column, String, Integer, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base
import sqlalchemy_test
from online_dif import TreeNode

class ForestBuilder:
    def __init__(self, n_estimators, max_samples, random_state, x:np.ndarray):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.x = x
        self.create_forest(n_estimators, max_samples, random_state,x)
    
    def create_forest(self, n_estimators,max_samples,random_state,x):
        clf = IsolationForest(n_estimators=n_estimators,
                                max_samples=max_samples,
                                random_state=random_state)
        clf.fit(x)
        for i in range(n_estimators):    
            root_node = TreeNode()
            root_node.node_id = 0
            root_node.feature = clf.estimators_[i].tree_.feature[0]
            root_node.threshold = clf.estimators_[i].tree_.threshold[0]
            count = self.build_tree_2(root_node, clf.estimators_[i].tree_.children_left, clf.estimators_[i].tree_.children_right, clf.estimators_[i].tree_.feature, clf.estimators_[i].tree_.threshold)
            self.traverse_tree_2(root_node, i)

    def build_tree_2(self, root:TreeNode, left_subtree:np.ndarray, right_subtree:np.ndarray, feature:np.ndarray, threshold:np.ndarray) -> int:
        count = 1
        temp = root
        # Base = declarative_base()
        # engine = create_engine("mysql://root:@localhost/dif", echo=True)
        Session = sessionmaker(bind=sqlalchemy_test.engine)
        session = Session()
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
        session.commit()      
        return count
    
    def traverse_tree_2(self, root : TreeNode, tree_id:int) -> int:
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
                record = sqlalchemy_test.TreeNode(tree_id,tree_id,temp.node_id,left_child_node_id,right_child_node_id,parent_node_id,temp.feature,temp.threshold) 
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