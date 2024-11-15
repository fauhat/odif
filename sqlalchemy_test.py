from sqlalchemy import create_engine, ForeignKey, Column, String, Integer, CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class TreeNode(Base):
    __tablename__ = "treenode"

    ensemble_id = Column("ensemble_id", Integer, primary_key=True)
    tree_id = Column("tree_id", Integer, primary_key=True)
    node_id = Column("node_id", Integer, primary_key=True)
    left_child = Column("left_child", Integer)
    right_child = Column("right_child", Integer)
    parent = Column("parent", Integer)
    feature = Column("feature", Integer)
    threshold = Column("threshold", Integer)
    

    def __init__(self, ensemble_id, tree_id, node_id, left_child, right_child, parent, feature, threshold):
        self.ensemble_id = ensemble_id
        self.tree_id = tree_id
        self.node_id = node_id
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.parent = parent
    
engine = create_engine("mysql://root:@localhost/dif", echo=True)
#Base.metadata.create_all(bind=engine)