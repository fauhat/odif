from sklearn.tree import ExtraTreeRegressor
from algorithms.custom_tree import CustomTree


class CustomExtraTreeRegressor:
    
    def __init__(self,  extra_tree_regressor: ExtraTreeRegressor) -> None:
        self.tree_ = CustomTree()
        self.tree_regressor = extra_tree_regressor
        
    def apply(self, x):
        return self.tree_regressor.apply(x)
    
    def decision_path(self, x):
        return self.tree_regressor.decision_path(x)