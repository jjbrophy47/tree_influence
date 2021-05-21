"""
Python tree implementation prototype to be rewritten in Cython.

Reference: https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/tree/_tree.pyx
"""
import numpy as np

# cdef struct Node:
#     SIZE_t left_child                    # id of the left child of the node
#     SIZE_t right_child                   # id of the right child of the node
#     SIZE_t feature                       # Feature used for splitting the node
#     DOUBLE_t threshold                   # Threshold value at the node


class Node(object):

    def __init__(self, node_id, left_child=None, right_child=None, feature=-1, threshold=-1, leaf_val=-1, is_leaf=0):
        self.node_id = node_id
        self.left_child = left_child
        self.right_child = right_child
        self.feature = feature
        self.threshold = threshold
        self.leaf_val = leaf_val
        self.is_leaf = is_leaf


class Tree:
    """
    The Tree object is a binary tree structure.
    The tree structure is used for predictions.
    """

    def __init__(self, children_left, children_right, feature, threshold, leaf_vals):
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.leaf_vals = leaf_vals

        self.root_ = self._add_node(node_id=0)


    # private
    def _add_node(self, node_id):
        """
        Recursively create a node and return it.
        """
        node = Node(node_id=node_id)

        if self.children_left[node_id] != self.children_right[node_id]:  # split node
            node.feature = feature[node_id]
            node.threshold = threshold[node_id]

            if children_left[node_id] != -1:
                node.left_child = self._add_node(children_left[node_id])

            if children_right[node_id] != -1:
                node.left_child = self._add_node(children_left[node_id])

        else:  # leaf node
            node = Node(node_id=node_id, leaf_val=leaf_vals[node_id], is_leaf=1)

    def _predict(self, X):
        assert X.ndim == 2

        out = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            node = self.root_

            while not node.is_leaf:
                if X[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right

            out[i] = node.leaf_val

        return out
