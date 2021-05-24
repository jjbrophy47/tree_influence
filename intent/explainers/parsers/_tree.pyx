# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
"""
Standardized tree.
"""
cimport cython

from cpython cimport Py_INCREF
from cpython.ref cimport PyObject

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.stdio cimport printf
from libc.math cimport pow

import numpy as np
cimport numpy as np
np.import_array()

cdef class _Tree:

    def __cinit__(self,
                  SIZE_t[:]  children_left,
                  SIZE_t[:]  children_right,
                  SIZE_t[:]  feature,
                  DTYPE_t[:] threshold,
                  DTYPE_t[:] leaf_vals):
        """
        Constructor.
        """
        self.children_left = children_left
        self.children_right = children_right
        self.feature = feature
        self.threshold = threshold
        self.leaf_vals = leaf_vals

        self.root_ = self._add_node(0, 0, 0)

    def __dealloc__(self):
        """
        Destructor.
        """
        if self.root_:
            self._dealloc(self.root_)
            free(self.root_)

    cpdef np.ndarray predict(self, float[:, :] X):
        """
        Predict leaf values for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[float] out = np.zeros((n_samples,), dtype=np.float32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:
                    if X[i, node.feature] <= node.threshold:
                        node = node.left_child
                    else:
                        node = node.right_child

                out[i] = node.leaf_val

        return out

    cpdef np.ndarray apply(self, float[:, :] X):
        """
        Predict leaf index for x in X.
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[int] out = np.zeros((n_samples,), dtype=np.int32)

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:
                    if X[i, node.feature] <= node.threshold:
                        node = node.left_child
                    else:
                        node = node.right_child

                out[i] = node.node_id

        return out

    cpdef SIZE_t get_node_count(self):
        """
        Get total no. nodes.
        """
        return self._get_node_count(self.root_)

    # private
    cdef Node* _add_node(self,
                         SIZE_t node_id,
                         SIZE_t depth,
                         bint   is_left) nogil:
        """
        Recursively create a subtree and return it.
        """
        cdef Node* node = self._initialize_node(node_id, depth, is_left)

        # decision node
        if self.children_left[node_id] != self.children_right[node_id]:
            node.feature = self.feature[node_id]
            node.threshold = self.threshold[node_id]

            if self.children_left[node_id] != -1:
                node.left_child = self._add_node(self.children_left[node_id], depth + 1, 0)

            if self.children_right[node_id] != -1:
                node.right_child = self._add_node(self.children_right[node_id], depth + 1, 1)

        # leaf node
        else:
            node.leaf_val = self.leaf_vals[node_id]
            node.is_leaf = 1

        return node

    cdef Node* _initialize_node(self,
                                SIZE_t node_id,
                                SIZE_t depth,
                                bint   is_left) nogil:
        """
        Create and initialize a new node.
        """
        cdef Node *node = <Node *>malloc(sizeof(Node))
        node.node_id = node_id
        node.count = 0
        node.depth = depth
        node.is_left = is_left
        node.is_leaf = 0
        node.feature = -1
        node.threshold = -1
        node.leaf_val = -1
        node.left_child = NULL
        node.right_child = NULL
        return node

    cdef SIZE_t _get_node_count(self, Node* node) nogil:
        """
        Count total no. of nodes in the tree.
        """
        if not node:
            return 0
        else:
            return 1 + self._get_node_count(node.left_child) + self._get_node_count(node.right_child)


    cdef void _dealloc(self, Node *node) nogil:
        """
        Recursively free all nodes in the subtree.

        NOTE: Does not deallocate "root" node, that must
              be done by the caller!
        """
        if not node:
            return

        # traverse to the bottom nodes first
        self._dealloc(node.left_child)
        self._dealloc(node.right_child)

        # free children
        free(node.left_child)
        free(node.right_child)

        # reset node properties just in case
        node.node_id = -1
        node.count = -1
        node.depth = -1
        node.is_left = 0
        node.is_leaf = 0
        node.feature = -1
        node.threshold = -1
        node.leaf_val = -1
        node.left_child = NULL
        node.right_child = NULL

