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
        self.node_count_ = 0
        self.leaf_count_ = 0

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


    cpdef void update_node_count(self, float[:, :] X):
        """
        Increment each node count if x in X pass through it.
        """

        # In
        cdef SIZE_t n_samples = X.shape[0]

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_
                node.count += 1

                while not node.is_leaf:
                    if X[i, node.feature] <= node.threshold:
                        node = node.left_child
                    else:
                        node = node.right_child

                node.count += 1


    cpdef np.ndarray leaf_path(self, float[:, :] X, bint output, bint weighted):
        """
        Return 2d vector of leaf one-hot encodings, shape=(X.shape[0], no. leaves).
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_leaves = self.leaf_count_
        cdef np.ndarray[float] out = np.zeros((n_samples, n_leaves), dtype=np.float32)
        cdef DTYPE_t val = 1.0

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

                val = 1.0

                if output:
                    val = node.leaf_val

                if weighted:
                    val /= node.count

                out[i][node.leaf_id] = val

        return out


    cpdef np.ndarray feature_path(self, float[:, :] X, bint output, bint weighted):
        """
        Return 2d vector of feature one-hot encodings, shape=(X.shape[0], no. nodes).
        """

        # In / out
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_nodes = self.node_count_
        cdef np.ndarray[float] out = np.zeros((n_samples, n_nodes), dtype=np.float32)
        cdef DTYPE_t val = 1.0

        # Incrementers
        cdef SIZE_t i = 0
        cdef Node*  node = NULL

        with nogil:

            for i in range(n_samples):
                node = self.root_

                while not node.is_leaf:
                    val = 1.0

                    if weighted:
                        val /= node.count

                    out[i][node_id] = val

                    # traverse
                    if X[i, node.feature] <= node.threshold:
                        node = node.left_child
                    else:
                        node = node.right_child

                # leaf
                val = 1.0

                if output:
                    val = node.leaf_val

                if weighted:
                    val /= node.count

                out[i][node.node_id] = val

        return out

    # private
    cdef Node* _add_node(self,
                         SIZE_t node_id,
                         SIZE_t depth,
                         bint   is_left) nogil:
        """
        Pre-order traversal: Recursively create a subtree and return it.
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
            node.leaf_id = self.leaf_count_
            node.leaf_val = self.leaf_vals[node_id]
            node.is_leaf = 1
            self.leaf_count_ += 1

        self.node_count_ += 1

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
        node.leaf_id = -1
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
        node.leaf_id = -1
        node.count = -1
        node.depth = -1
        node.is_left = 0
        node.is_leaf = 0
        node.feature = -1
        node.threshold = -1
        node.leaf_val = -1
        node.left_child = NULL
        node.right_child = NULL

