import numpy as np
import random as rand

class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.tree = None
        self.verbose = verbose
        self.leaf_size = leaf_size

    def author(self):
            """
            :return: The GT username of the student
            :rtype: str
            """
            return "tbhalla6"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
            return "N/A"


    def find_ran_feature(self, data_x, data_y):
        ## get a random x in data to split on
        return rand.randint(0, data_x.shape[1]- 1)


    def build_tree_(self, data_x, data_y):

        if data_x.shape[0] <= self.leaf_size or len(data_x.shape) == 1:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        if np.unique(data_y).shape[0] == 1:
            return np.array([[-1, data_y[0], np.nan, np.nan]])

        # covering base case of no unique Y, size 1 and less.

        rand_feature = self.find_ran_feature(data_x, data_y)
        split_feature = np.median(data_x[:, rand_feature]) ## splitting at median as demonstrated in lecture
        ## why is this working but /2 is not?
        # setting up data for recursive calls
        left_split_feature = data_x[:, rand_feature] <= split_feature
        right_split_feature = data_x[:, rand_feature] > split_feature

        if np.all(left_split_feature) or np.all(right_split_feature):
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        if np.sum(left_split_feature) == 0 or np.sum(right_split_feature) == 0:
            return np.array([[-1, data_y.mean(), np.nan, np.nan]])

        # recursive calls to build trees, left builds all data <= split while right focus on > split val
        left_tree = self.build_tree_(data_x[left_split_feature],data_y[left_split_feature])
        right_tree = self.build_tree_(data_x[right_split_feature], data_y[right_split_feature])

        # root = np.array([[feature, split_feature, 1, left_tree.shape[0] + 1]])
        # return np.vstack((root, left_tree, right_tree))

        if len(left_tree.shape) == 1:
            till_leaf = 2
        else:
            till_leaf = left_tree.shape[0] + 1

        root = np.array([[rand_feature, split_feature, 1, till_leaf]])
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.sign(np.mean(data_y)), np.nan, np.nan]])
        return np.vstack((root, left_tree, right_tree))


    def add_evidence(self, data_x, data_y):
        # Build training and tree
        self.tree = self.build_tree_(data_x, data_y)

    def query(self, points):
        # generate predictions for all query points
        output = []
        for point in points:
            curr = 0
            while int(self.tree[curr, 0]) != -1 :##while we aren't at a leaf node we will keep going down
                #base case
                index = int(self.tree[curr, 0])
                split = float(self.tree[curr, 1])
                left_split = int(self.tree[curr, 3])

                if point[index] <= split:
                    curr = curr + 1 # keep going down left side
                else:
                    curr = curr + left_split # going to go down right side - this references the
                    ## slide show from class for traversal in array format
            output.append(float(self.tree[curr, 1]))
        return np.array(output)

