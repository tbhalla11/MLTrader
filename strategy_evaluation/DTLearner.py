""""""
"""  		  	   		 	 	 			  		 			     			  	 
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		 	 	 			  		 			     			  	 
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		  	   		 	 	 			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	 	 			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	 	 			  		 			     			  	 
All Rights Reserved  		  	   		 	 	 			  		 			     			  	 

Template code for CS 4646/7646  		  	   		 	 	 			  		 			     			  	 

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	 	 			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	 	 			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	 	 			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	 	 			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	 	 			  		 			     			  	 
or edited.  		  	   		 	 	 			  		 			     			  	 

We do grant permission to share solutions privately with non-students such  		  	   		 	 	 			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	 	 			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	 	 			  		 			     			  	 
GT honor code violation.  		  	   		 	 	 			  		 			     			  	 

-----do not edit anything above this line---  		  	   		 	 	 			  		 			     			  	 

Student Name: Tucker Balch (replace with your name)  		  	   		 	 	 			  		 			     			  	 
GT User ID: tb34 (replace with your User ID)  		  	   		 	 	 			  		 			     			  	 
GT ID: 900897987 (replace with your GT ID)  		  	   		 	 	 			  		 			     			  	 
"""

import warnings

import numpy as np


class DTLearner(object):

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

    def find_feature_correlation(self, data_x, data_y):
        ## building a function to find the highest correlations between x and y
        max_corr = np.corrcoef(data_x, data_y, rowvar=False)
        corr_values = np.abs(max_corr[:-1, -1])  # Extract correlations with data_y
        return np.argmax(corr_values)

    def build_tree_(self, data_x, data_y):

        if data_x.shape[0] <= self.leaf_size or len(data_x.shape) == 1:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])
        if np.unique(data_y).shape[0] == 1:
            return np.array([[-1, data_y[0], np.nan, np.nan]])

        # covering base case of no unique Y, size 1 and less.

        feature = self.find_feature_correlation(data_x, data_y)
        split_feature = np.median(data_x[:, feature])  ## splitting at median as demonstrated in lecture
        # setting up data for recursive calls
        left_split_feature = data_x[:, feature] <= split_feature
        right_split_feature = data_x[:, feature] > split_feature

        if np.sum(left_split_feature) == 0 or np.sum(right_split_feature) == 0:
            return np.array([[-1, data_y.mean(), np.nan, np.nan]])

        # recursive calls to build trees, left builds all data <= split while right focus on > split val
        left_tree = self.build_tree_(data_x[left_split_feature], data_y[left_split_feature])
        right_tree = self.build_tree_(data_x[right_split_feature], data_y[right_split_feature])

        # root = np.array([[feature, split_feature, 1, left_tree.shape[0] + 1]])
        # return np.vstack((root, left_tree, right_tree))

        if len(left_tree.shape) == 1:
            till_leaf = 2
        else:
            till_leaf = left_tree.shape[0] + 1

        root = np.array([[feature, split_feature, 1, till_leaf]])
        return np.vstack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        # Build training and tree
        self.tree = self.build_tree_(data_x, data_y)

    def query(self, points):
        # generate predictions for all query points
        if self.tree is None:
            return

        output = []
        for point in points:
            curr = 0
            while int(self.tree[curr, 0]) != -1:  ##while we aren't at a leaf node we will keep going down
                # base case
                index = int(self.tree[curr, 0])
                split = float(self.tree[curr, 1])
                left_split = int(self.tree[curr, 3])

                if point[index] <= split:
                    curr = curr + 1  # keep going down left side
                else:
                    curr = curr + left_split  # going to go down right side - this references the
                    ## slide show from class for traversal in array format
            output.append(float(self.tree[curr, 1]))
        print(output)
        return np.array(output)

