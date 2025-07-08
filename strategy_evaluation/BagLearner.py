import numpy as np
import random as rand

class BagLearner(object):

    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        self.learner = learner
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.kwargs = kwargs # args should be passed in with learners

        self.learners = []
        # add learners to learner to iterate through later
        for bag in range(self.bags):
            self.learners.append(learner(**kwargs))


    def author(self):
            """
            :return: The GT username of the student
            :rtype: str
            """
            return "tbhalla6"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
            return "N/A"

    def add_evidence(self, data_x, data_y):
        #create a learner for each learner thats passed into the func
        for learner in self.learners:
            indices = np.random.randint(0, data_x.shape[0], data_x.shape[0])
            x = data_x[indices]
            y = data_y[indices]
            learner.add_evidence(x, y)


    def query(self, points):
        res = []
        for i in self.learners:
            res.append(i.query(points)) #generate y for each learner and construct a mean
            #append crashes? Why - this is np not scaler
        return np.mean(res, axis=0)
