import os
ROOT = os.environ['HOME'] + '/python/'


class Sequences:
    def __init__(self,
                 data_label,
                 sequence_length = 30,
                 RANDOM_STATE = 6,
                 train_size = 0.5,
                 validate_size = 0.25,
                 test_size = 0.25):

        assert train_size + validate_size + test_size == 1
        self.header = {'sequence_length': sequence_length,
                       'RANDOM_STATE' : RANDOM_STATE,
                       'data_label' : data_label,
                       'train_size' : train_size,
                       'validate_size' : validate_size,
                       'test_size' : test_size}

        self.X_train = []
        self.y_train = []
        self.X_validate = []
        self.y_validate = []
        self.X_test = []
        self.y_test = []
