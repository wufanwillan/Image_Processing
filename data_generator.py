import h5py
import os
import config
import random
import numpy as np

class DataGenerator(object):
    def __init__(self, fn, batch_size, index_name='train'):
        self.fn = fn
        self.hf = h5py.File(os.path.join(config.get_hdf5_path(), fn), 'r')
        self.batch_size = batch_size
        self.index_name = index_name
        self.data_len = len(self.hf['%s_y' % self.index_name])
        self.indexes = np.arange(self.data_len)
        print 'data generator file', fn, 'data length', self.data_len
        self.intentions = self.hf['%s_I' % self.index_name]
        self.intention_indexes = {}
        for i in xrange(len(config.intention_list)):
            self.intention_indexes[i] = []
        for i in xrange(len(self.intentions)):
            self.intention_indexes[np.argmax(self.intentions[i])].append(i)
        # delete those zero values
        for i in xrange(len(config.intention_list)):
            if len(self.intention_indexes[i]) == 0:
                del self.intention_indexes[i]
        self.current_idx=0

    def get_label_dim(self):
        dim = self.hf['%s_y' % self.index_name][-1].shape[-1]
        print 'label dim', dim
        return dim

    def _inc_idx(self):
        self.current_idx += self.batch_size
        if self.current_idx+self.batch_size >= self.data_len:
            self.current_idx = 0
        return (self.current_idx, self.current_idx+self.batch_size)

    def _sample_idx(self):
        import random
        keys = self.intention_indexes.keys()
        intention_idx = keys[random.randint(0, len(keys)-1)]
        index = self.intention_indexes[intention_idx]
        idx =  [index[i] for i in random.sample(xrange(len(index)), self.batch_size)]
        idx = np.in1d(self.indexes, idx)
        return idx

    def next(self):
        if config.USE_INCREASE:
            low, high = self._inc_idx()
            X = []
            for fid in xrange(config.K_FRAMES):
                X.append(self.hf['%s_X_%s' % (self.index_name, fid)][low:high])
            if config.USE_INTENTION:
                X.append(self.hf['%s_I' % (self.index_name)][low:high])
            X.append(self.hf['%s_y' % (self.index_name)][low:high])
        else:
            idx = self._sample_idx()
            X = []
            for fid in xrange(config.K_FRAMES):
                X.append(self.hf['%s_X_%s' % (self.index_name, fid)][idx, :])
            if config.USE_INTENTION:
                X.append(self.hf['%s_I' % (self.index_name)][idx, :])
            X.append(self.hf['%s_y' % (self.index_name)][idx, :])

        if config.USE_INTENTION:
            y = [np.zeros([self.batch_size, 2])]*len(config.intention_list)
        else:
            y = [np.zeros([self.batch_size, 2])]

        return (X, y)

    def __iter__(self):
        return self

    ## used for keras fit methods, it consumes too much memory for the dummy label for large size data
    ## so it's abandoned
    '''
    def data(self):
        Xs = []
        tXs = []
        max_train_batch = len(self.hf['train_y'])/self.batch_size*self.batch_size
        max_test_batch = len(self.hf['test_y'])/self.batch_size*self.batch_size
        for fid in xrange(config.K_FRAMES):
            Xs.append(self.hf['train_X_%d' % fid][:max_train_batch])
            tXs.append(self.hf['test_X_%d' % fid][:max_test_batch])
        if config.USE_INTENTION:
            Xs.append(self.hf['train_I'][:max_train_batch])
            tXs.append(self.hf['test_I'][:max_test_batch])
        Xs.append(self.hf['train_y'][:max_train_batch])
        tXs.append(self.hf['test_y'][:max_test_batch])
        # dummy label
        y = [np.zeros([max_train_batch, 2])]*len(config.intention_list)
        ty = [np.zeros([max_test_batch, 2])]*len(config.intention_list)
        #y = self.hf['train_y'][:max_train_batch]
        #ty = self.hf['test_y'][:max_test_batch]
        return (Xs, y, tXs, ty)


    def turning_data(self):
        Xs = []
        max_batch = len(self.hf['turn_y'])/self.batch_size*self.batch_size
        for fid in xrange(config.K_FRAMES):
            Xs.append(self.hf['turn_X_%d' % fid][:max_batch])
        if config.USE_INTENTION:
            Xs.append(self.hf['turn_I'][:max_batch])
        Xs.append(self.hf['turn_y'][:max_batch])
        y = [np.zeros([max_batch, 2])]*len(config.intention_list)
        #y = self.hf['turn_y'][:max_batch]
        return (Xs, y)
    '''

if __name__=='__main__':
    dgn = DataGenerator(config.DROP_K_SPLIT_FN, config.BATCH_SIZE, 'train')
