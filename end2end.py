import config
import os
import numpy as np
from time import time
from data_generator import DataGenerator
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import Model
from keras import backend as K
from keras.layers.advanced_activations import ELU as ReLU
from keras.regularizers import l2,l1
#from keras.optimizers import SGD
from keras.optimizers import SGD,Adagrad,Adadelta,RMSprop,Adam
from keras.layers import (
    Input,
    BatchNormalization,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout,
    Lambda,
    Reshape,
    AveragePooling2D,
    merge
)

def load_model(model, path):
    print 'load pre-trained model', path
    model.load_weights(path)

def filter_loss_function(args):
    y_pred, y_true, intention=args
    '''
    if config.DIM_ORDER=='tf':
        weight = K.expand_dims(config.IMPORTANCE, -1)
    else:
        weight = config.IMPORTANCE.reshape(-1,1)
    mse = K.mean(K.dot(K.square(y_pred-y_true), weight), axis=-1)
    '''
    mse = K.mean(K.square(y_pred-y_true), axis=-1)
    loss = merge([intention, mse], 'mul')
    return loss

def loss_function(args):
    y_pred, y_true=args
    '''
    if config.DIM_ORDER=='tf':
        weight = K.expand_dims(config.IMPORTANCE, -1)
    else:
        weight = config.IMPORTANCE.reshape(-1,1)
    mse = K.mean(K.dot(K.square(y_pred-y_true), weight), axis=-1)
    '''
    mse = K.mean(K.square(y_pred-y_true), axis=-1)
    return mse

INIT='he_normal'

def ProcessModel(input_shape):
    # process model
    process_input = Input(shape=input_shape)
    if config.FC_DR:
        x = Flatten()(process_input)
        x = Dense(4096, init=INIT, W_regularizer = l2(config.L2_LAMBDA), activation='relu')(x)
        x = Dropout(config.DROPOUT)(x)
        x = Dense(4096, init=INIT, W_regularizer = l2(config.L2_LAMBDA), activation='relu')(x)
        x = Dropout(config.DROPOUT)(x)
    else:
        x = Flatten()(process_input)
        x = Dropout(config.DROPOUT)(x)
    process_model = Model(input=process_input, output=x)
    return process_model

def IntentionModel(input_shape):
    input = Input(shape=input_shape)
    x = Dense(64, init=INIT, W_regularizer = l2(config.L2_LAMBDA), activation='relu')(input)
    x = Dropout(config.DROPOUT)(x)
    model = Model(input=input, output=x)
    return model

def filter_layer(args):
    input, intention_output, intention_input = args
    oups = []
    for _ in xrange(len(config.intention_list)):
        merged = merge([input, intention_output], 'concat')
        oups.append(merged)
    oups = K.stack(oups)
    index = K.argmax(intention_input)
    feats = []
    for i in xrange(config.BATCH_SIZE):
        idx = K.cast(index[i], np.int32)
        feats.append(oups[idx, i, :])
    index_feat = K.stack(feats)
    return index_feat

def build_model(cmd_dim, input_tensor=None):
    config.print_configuration()
    print 'constructing model:', config.NET, ' Mode:', config.MODE
    feat_model = config.Model(include_top=True, weights='imagenet', input_tensor=input_tensor)
    layer_dict = dict([(l.name, l) for l in feat_model.layers])
    inp = feat_model.layers[0].input
    oup = layer_dict[config.FEAT_LAYER].output
    shared_model = Model(input=inp, output=oup)
    feat_shape = shared_model.output_shape[1:]
    print 'feat_shape', feat_shape
    input_shape = feat_model.layers[0].input_shape
    print 'input shape', input_shape

    feats = []
    inps = []

    process_model = ProcessModel(feat_shape)
    for i in xrange(config.K_FRAMES):
        k_input = Input(shape=input_shape[1:])
        k_output = shared_model(k_input)
        feat = process_model(k_output)
	inps.append(k_input)
        feats.append(feat)

    if config.USE_INTENTION:
        if config.USE_DISCRETE_INTENTION:
            intention_input = Input(shape=(config.encoder.n_values_[0],))
            intention_output = IntentionModel((config.encoder.n_values_[0],))(intention_input)
        else:
            intention_input = Input(shape=(config.NUM_INTENTION,))
            intention_output = IntentionModel((config.NUM_INTENTION,))(intention_input)
        inps.append(intention_input)

    if config.K_FRAMES > 1:
        merged = merge(feats, mode='concat')
    else:
        merged = feats[0]

    if config.USE_INTENTION and config.USE_FILTER:
        FEAT_LEN=process_model.output_shape[-1]*config.K_FRAMES + intention_output.get_shape()[-1].value
        filter_feat = Lambda(filter_layer, output_shape=(FEAT_LEN,))([merged, intention_output, intention_input])

    label = Input(shape=(cmd_dim,))
    inps.append(label)
    oups = []
    preds=[]
    if config.USE_INTENTION:
        for i in xrange(len(config.intention_list)):
            if config.USE_FILTER:
                x = Dense(cmd_dim, init=INIT, W_regularizer = l2(config.L2_LAMBDA), activation='linear')(filter_feat)
            else:
                x = merge([merged, intention_output], 'concat')
                x = Dense(cmd_dim, init=INIT, W_regularizer = l2(config.L2_LAMBDA), activation='linear')(x)
            preds.append(x)
            intention_out=Lambda(lambda x : x[:, i])(intention_input)
            loss_out = Lambda(filter_loss_function, output_shape=(1,))([x, label, intention_out])
            oups.append(loss_out)
    else:
        x = Dense(cmd_dim, init=INIT, W_regularizer = l2(config.L2_LAMBDA), activation='linear')(merged)
        preds.append(x)
        loss_out = Lambda(loss_function, output_shape=(1,))([x, label])
        oups.append(loss_out)

    model = Model(input=inps, output=oups)
    print ('constructing model done')
    print 'whole model\n', model.summary()

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_arch_path = os.path.join(current_dir, 'K'+repr(config.K_FRAMES)+'_'+config.get_name()+'_I%s_S%s_F%s.png' % (repr(config.USE_INTENTION), repr(config.USE_INCREASE), repr(config.USE_FILTER)))
    plot(model, to_file=model_arch_path, show_shapes=True)
    model_path = os.path.join(config.get_model_path(config.TASK), config.MODEL_FN)
    # get prediction layers
    print 'learning phase', K.learning_phase()
    inps.append(K.learning_phase())
    prediction = K.function(inps, preds)
    return model, prediction

class ModelSaver(Callback):
    def __init__(self, filepath, skip=1, verbose=0, save_weights_only=False):
        self.verbose = verbose
        self.save_weights_only = save_weights_only
        self.filepath = filepath
	self.skip = skip
	assert skip > 0, 'skip should be non-negative'

    def on_epoch_end(self, epoch, logs={}):
        # monitor learning rate
        optimizer = self.model.optimizer
        pure_lr = optimizer.lr
        lr = K.eval(pure_lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nPure LR: %s LR: %s\n' %(repr(K.eval(pure_lr)), repr(lr)))

	if epoch % self.skip == 0:
	    if self.verbose > 0:
		print('Epoch %05d: ModelSaver: saving model to %s' % (epoch, self.filepath))
	    if self.save_weights_only:
		self.model.save_weights(self.filepath, overwrite=True)
	    else:
		self.model.save(self.filepath, overwrite=True)

def save_history(history):
    import pickle
    import sys
    sys.setrecursionlimit(10000)
    output = open(config.get_history_fn(), 'wb')
    pickle.dump(history.history, output)
    output.close()

def load_history():
    import pickle
    pkl_file = open(config.get_history_fn(), 'rb')
    history = pickle.load(pkl_file)
    return history

def plot_history(history):
    import matplotlib.pyplot as plt
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    '''
    model.layers['pred
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    figure = plt.figure()
    figure.savefig('accuracy_'+config.MODEL_FN)
    '''
    # summarize history for loss
    loss = map(lambda x: x, history.history['loss'])
    val_loss = map(lambda x: x, history.history['val_loss'])
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss_'+config.HISTORY_FN)
    plt.show()

def predict(preds, input):
    idx=0
    if config.USE_INTENTION:
        intention = input[-2]
        idx = np.argmax(intention)
    # keras learning phase
    input.append(0)
    # learning phase for keras
    pred = preds(input)[idx]
    return pred

def train(premodel=True):
    dgn = DataGenerator(config.DROP_K_SPLIT_FN, config.BATCH_SIZE, 'train')
    test_dgn = DataGenerator(config.DROP_K_SPLIT_FN, config.BATCH_SIZE, 'test')

    model, preds = build_model(dgn.get_label_dim())
    best_weights_filepath = os.path.join(config.get_model_path(), config.MODEL_FN)
    if premodel:
        # load pre_trained model
        load_model(model, best_weights_filepath)

    #sgd = SGD(lr=config.LEARNING_RATE, clipnorm=5, decay=20*config.LEARNING_RATE/config.EPOCHS, momentum=0.95, nesterov=True)
    #sgd = SGD(lr=config.LEARNING_RATE, clipnorm=5, decay=0.5, momentum=0.95, nesterov=True)
    sgd = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-8, decay=1e-4)
    #sgd=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    #sgd=Adadelta(lr=1e-1, rho=0.95, epsilon=1e-6)

    losses = []
    if config.USE_INTENTION:
        for i in xrange(len(config.intention_list)):
            losses.append(lambda y_true, y_pred: y_pred)
    else:
        losses.append(lambda y_true, y_pred: y_pred)
    model.compile(loss=losses, optimizer=sgd)

    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    latest_weights_filepath = os.path.join(config.get_model_path(), config.LATEST_FN)
    model_saver_callback = ModelSaver(latest_weights_filepath, skip=config.SAVE_EVERY_OTHER_EPOCH, verbose=1)

    # train model
    #history = model.fit(X, y, shuffle='batch', validation_data=(tX, ty), nb_epoch=config.EPOCHS, batch_size=config.BATCH_SIZE, verbose=1, callbacks=[saveBestModel, model_saver_callback])
    history = model.fit_generator(dgn, validation_data=test_dgn, samples_per_epoch=config.EPOCH_SIZE, nb_val_samples=config.VALIDATE_SIZE, nb_epoch=config.EPOCHS, verbose=1, callbacks=[saveBestModel, model_saver_callback])
    save_history(history)

    # plot history
    plot_history(history)
    #reload best weights
    load_model(model, best_weights_filepath)
    return model

def main(pre_model=False):
    # train
    model = train(pre_model)
    # test
    #print 'pred', model.predict(np.array([X[0]])), 'y', y[0]

def test():
    config.BATCH_SIZE=1
    model, preds = build_model(config.CMD_DIM)
    fn = os.path.join(config.get_model_path(), config.MODEL_FN)
    #fn = os.path.join(config.get_model_path(), config.LATEST_FN)
    model.load_weights(fn)

    dgn = DataGenerator(config.DROP_K_SPLIT_FN, config.BATCH_SIZE)
    X,y,tX,ty = dgn.next()
    start = time()
    for i in xrange(min(50, len(X[0]))):
        tmpX = [np.expand_dims(t[i], axis=0) for t in X]
        '''
        import cv2
	data = tmpX[0][0]#.swapaxes(0,2)
        cv2.imshow('test', data)
        cv2.waitKey(0)
        '''
        print '\n\n#########################'
        print 'loss', model.predict(tmpX)
        print 'truth', tmpX[-1:]
        print 'pred', predict(preds, tmpX)[0]
        print '#########################'
    end = time()
    print 'time', end-start

def test_turning():
    config.BATCH_SIZE=1
    model, preds = build_model(config.CMD_DIM)
    fn = os.path.join(config.get_model_path(), config.MODEL_FN)
    #fn = os.path.join(config.get_model_path(), config.LATEST_FN)
    model.load_weights(fn)

    dgn = DataGenerator(config.TURNING_FN, config.BATCH_SIZE)
    tX,ty = dgn.next()
    start = time()
    for i in xrange(10):
        print 'len tX', len(tX), tX[0].shape
        tmpX = [np.expand_dims(t[i], axis=0) for t in tX]
        '''
        import cv2
	data = tmpX[0][0]#.swapaxes(0,2)
        cv2.imshow('test', data)
        cv2.waitKey(0)
        '''
        print '\n\n#########################'
        print 'loss', model.predict(tmpX)
        print 'truth', tmpX[-1]
        print 'pred', predict(preds, tmpX)[0]
        print '#########################'
    end = time()
    print 'time', end-start

def get_heatmap():
    config.BATCH_SIZE=1
    model, preds = build_model(config.CMD_DIM)
    fn = os.path.join(config.get_model_path(), config.MODEL_FN)
    #fn = os.path.join(config.get_model_path(), config.LATEST_FN)
    model.load_weights(fn)

def test_plot_history():
    history = load_history()
    from munch import Munch
    hist = Munch(history = history)
    print hist
    plot_history(hist)

if __name__ == '__main__':
    main()
    #test()
    #test_turning()
    #test_plot_history()
    #get_heatmap()
