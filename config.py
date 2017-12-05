import os
import glob
import sys
import numpy as np
from keras import backend as K

USE_INTENTION=True
USE_INCREASE=True
USE_FILTER=False

DROPOUT=0.3
L2_LAMBDA=1e-5
# MAX_VALUE are used to normalize the regression value to [0,1]
#MAX_VALUE = 0.5
MAX_VALUE = 5
MAX_X = 2
MAX_Z = 2
#MAX_X = 20
#MAX_Z = 660

SAVE_EVERY_OTHER_EPOCH=4
drop_forward = 0
# nueral net parameters
CMD_DIM=2
# weight the importance of steering dimension of the objective function
BETA=1 #0
IMPORTANCE=np.array((1.0, 1.0*BETA), dtype=np.float32)
K_FRAMES=1
SLIDING = 1 #max(1, K_FRAMES/2)

Models = ['vgg16', 'vgg19', 'inception_v3', 'resnet']
if 'IDX' not in os.environ:
    print 'Please set IDX to 0, 1, 2 or 3 in the enviroment.'
    print 'Models ', Models
    sys.exit(-1)

IDX=int(os.environ['IDX'])
NET = Models[IDX]
FC_DR=True
if NET == Models[0]:
    from models.vgg16 import VGG16 as Model
    from models.resnet50 import IntentionModel as IntentionModel
    from models.vgg16 import preprocess_image as Preprocess
    FEAT_LAYER='block5_pool'
    FC_DR=True
elif NET == Models[1]:
    from models.vgg19 import VGG19 as Model
    from models.resnet50 import IntentionModel as IntentionModel
    from models.vgg19 import preprocess_image as Preprocess
    FEAT_LAYER='block5_pool'
    FC_DR=True
elif NET == Models[2]:
    from models.inception_v3 import InceptionV3 as Model
    from models.resnet50 import IntentionModel as IntentionModel
    from models.inception_v3 import preprocess_image as Preprocess
    FEAT_LAYER='avg_pool'
    FC_DR=False
elif NET == Models[3]:
    from models.resnet50 import ResNet50 as Model
    from models.resnet50 import IntentionModel as IntentionModel
    from models.resnet50 import preprocess_image as Preprocess
    FEAT_LAYER='avg_pool'
    FC_DR=False

DIM_ORDER = K.image_dim_ordering()
assert DIM_ORDER in {'tf', 'th'}
#SUFFIX='local_delta'
#SUFFIX='test'
#SUFFIX='local_result'
'''this is the old demo suffix'''
#SUFFIX='test'
'''new test'''
#SUFFIX='rmsprop'
#SUFFIX='server_bench_adam'
#SUFFIX='result'
#SUFFIX='no_drop'
#assert SUFFIX in {'test', 'large'}
'''new test'''
SUFFIX='testparameter'
'''end'''

def get_name():
    return DIM_ORDER + '_' + NET

#MODE='fixed'
MODE='fine_tune'
assert MODE in {'fixed', 'fine_tune'}
#TASK='logitech_park'
TASK = 'multislot_parking'
#TASK = 'trial_parking'
#TASK = '3at_parking'
#TASK = 'nyp_lane'
#TASK = 'nyp_freespace'
#TASK = '3at_corridor'
#TASK = '3at_lane_follow'
#TASK = '3at_lane_change'
#TASK='nyp_slane'
#TASK='office'
#TASK='intention'
#TASK='server_intention'
#TASK='demo'
#TASK='obstacle'
#TASK='demo'
#TASK='dp_skills'
#TASK='new_intention'
#TASK='sim'
#TASK='maze'
#TASK='sim_intention'
#TASK='change_demo'
#TASK='change_lane'
#TASK='NUS'
#TASK='NUS_intention'
#TASK='toyota'
#TASK='simulation'
#TASK='k_f'
#TASK='office'
#TASK='server_office'
#TASK='sim_intention'
#TASK='no_drop_office'
#TASK='debug'

# topics
TOPICS=['/train/cmd_vel', '/train/image_raw', '/train/intention']
#TOPICS=['/actual_vel', '/actual_str', '/image_raw', '/train/intention']    #toyota
#TOPICS=['/navigation_velocity_smoother/raw_cmd_vel', '/image', '/test_intention']

# set paths for storing models and results locally
#BASE='/media/data/umv'
#BASE='/media/psl-ctg/DATA2/umv'
#BASE='/media/ubuntu/Data/umv'
BASE='/media/ubuntu/Data/umv'
DATA_PATH=BASE
MODEL_PATH=BASE+'/model'
HDF5_PATH=BASE+'/hdf5'
#DATA_PATH='/media/data/3at'
#MODEL_PATH='/media/data/umv/model'
#HDF5_PATH='/media/data/umv/hdf5'
DATA_FN=get_name() +'.h5'
SPLIT_FN='split_'+DATA_FN
K_SPLIT_FN='K' + repr(K_FRAMES) + '_split_'+DATA_FN
DROP_K_SPLIT_FN='DROP_'+K_SPLIT_FN
TURNING_FN='turning_'+K_SPLIT_FN
MODEL_FN=MODE+'_K' + repr(K_FRAMES) + '_%s_%s_%s' % (repr(USE_INTENTION), repr(USE_INCREASE), repr(USE_FILTER)) + '_' + SUFFIX + '_weights_' + DATA_FN
HISTORY_FN=MODE+'_K' + repr(K_FRAMES) + '_%s_%s_%s' % (repr(USE_INTENTION), repr(USE_INCREASE), repr(USE_FILTER)) + '_' + SUFFIX + '_' + get_name() + '.pdf'
LATEST_FN=MODE+'_K' + repr(K_FRAMES) + '_%s_%s_%s' % (repr(USE_INTENTION), repr(USE_INCREASE), repr(USE_FILTER)) + '_' + SUFFIX + '_latest_weights_' + DATA_FN

# parameters for training
BATCH_SIZE=8
EPOCHS=100
EPOCH_SIZE=20000/BATCH_SIZE*BATCH_SIZE
VALIDATE_SIZE=2000/BATCH_SIZE*BATCH_SIZE
LEARNING_RATE=0.02

# helper functions
def get_pre_model_fn(M=MODE, INTENTION=USE_INTENTION):
    return M+'_K' + repr(K_FRAMES) + '_' + repr(INTENTION) + '_weights_' + DATA_FN

def get_data_path():
    return os.path.join(DATA_PATH, TASK)

def get_model_path(task=TASK):
    d = os.path.join(MODEL_PATH, task)
    make_dir(d)
    return d

def get_hdf5_path():
    d = os.path.join(HDF5_PATH, TASK)
    make_dir(d)
    return d

def get_data_bags():
    return glob.glob(get_data_path() + '/*.bag')

def get_logging_fn():
    from datetime import datetime
    return MODE+'_K'+repr(K_FRAMES) + '_' + get_name()+'_'+SUFFIX+'_'+TASK+datetime.now().strftime('_%H_%M_%d_%m_%Y.log')

def get_history_fn():
    return MODE+'_K'+repr(K_FRAMES) + '_' + get_name()+'_'+SUFFIX+'_'+TASK+'.pkl'

def make_dir(d):
    try:
        os.stat(d)
    except:
        os.mkdir(d)
    return d

# for tensorflow gpu memrory allocation only
if DIM_ORDER == 'tf':
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    def get_session(device=0, gpu_fraction=0.3):
        os.environ["CUDA_VISIBLE_DEVICES"]=repr(device)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction=gpu_fraction

        num_threads = os.environ.get('OMP_NUM_THREADS')
        if num_threads:
            config.intra_op_parallelism_threads=num_threads

        return tf.Session(config=config)

    KTF.set_session(get_session(0, 0.48))

#########################################################################################
# for controller after this line
#########################################################################################
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
RATE=30
# name, type, queue_size
PUBLISH_TOPIC = ('RosAria/cmd_vel', Twist, 1)
#PUBLISH_TOPIC = ('/mobile_base/commands/velocity', Twist, 1)
#PUBLISH_TOPIC = ('/pred_vel', Twist, 1)
#SUBSCRIBE_TOPIC = ('/train/image_raw', Image, 1)
#SUBSCRIBE_TOPIC = ('/image', Image, 1)
SUBSCRIBE_TOPIC = ('/image_raw', Image, 1)

#INTETION
USE_DISCRETE_INTENTION=True
NUM_INTENTION=50
FORWARD='forward'
BACKWARD='backward'
STOP='stop'
LEFT='left'
RIGHT='right'
#intention_list=[FORWARD, BACKWARD, LEFT, RIGHT, STOP]
#intention_list=[FORWARD, RIGHT]
intention_list=[FORWARD, RIGHT, LEFT]
#intention_list=[FORWARD]
#intention_list=[RIGHT, LEFT]
#intention_list=[FORWARD]
INTENTION_DICT={}
for i in intention_list:
    INTENTION_DICT[i] = 0
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
l_encoder = LabelEncoder()
l_encoder.fit(intention_list)
encoder = OneHotEncoder()

encoder.fit(np.reshape([l_encoder.transform(l) for l in intention_list], (-1, 1)))
def transform(intention):
    l = l_encoder.transform(intention)
    return encoder.transform(l).toarray()[0]

def transform_pred(intention):
    l = l_encoder.transform(intention)
    return encoder.transform(l).toarray()

def print_configuration():
    # print config information
    print ("##################configuration###############")
    print ("net name: ", get_name())
    print ("USE INTENTION: ", USE_INTENTION)
    print ("USE INCREASE: ", USE_INCREASE)
    print ("USE FILTER: ", USE_FILTER)
    print ("TASK: ", TASK)
    print ("SUFFIX: ", SUFFIX)
    print ("K_FRAMES: ", K_FRAMES)
    print ("##################end configuration###############")
