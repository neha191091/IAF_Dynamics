import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import dateutil.tz
import datetime

_CHKPT_PATH = 'chkpt/'

def get_timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M')
    print (timestamp)
    return timestamp

def save_checkpoint(sess, timestamp, checkpoint=0, var_list=None):
    if not os.path.exists(_CHKPT_PATH):
        os.makedirs(_CHKPT_PATH)
        # save model
    fname = _CHKPT_PATH + '%s' % timestamp + '_checkpoint%d.ckpt' % checkpoint
    saver = tf.train.Saver(var_list)
    save_path = saver.save(sess, fname)
    print("Model saved in %s" % save_path)


def load_checkpoint(sess, filename):
    # load model
    fname = filename
    try:
        saver = tf.train.Saver()
        saver.restore(sess, fname)
        print("Model restored from %s" % fname)
    except:
        print("Failed to load model from %s" % fname)




