import tensorflow as tf
import numpy as np

import time
import matplotlib.pyplot as plt
import seaborn as sns

import cPickle as pickle

from IPython import display

from inserted_iaf_dynamics import DVBFNoKL
from baseline import DVBF
from world import PendulumFullObs
import utils
import os


X, U, R, S = pickle.load(open("data/training_set.p", "rb"))

# Define the model
n_obs = 3
n_control = 1
n_latent =  4
n_enc = 10

chkpoint_file = 'chkpt/2017_10_16_17_58_checkpoint999.ckpt'
m = DVBFNoKL(3, 1, 4, 10, chkpoint_file=chkpoint_file)


timestamp=utils.get_timestamp()

_RESULT_PATH = 'results/'
if not os.path.exists(_RESULT_PATH):
    os.makedirs(_RESULT_PATH)


X_temp, U_temp, R_temp, S_temp = pickle.load(open("data/test_set.p", "rb"))
x_gen, x_obs, rec_loss_val = m.sess.run((m.gen_x_mean, m.px_mean, m.rec_loss), feed_dict={m.x: X_temp, m.u:U_temp})
rec_diff = np.mean(np.abs(X_temp - x_obs))
gen_diff = np.mean(np.abs(X_temp - x_gen))


result_details_path = _RESULT_PATH + 'evaluation_details.txt'
result_details_file = open(result_details_path, 'a+')
result_details_file.write(str(timestamp) + ':  rec_diff_inserted_iaf: ' + str(rec_diff) + ';  gen_diff_inserted_iaf: ' + str(gen_diff) + ';  rec_loss: ' + str(rec_loss_val) + '\n')
result_details_file.close()
# Plot the position and reward of low dim pendulum
e = int(np.random.rand() * 100)
e = 1

plt.close()
f, axarr = plt.subplots(1, 2, figsize=(15, 6))
axarr[0].plot(x_gen[:, e, 0])
axarr[0].plot(x_gen[:, e, 1])
axarr[0].plot(x_gen[:, e, 2])
axarr[0].set_ylim(-1.1, 1.1)
axarr[1].plot(X_temp[:, e, 0])
axarr[1].plot(X_temp[:, e, 1])
axarr[1].plot(X_temp[:, e, 2])
axarr[1].set_ylim(-1.1, 1.1)
axarr[0].set_title('Generated Trajectory from Baseline Checkpoint: ')
axarr[1].set_title('Generated Trajectory from Ground Truth: ')
plt.savefig(_RESULT_PATH + '%s' % timestamp + '_PositionPlotInsertedIafGen.png')
# plt.show()

plt.close()
f, axarr = plt.subplots(1, 2, figsize=(15, 6))
axarr[0].plot(np.arctan2(x_gen[:, e, 1], x_gen[:, e, 0]))
axarr[1].plot(np.arctan2(X_temp[:, e, 1], X_temp[:, e, 0]))
axarr[0].set_title('Generated Trajectory from Baseline Checkpoint: ')
axarr[1].set_title('Generated Trajectory from Ground Truth: ')
plt.savefig(_RESULT_PATH + '%s' % timestamp + '_ArcTanPlotInsertedIafGen.png')

plt.close()
f, axarr = plt.subplots(1, 2, figsize=(15, 6))
axarr[0].plot(x_obs[:, e, 0])
axarr[0].plot(x_obs[:, e, 1])
axarr[0].plot(x_obs[:, e, 2])
axarr[0].set_ylim(-1.1, 1.1)
axarr[1].plot(X_temp[:, e, 0])
axarr[1].plot(X_temp[:, e, 1])
axarr[1].plot(X_temp[:, e, 2])
axarr[1].set_ylim(-1.1, 1.1)
axarr[0].set_title('Generated Trajectory from Iaf Checkpoint: ' + chkpoint_file)
axarr[1].set_title('Generated Trajectory from Ground Truth: ')
plt.savefig(_RESULT_PATH + '%s' % timestamp + '_PositionPlotInsertedIafObs.png')

plt.close()
f, axarr = plt.subplots(1, 2, figsize=(15, 6))
axarr[0].plot(np.arctan2(x_obs[:, e, 1], x_obs[:, e, 0]))
axarr[1].plot(np.arctan2(X_temp[:, e, 1], X_temp[:, e, 0]))
axarr[0].set_title('Generated Trajectory from Iaf Checkpoint: ' + chkpoint_file)
axarr[1].set_title('Generated Trajectory from Ground Truth: ')
plt.savefig(_RESULT_PATH + '%s' % timestamp + '_ArcTanPlotInsertedIafObs.png')