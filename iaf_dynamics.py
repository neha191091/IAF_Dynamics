import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import MultivariateNormalDiag, kl_divergence
from base import Mlp

class BaselineTransitionNoKL(object):

    def __init__(self, n_latent, n_enc, n_control):

        self.n_latent = n_latent
        self.n_enc = n_enc
        self.n_control = n_control

        # Mlp for posterior and prior
        n_input_q = self.n_enc + self.n_latent + self.n_control
        n_input_p = self.n_latent + self.n_control
        n_output = self.n_latent + self.n_latent
        layers = [256, 256]
        activation = ['relu', 'relu']
        out_activation = lambda x: x
        self.q_mlp = Mlp(n_input_q, layers, n_output, activation, out_activation)
        self.p_mlp = Mlp(n_input_p, layers, n_output, activation, out_activation)

    def q_transition(self, z, enc, u):

        # Concat the inputs
        zeu = tf.concat([z, enc, u], 1)

        # Get mean and var
        out = self.q_mlp(zeu)
        mean = out[:, :self.n_latent]
        var = out[:, self.n_latent:]

        return mean, var ** 2 + 1e-5

    def p_transition(self, z, u):

        # Concat the inputs
        zu = tf.concat([z, u], 1)

        # Get mean and var
        out = self.p_mlp(zu)
        mean = out[:, :self.n_latent]
        var = out[:, self.n_latent:]

        return mean, var ** 2 + 1e-5

    def one_step(self, a, x):
        
        z = a[0]
        u, enc = x

        q_mean, q_var = self.q_transition(z, enc, u)
        p_mean, p_var = self.p_transition(z, u)
        
        q = MultivariateNormalDiag(q_mean, tf.sqrt(q_var))
        p = MultivariateNormalDiag(p_mean, tf.sqrt(p_var))

        z_step = q.sample()

        # TODO: Not sure why this reshape is necessary...
        log_q = tf.reshape(q.log_prob(z_step), shape=tf.shape(a[1]))
        log_p = tf.reshape(p.log_prob(z_step), shape=tf.shape(a[2]))

        # TODO: Remove after verifying the veracity of monte carlo est
        kl = kl_divergence(q,p)
        mc_est = log_q -log_p

        return z_step, log_q, log_p, kl, mc_est

    def gen_one_step(self, z, u):

        p_mean, p_var = self.p_transition(z, u)
        
        p = MultivariateNormalDiag(p_mean, tf.sqrt(p_var))

        z_step = p.sample()

        return z_step


class DVBFNoKL():
    def __init__(self, n_obs, n_control, n_latent, n_enc, learning_rate=0.001):

        self.learning_rate = tf.placeholder(tf.float32)
        
        # Dimensions
        self.n_output = n_obs
        self.n_obs = n_obs
        self.n_control = n_control
        self.n_latent = n_latent
        self.n_enc = self.n_latent

        # The placeholder from the input
        self.x = tf.placeholder(tf.float32, [None, None, self.n_obs], name="X")
        self.u = tf.placeholder(tf.float32, [None, None, self.n_control], name="U")

        # Initialize p(z0), p(x|z), q(z'|enc, u, z) and p(z'|z) as well as the mlp that
        # generates a low dimensional encoding of x, called enc
        self._init_generative_dist()
        self._init_start_dist()
        self._init_encoding_mlp()
        self.transition = BaselineTransitionNoKL(self.n_latent, self.n_enc, self.n_control)
        
        # Get the encoded representation of the observations (this makes sense when observations are highdimensional images for example)
        enc = self.get_enc_rep(self.x)
        
        # Get the latent start state
        q0 = self.get_start_dist(self.x[0])
        z0 = q0.sample()
                               
        # Trajectory rollout in latent space + calculation of KL(q(z'|enc, u, z) || p(z'|z))

        # TODO: Change after verifying the veracity of monte carlo est
        z, log_q, log_p, kl, mc_est = tf.scan(self.transition.one_step, (self.u[:-1], enc[1:]), (z0, tf.zeros([tf.shape(z0)[0],]), tf.zeros([tf.shape(z0)[0],]), tf.zeros([tf.shape(z0)[0],]), tf.zeros([tf.shape(z0)[0],])))
        self.z = tf.concat([[z0], z], 0)
        
        # Get the generative distribution p(x|z) + calculation of the reconstruntion error
        px = self.get_generative_dist(z)
        rec_loss = -px.log_prob(self.x[1:])
        
        # Generating trajectories given only an initial observation
        gen_z = tf.scan(self.transition.gen_one_step, self.u[:-1], z0)
        self.gen_z = tf.concat([[z0], gen_z], 0)
        gen_px = self.get_generative_dist(self.gen_z)
        self.gen_x_mean = gen_px.mean()
        
        # Create the losses
        self.rec_loss = rec_loss
        self.log_p = log_p
        self.log_q = log_q
        self.total_loss = tf.reduce_mean(self.rec_loss + self.log_q - self.log_p)

        # TODO: Remove after verifying the veracity of monte carlo est
        self.kl_analytical = kl
        self.kl_monte_carlo = mc_est

        # Use the Adam optimizer with clipped gradients
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(self.total_loss)
        capped_grads_and_vars = [(tf.clip_by_value(grad, -1, 1), var) if grad is not None else (grad, var) for 
                                 (grad, var) in grads_and_vars]
        self.optimizer = self.optimizer.apply_gradients(capped_grads_and_vars)

        # Save weights
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
    def _init_start_dist(self):
        self.init_mlp = Mlp(self.n_obs, [128], 2 * self.n_latent, ['relu'], lambda x: x)
        
    def _init_encoding_mlp(self):
        self.encoding_mlp = Mlp(self.n_obs, [128], self.n_enc, ['relu'], lambda x: x)
        
    def _init_generative_dist(self):
        self.generative_mlp = Mlp(self.n_latent, [128], self.n_obs, ['relu'], lambda x: x)
        self.x_var = tf.Variable(tf.random_normal((1, self.n_obs)) * 0.001)
        
    def get_start_dist(self, obs0):
        out = self.init_mlp(obs0)
        z0_mean = out[:, :self.n_latent]
        z0_var = out[:, self.n_latent:] ** 2 + 1e-5
        return MultivariateNormalDiag(z0_mean, tf.sqrt(z0_var))
    
    def get_enc_rep(self, x):
        enc = self.encoding_mlp(tf.reshape(x, (-1, self.n_obs)))
        enc = tf.reshape(enc, (tf.shape(x)[0], -1, self.n_enc))
        return enc
        
    def get_generative_dist(self, z):
        x_mean = self.generative_mlp(tf.reshape(z, (-1, self.n_latent)))
        x_mean = tf.reshape(x_mean, (tf.shape(z)[0], -1, self.n_obs))
        x_var = tf.zeros_like(x_mean) + self.x_var ** 2 + 1e-8
        x_var = tf.reshape(x_var, (tf.shape(z)[0], -1, self.n_obs))
        
        return MultivariateNormalDiag(x_mean, tf.sqrt(x_var))

    def save(self, name='dvbf', global_step=0):
        return self.saver.save(self.sess, name, global_step)

    def restore(self, path):
        self.saver.restore(self.sess, path)
        
    def train(self, batch_x, batch_u, learning_rate):
        # TODO: Change after verifying the veracity of monte carlo est
        _, total_loss, kl, mc = self.sess.run((self.optimizer, self.total_loss , self.kl_analytical, self.kl_monte_carlo), feed_dict={self.x: batch_x, self.u: batch_u, self.learning_rate: learning_rate})
        return total_loss, kl, mc