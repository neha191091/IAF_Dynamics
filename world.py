import gym
import numpy as np
import scipy
import scipy.stats
import scipy.ndimage as nd
from scipy.integrate import odeint



def pendulum_reward(x, u, goal):
    r = 1.0 * (x[0] - 1.0) ** 2 + 0.5 * (x[1]) ** 2 + 0.01 * x[2] ** 2

    return r / 4.5  # approximate normalization


def pendulum_reward_angle(x, u, goal):
    r = np.arctan2(x[1], x[0]) ** 2 + 0.04 * x[2] ** 2

    return r / 12.5  # approximate normalization


def reacher_reward(x, u, goal):

    r = (x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2 + (x[2] - goal[2]) ** 2 + (x[3] - goal[3]) ** 2 + \
        0.005 * x[6] ** 2 + 0.005 * x[7] ** 2

    return r / 50  # approximate normalization


class OUNoise:
    def __init__(self, action_dimension=1, mu=0.0, theta=0.15, sigma=1.5):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def control(self, observation):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        return np.clip(self.state, [-2.0], [2.0])


class BaseWorld:

    def __init__(self, noisy=False, var=0.1, goal=[0.0, 0.0]):
        self.env = NotImplemented
        self.noisy = noisy
        self.var = var
        self.n_obs = 3
        self.n_control = 1
        self.goal_obs = None
        self.default_exploration_policy = NotImplemented
        self.default_reward = NotImplemented
        self.init_action = [[0.0]]
        self.control_limits = NotImplemented

    def reset(self):
        return NotImplemented

    def set_state(self, state):
        self.env.state = NotImplemented

    def get_control_limits(self):
        return self.control_limits

    def _transfer_obs(self, env_obs):
        return NotImplemented

    def set_goal(self, goal):
        self.goal_obs = NotImplemented

    def get_goal_obs(self):
        return self._transfer_obs(self.goal_obs)

    def sample_action(self):
        return self.env.action_space.sample()

    def step(self, control):
        state, reward, done, info = self.env.step(control)
        obs = self._transfer_obs(state)
        if self.noisy:
            obs += self.var * np.reshape(np.random.randn(self.n_obs), obs.shape)
        return obs, reward, state

    def render(self):
        return self.env.render(mode='rgb_array')

    def close(self):
        self.env.render(close=True)

    def get_data_set(self, policy=None, custom_reward=True, episodes=1000, steps=100):

        if policy is None:
            policy = self.default_exploration_policy

        X = []
        U = []
        R = []
        S = []

        for e in range(episodes):
            self.reset()
            state = self.env._get_obs()

            if policy:
                policy.reset()

            X.append([])
            U.append([])
            R.append([])
            S.append([])

            for i in range(steps):

                control = policy.control(state)
                if custom_reward is True:
                    r = self.default_reward(state, control, self.goal_obs)
                observation, r2, state = self.step(control)

                if custom_reward is False:
                    r = r2

                U[e].append(control)
                X[e].append(observation)
                S[e].append(state)
                R[e].append([r])

        X = np.array(X).swapaxes(0, 1)
        U = np.array(U).swapaxes(0, 1)
        R = np.array(R).swapaxes(0, 1)
        S = np.array(S).swapaxes(0, 1)

        return X, U, R, S

    def get_initial_steps(self, state=None, n_steps=5, custom_reward=True):

        if state:
            self.set_state(state)
        else:
            self.reset()
        state = self.env._get_obs()

        if custom_reward is None:
            custom_reward = self.default_reward

        init_obs = []
        init_us = []
        init_rs = []

        for _ in range(n_steps):

            control = self.init_action
            if custom_reward is True:
                r = self.default_reward(state, control, self.goal_obs)
            observation, r2, state = self.step(control)
            if custom_reward is False:
                r = r2

            init_us.append(control)
            init_obs.append(observation)
            init_rs.append([r])

        return init_obs, init_us, init_rs


class PendulumFullObs(BaseWorld):
    def __init__(self, noisy=False, var=0.1, goal=[0.0, 0.0]):
        BaseWorld.__init__(self, noisy=noisy, var=var)
        self.env = gym.make('Pendulum-v0')
        self.env.reset()
        self.noisy = noisy
        self.var = var
        self.n_obs = 3
        self.n_control = 1
        self.goal_obs = np.array([np.cos(goal[0]), np.sin(goal[0]), goal[1]])
        self.default_exploration_policy = OUNoise(action_dimension=1, mu=0.0, theta=0.15, sigma=1.5)
        self.default_reward = pendulum_reward
        self.init_action = [[0.0]]
        self.control_limits = [[-2.0, 2.0]]

    def reset(self):
        return self._transfer_obs(self.env.reset())

    def set_state(self, state):
        self.env.state = state

    def _transfer_obs(self, env_obs):
        return env_obs[:self.n_obs]

    def set_goal(self, goal):
        self.goal_obs = np.array([np.cos(goal[0]), np.sin(goal[0]), goal[1]])


class PendulumPartObs(PendulumFullObs):
    def __init__(self, noisy=False, var=0.1):
        PendulumFullObs.__init__(self, noisy=noisy, var=var)
        self.n_obs = 2


class PendulumPixelObs(PendulumFullObs):
    def __init__(self, noisy=False, var=0.1, goal=[0.0, 0.0]):
        PendulumFullObs.__init__(self, noisy=noisy, var=var, goal=[0.0, 0.0])
        self.img_dims = [16, 16]
        self.n_obs = self.img_dims[0] * self.img_dims[1]

    def _transfer_obs(self, env_obs):
        env_obs = np.array(env_obs)
        env_obs = np.reshape(env_obs, [3,])
        x = np.linspace(-4, 4, self.img_dims[0])
        y = np.linspace(-4, 4, self.img_dims[1])
        xv, yv = np.meshgrid(x, y)
        theta = np.arctan2(env_obs[1], env_obs[0])
        r = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])

        obs = scipy.stats.norm.pdf(np.dot(np.concatenate((xv.ravel()[:, np.newaxis], yv.ravel()[:, np.newaxis]), 1), r),
                                   loc=[0, 2.0], scale=[0.5, 0.9]).prod(1)

        return obs


class ReacherFullObs(PendulumFullObs):

    def __init__(self, noisy=False, var=0.1, goal=[0.0, 0.0], env='Reacher-v1'):
        BaseWorld.__init__(self, noisy=noisy, var=var)
        self.env = gym.make(env)
        self.env.reset()
        self.noisy = noisy
        self.var = var
        self.n_obs = 11
        self.n_control = 2
        self.init_action = [[0.0], [0.0]]
        self.goal_obs = None
        self.set_goal(goal)
        self.default_exploration_policy = OUNoise(action_dimension=2, mu=0.0, theta=0.25, sigma=0.6)
        self.default_reward = reacher_reward
        self.control_limits = [[-2.0, 2.0], [-2.0, 2.0]]
        self.internal_goal = [0.0, 0.0]

    def reset(self):
        self.env.reset()
        random_state = np.random.rand(2, ) * 1.2 * np.pi - 0.6 * np.pi
        self.set_state(random_state)
        random_goal = np.random.rand(2,) * 1.2 * np.pi - 0.6 * np.pi
        #random_goal = [0.0, 0.0]
        self.set_goal(random_goal)

        return self._transfer_obs(self.env._get_obs())

    def step(self, control):
        state, reward, done, info = self.env.step(control)
        # Approximate reward normalization
        reward /= -10.0
        obs = self._transfer_obs(state)
        if self.noisy:
            obs += self.var * np.reshape(np.random.randn(self.n_obs), obs.shape)
        return obs, reward, state

    def _transfer_obs(self, env_obs):
        # approximate normalization
        env_obs[4:6] /= 0.25
        env_obs[6:8] /= 100
        return env_obs

    def render(self):
        return self.env.render(mode='rgb_array')

    def set_state(self, state):
        x = (np.cos(self.internal_goal[0]) + np.cos(self.internal_goal[0] + self.internal_goal[1])) / 10.0
        y = (np.sin(self.internal_goal[0]) + np.sin(self.internal_goal[0] + self.internal_goal[1])) / 10.0
        angles = np.array(np.concatenate([state, np.array([x, y])]))
        velocities = np.array([0.0, 0.0, 0.0, 0.0])
        self.env.set_state(angles, velocities)

    def set_goal(self, goal):
        self.internal_goal = goal
        self.goal_obs = np.array([np.cos(goal[0]), np.cos(goal[1]), np.sin(goal[1]), np.sin(goal[1]), 0.0, 0.0])

        # Set the rendered red dot to match this goal position
        angles = np.array(self.env.model.data.qpos.flat)
        velocities = np.array(self.env.model.data.qvel.flat)
        x = (np.cos(goal[0]) + np.cos(goal[0] + goal[1])) / 10.0
        y = (np.sin(goal[0]) + np.sin(goal[0] + goal[1])) / 10.0
        angles[-2:] = [x, y]
        self.env.set_state(angles, velocities)

    def get_goal_obs(self):
        return self.goal_obs