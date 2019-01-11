
import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf

EPISODES = 1000#迭代次数

class model:
    def __init__(self,state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate=0.001
        self.x = tf.placeholder(shape=[None, 4], name="x",dtype='float64')
        self.y = tf.placeholder(shape=[None, 2], name="y",dtype='float64')

    def _build_model(self,):
        with tf.variable_scope("dnn"):
            hidden1 = tf.contrib.layers.fully_connected(self.x, 24, activation_fn=tf.nn.relu, scope="hidden1")
            hidden2 = tf.contrib.layers.fully_connected(hidden1, 24, activation_fn=tf.nn.relu, scope="hidden2")
            self.predictions = tf.contrib.layers.fully_connected(hidden2, self.action_size, scope="outputs", activation_fn=None)

        with tf.variable_scope('loss'):
            self.mse = tf.reduce_sum(tf.square(self.predictions - self.y))
        with tf.variable_scope('train'):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(self.mse)


    def predict(self, sess, s):

        return sess.run(self.predictions , {self.x: s})

    def fit(self, sess, s, y):
        feed_dict = {self.x: s, self.y: y}
        _,loss = sess.run(
            [self.training_op, self.mse],
            feed_dict)
        return loss



class DQNAgent:
    def __init__(self, state_size, action_size,dqn):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)#经验回放缓存区
        self.gamma = 0.95    #  Q函数discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = dqn




    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, sess,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(sess,state)
        return sess.run(tf.argmax(input=act_values, axis=1))  # returns action

    def replay(self,sess, batch_size):
        minibatch = random.sample(self.memory, batch_size)#均匀抽样
        for state, action, reward, next_state, done in minibatch:
            target = reward
            act_values = self.model.predict(sess, next_state)

            if not done:
                target = (reward + self.gamma *
                          sess.run(tf.argmax(input=act_values, axis=1)))

            target_f = self.model.predict(sess,state)
            target_f[0][action] = target

            self.model.fit(sess,state, target_f)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay#epsilon_decay逐渐减小，知道其为epsilon_min




if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    dqn=model(state_size, action_size)

    dqn._build_model()
    agent = DQNAgent(state_size, action_size, dqn)

    done = False
    batch_size = 32
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(EPISODES):
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            for time in range(500):
                env.render()
                action = agent.act(sess,state)
                if type(action)!=int:
                    action=action[0]
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, time, agent.epsilon))
                    break
                if len(agent.memory) > batch_size:

                    agent.replay(sess,batch_size)