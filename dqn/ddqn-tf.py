import random
import gym
import numpy as np
from collections import deque

import tensorflow as tf

EPISODES = 5000

class model:
    def __init__(self,state_size, action_size,name,sess):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate=0.001
        self.x = tf.placeholder(shape=[None, 4], name="x",dtype='float64')
        self.y = tf.placeholder(shape=[None, 2], name="y",dtype='float64')
        self.name = name
        self.sess = sess
        self._build_model()


    def _build_model(self):
        with tf.variable_scope(self.name+"_dnn"):
            hidden1 = tf.contrib.layers.fully_connected(self.x, 24, activation_fn=tf.nn.relu, scope="hidden1")
            hidden2 = tf.contrib.layers.fully_connected(hidden1, 24, activation_fn=tf.nn.relu, scope="hidden2")
            self.predictions = tf.contrib.layers.fully_connected(hidden2, self.action_size, scope="outputs", activation_fn=None)

        with tf.variable_scope(self.name+'_loss'):
            self.huber_loss= tf.losses.huber_loss(self.y,self.predictions)
        with tf.variable_scope(self.name+'_train'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(self.huber_loss)
        sess.run(tf.global_variables_initializer())

    def predict(self, sess, s):

        return sess.run(self.predictions , {self.x: s})

    def fit(self, sess, s, y):
        feed_dict = {self.x: s, self.y: y}
        _,loss = sess.run(
            [self.training_op, self.huber_loss],
            feed_dict)
        return loss




class DQNAgent:
    def __init__(self, sess,state_size, action_size,model):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.behaviour_model = model[0]
        self.target_model = model[1]
        self.update_target_model(sess)

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def update_target_model(self,sess):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(self.behaviour_model.name)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(self.target_model.name)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        sess.run(update_ops)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, sess, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.behaviour_model.predict(sess, state)
        return sess.run(tf.argmax(input=act_values, axis=1))  # returns action

    def replay(self,sess, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.behaviour_model.predict(sess,state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(sess,next_state)[0]
                target[0][action] = reward + self.gamma *  sess.run(tf.argmax(input=t))

            self.behaviour_model.fit(sess,state, target)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    with tf.Session() as sess:
        model_list = []
        for i in range(2):
            m = model(4, 2, str(i),sess)
            model_list.append(m)
        agent = DQNAgent(sess,state_size, action_size,model_list)

        done = False
        batch_size = 32

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
                    agent.update_target_model(sess)
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, time, agent.epsilon))
                    break
                if len(agent.memory) > batch_size:
                    agent.replay(sess,batch_size)