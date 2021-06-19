import gym
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

problem = "Pendulum-v0"
env = gym.make(problem)

actor_model = models.load_model("./target_actor.h5")

def policy(state):
    sampled_actions = tf.squeeze(actor_model.predict(state)) * 1.5
    #if np.random.random() < 0.1:
    #    sampled_actions = sampled_actions + np.random.normal()
    legal_action = np.clip(sampled_actions, -2.0, 2.0)
    return [np.squeeze(legal_action)]

while True:
    prev_state = env.reset()

    while True:
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        action = policy(tf_prev_state)

        state, reward, done, info = env.step(action)
        prev_state = state
        env.render()

        if done:
            print("game over")
            break
