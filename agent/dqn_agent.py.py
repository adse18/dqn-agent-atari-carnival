import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Input
import cv2
import os

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

state_size = (96, 80, 1)

class DQN_Agent:
    # Initializes attributes and constructs CNN model and target_model

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory

        # Hyperparameters
        self.gamma = 0.9  # Discount rate, how much is future reward worth
        self.epsilon = 0.9  # Exploration rate
        self.epsilon_min = 0.1  # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.9  # Decay rate for epsilon
        self.update_rate = 4  # Number of steps until updating the target network

        # CNN Hyperparameters
        self.learning_rate = 0.020
        self.loss = tf.compat.v1.losses.huber_loss 
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=self.learning_rate,
            rho=0.95,
            momentum=0.02,
            epsilon=0.00001,
            centered=True
        )

        # Construct DQN models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    #
    # Constructs CNN
    #
    def _build_model(self):
        model = Sequential()

        model.add(Input(shape=self.state_size))
        # Conv Layers
        # The first number is the output depth. The tuples are kernel size.
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', activation='relu'))
        model.add(Conv2D(64, (4, 4), strides=2, padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.add(Flatten())
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    # Stores experience in replay memory
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Chooses action based on epsilon-greedy policy
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state, verbose=0)

        return np.argmax(act_values[0])  # Returns action using policy

    # Trains the model using randomly selected experiences in the replay memory
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)  # Batch size is the return size

        for state, action, reward, next_state, done in minibatch:

            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)))  # TD
            else:
                target = reward

            # Construct the target vector as follows:
            # Use the current model to output the Q-value predictions.
            target_f = self.model.predict(state, verbose=0)  # This constructs our Q value

            # Rewrite the chosen action value with the computed target
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)  # Train model so Q-value match the target function

        #  Simple decay rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # Decay rate

    # Sets the target model parameters to the current model parameters
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

'''Preprocessing: To save on computations'''

# Helpful preprocessing taken from github.com/ageron/tiny-dqn
def process_frame(frame):
    frame_np = np.array(frame[0])  # Convert frame[0] to numpy array

    # Check if frame is grayscale or RGB and convert accordingly
    if len(frame_np.shape) == 2:  # Grayscale image
        img = frame_np
    elif len(frame_np.shape) == 3 and frame_np.shape[2] == 3:  # RGB image
        img = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("ACHTUNG! Unsupported frame format or number of channels.")

    img = img / 255  # Normalize
    img = cv2.resize(img, state_size[:2])  # Resize the image
    return np.expand_dims(img.reshape(state_size), axis=0)
