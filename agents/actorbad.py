from keras import layers
from keras import optimizers
from keras import models
from keras import intializers
from keras import regularizers
from keras import backend as kb

class Actor:
    
    def __init__(self, state_size, action_size action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        
        self.build_model()
        
    def build_model(self):
        #define input layer for states
        states = layers.Input(share=(self.sate_size,), name='state')
        
        #adding hidden layers
        network = layers.Dense(units=32, activation='relu')(states)
        network = layers.Dense(units=64, activation='relu')(network)
        network = layers.Dense(units=64, activation='relu')(network)
        network = layers.normalization.BatchNormalization()(network()
                                                            