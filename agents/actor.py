from keras import layers
from keras import optimizers
from keras import models
from keras import initializers
from keras import regularizers
from keras import backend as kb

class Actor:
    
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        
        self.build_model()
        
    def build_model(self):
        #define input layer for states
        states = layers.Input(shape=(self.state_size,), name='state')
        
        #adding hidden layers
        network = layers.Dense(units=32, activation='relu')(states)
        network = layers.Dense(units=128, activation='relu')(network)
        network = layers.Dense(units=32, activation='relu')(network)
        network = layers.normalization.BatchNormalization()(network)
        
        #add output layer
        model_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                                     kernel_initializer=initializers.random_uniform(minval=-0.001, maxval=0.001),
                                     name='model_actions')(network)
        
        #scale output to action range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(model_actions)
        
        #create model
        self.model = models.Model(inputs=states, outputs=actions)
        
        #define loss function
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = kb.mean( -action_gradients * actions)
        
        #define optimizer
        optimizer = optimizers.Adam(lr=0.0001)
        update_optimizer = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = kb.function(
            inputs=[self.model.input, action_gradients, kb.learning_phase()],
            outputs=[],
            updates=update_optimizer)
            
            
        
        
                                                          
                                                            
