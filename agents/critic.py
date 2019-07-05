from keras import layers, models, optimizers, initializers, regularizers
from keras import backend as kb

class Critic:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.build_model()
        
    def build_model(self):
        #Input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        #Hidden layers for states
        h_states = layers.Dense(units=32, activation='relu')(states)
        h_states = layers.Dense(units=64, activation='relu')(h_states)
        
        #Hidden layers for actions
        h_actions = layers.Dense(units=32, activation='relu')(actions)
        h_actions = layers.Dense(units=64, activation='relu')(h_actions)
        
        #Combine states and actions
        network = layers.Add()([h_states, h_actions])
        network = layers.Activation('relu')(network)
        
        #Add batch normlization layer
        network = layers.normalization.BatchNormalization()(network)
        
        #Add output layer to produce action values
        Q_values = layers.Dense(units=1, name='q_values',
                               kernel_initializer=initializers.random_uniform(minval=-0.0005, maxval=0.0005))(network)
        
        #Create Model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        #Compile Model
        self.model.compile(optimizer=optimizers.Adam(lr=0.001), loss='mse')
        
        #compute action gradients
        action_gradients = kb.gradients(Q_values, actions)
        
        self.get_action_gradients = kb.function(inputs=[*self.model.input, kb.learning_phase()],
                                                outputs=action_gradients)
        