import numpy as np
from physics_sim import PhysicsSim
import math

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pose=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pose: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_z = init_pose[2]

        # Goal
        self.target_pose = target_pose if target_pose is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """#Uses current pose of sim to return reward.
        #Original reward, commented out.
        #reward = 1.-.3*(abs(self.sim.pose[:3] -       self.target_pose[:3])).sum()
   


        #Takeoff - Punish upto -1 if copter moves away from x and y coordinates
        x_punish = np.tanh(abs(self.sim.pose[0] - self.target_pose[0]))
        y_punish = np.tanh(abs(self.sim.pose[1] - self.target_pose[1]))
        
        #Punish up to -1 for rotating, it is just supposed to take off without rotating!
        rot_1_punish = np.tanh(abs(self.sim.pose[3]))
        rot_2_punish = np.tanh(abs(self.sim.pose[4]))
        rot_3_punish = np.tanh(abs(self.sim.pose[5])) 
       
        #Punish if negative, reward if positive. Amplify movement on z axis, we want copter to take off in positive Z...
        z_punish_reward = 3*np.tanh(abs(self.sim.pose[2] - self.init_z))
        
        reward = z_punish_reward - x_punish - y_punish - rot_1_punish - rot_2_punish - rot_3_punish                       
        return reward
        """  
    
    
        #Hover
        punish = []
        distance = abs(np.linalg.norm(self.target_pose[:3] - self.sim.pose[:3]))
        #Reward 5 points if copter stays within 2 units on z-axis, if not punish up to 1 using tanh.
        if distance < 2:
            punish.append(-10)
        else:
            punish.append(np.tanh(distance))
        #Punish upto -1 for any other movements like rotation or velocity
        punish.append(np.tanh(np.linalg.norm(self.sim.pose[3:6])))
        punish.append(np.tanh(np.linalg.norm(self.sim.v)))
        reward = -sum(punish)
        return reward
       
 
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state
    