# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0,0)]+[(i,j) for i in range (m) for j in range(m) if i!=j]
        self.state_space = [[i,j,k] for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = random.choice(self.state_space)

        self.total_time = 0
        self.max_time = 24*30
        # Start the first round
        self.reset()
        

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint:           The vector is of size m + t + d."""
        
        state_encod =[0 for i in range(m+t+d)]
        state_encod[int(state[0])] = 1
        state_encod[m+int(state[1])] = 1
        state_encod[m+t+int(state[2])] = 1
        
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)     
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8) 

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # (0,0) is not considered as customer request and hence added 0 index separately as we will return action (0,0) too and therefore to avoid mismatch.
        actions = [self.action_space[i] for i in possible_actions_index]
  
        actions.append((0,0)) # (0,0) is not considered as customer request

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        #Formula for reward calculation 
        # 𝑅(𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘) = 𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞) + 𝑇𝑖𝑚𝑒(𝑖, 𝑝)) 𝑎 = (𝑝, 𝑞)
        #               = - 𝐶𝑓 𝑎 = (0,0)
        
        start_loc,time,day = state
        pickup,drop = action
        
        if action == (0,0):
            reward = -C
        else:
            time_to_pickup = int(Time_matrix[start_loc,pickup,int(time),int(day)])
            time_new = int((int(time) + time_to_pickup)%t)
            day_new = int((int(day)+ ((int(time) + time_to_pickup)//t))%d)
            time_to_drop = int(Time_matrix[pickup,drop,time_new,day_new])
            
            reward = R*time_to_drop - C*(time_to_drop + time_to_pickup)
        
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        start_loc,time,day = state
        pickup,drop = action
        
        X=start_loc
        T=int(time)
        D=int(day)
        
        if action == (0,0):
            #Since no ride action just moves the time component by 1 hour.
            time_to_no_ride = T+1
            time_after_no_ride = int((time_to_no_ride)%t)
            day_after_no_ride = int((D+ (time_to_no_ride//t))%d)
            
            self.total_time += time_to_no_ride
            
            T=time_after_no_ride
            D=day_after_no_ride
            
            #next_state =np.array([X,T,D])
            next_state = [X,T,D]
            
        else:
            time_to_pickup = int(Time_matrix[X,pickup,T,D])
            time_after_pickup = int((T + time_to_pickup)%t)
            day_after_pickup = int((D + ((T + time_to_pickup)//t))%d)
            time_to_drop = int(Time_matrix[pickup,drop,time_after_pickup,day_after_pickup])
            time_after_drop = int((time_after_pickup + time_to_drop)%t)
            day_after_drop = int((day_after_pickup + ((time_after_pickup + time_to_drop)//t))%d)
            
            self.total_time += time_to_pickup+time_to_drop
            
            X= drop
            T=time_after_drop
            D=day_after_drop
            
            #next_state =np.array([X,T,D])
            next_state = [X,T,D]
        
        if self.total_time>= self.max_time:
            done = True
        else:
            done = False
            
        return next_state,done


    def reset(self):
        return self.action_space, self.state_space, self.state_init
