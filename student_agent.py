# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
import simple_custom_taxi_env
import csv

#taxi_row, taxi_col, S1x, S1y, S2x, S2y, S3x, S3y, S4x, S4y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = next_obs
def loadTable():
        with open('q_table.csv', 'r') as f:
            reader = csv.reader(f)
            Q_table = {eval(row[0]): list(map(float, row[1:])) for row in reader}
        return Q_table

Q_table = loadTable()

def get_state(current_state, next_obs, action, current_passenger_picked, target_counter):
        """
        Convert observations into a structured state for Q-learning.
        Tracks passenger pickup and drop-off correctly.
        """
        taxi_row, taxi_col, S1x, S1y, S2x, S2y, S3x, S3y, S4x, S4y, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = next_obs
        # Compute directions towards the stations
        if len(targets) == 0:
            targets.append((S1x, S1y))
            targets.append((S2x, S2y))
            targets.append((S3x, S3y))
            targets.append((S4x, S4y))

        if target_counter is None:
            target_counter = 0
        on_station = (taxi_row, taxi_col) in [(S1x, S1y), (S2x, S2y), (S3x, S3y), (S4x, S4y)]

        reached = False

        if (taxi_row, taxi_col) == targets[target_counter]:
            #target_counter = (target_counter + 1) % 4
            reached = True
            
        tar_dir = (np.sign(targets[target_counter][0] - taxi_row), np.sign(targets[target_counter][1] - taxi_col))
        passenger_picked = current_passenger_picked
        
        if action == 4 and passenger_look and not current_passenger_picked and on_station:
            passenger_picked = True  # Passenger is now inside the taxi

        state = (
            tar_dir[0], tar_dir[1], 
            passenger_look, destination_look,
            obstacle_north, obstacle_south, obstacle_east, obstacle_west, action,
            passenger_picked
        )
       
        return state, on_station, reached

targets = []
current_state = None
current_passenger_picked = False
action = -1
epsilon = 0.00
max_epsilon = 0.0
decay_rate = 0.99
target_counter = 0

def get_action(obs):

    global current_state, current_passenger_picked, action, epsilon, decay_rate,targets, min_epsilon, target_counter
    
    next_state, _ , reached = get_state(current_state, obs, action, current_passenger_picked, target_counter)
    if reached:
         target_counter = (target_counter + 1) % 4
    #print('state: ', next_state)
    if next_state not in Q_table or np.random.rand() < epsilon:
        action = np.random.choice([0,1,2,3])
    else:
        action = np.argmax(Q_table[next_state])
    
    current_state = next_state
    current_obs = obs
    current_passenger_picked = current_state[-1]
    epsilon  = min(epsilon / decay_rate, max_epsilon)
    return action

    
