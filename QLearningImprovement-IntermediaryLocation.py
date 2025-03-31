## IMPROVEMENT on Q-Learning Algorithm

## B. ADD FEATURE TO GO BY INTERMEDIARY LOC BEFORE GOING TO TOP PRIORITY

import numpy as np

#setting constant values for Q-learning algorithm
gamma = 0.75 #discount factor
alpha = 0.9 #learning rate

##  PART 1 - DEFINING THE ENVIRONMENT
#defining the states
loc_to_state = {
    "A" : 0 ,
    "B" : 1 ,
    "C" : 2 ,
    "D" : 3 ,
    "E" : 4 ,
    "F" : 5 ,
    "G" : 6 ,
    "H" : 7 ,
    "I" : 8 ,
    "J" : 9 ,
    "K" : 10 ,
    "L" : 11
}

#Defining the actions
actions = [ 0,1,2,3,4,5,6,7,8,9,10,11 ]

#definfing the rewards matrix
#need to give high reward leading from top priority to 2nd priority
# J -> K --> reward has to be greater 1 and less than 1000
# as the Q learning process favours leading from J to K
#DIFFICULT TO IMPLEMENT AUTOMATICALLY
R_additionalreward = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,1,0,500,0],
    [0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0,1,0],
]) 
# another way to do it is to -> bad reward from J to F! 
# Since bad reward - action not favoured 
#DIFFICULT TO IMPLEMENT AUTOMATICALLY
R_badreward = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,-500,0,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0,1,0],
]) 
#Another way-> make a best_route() function 
#inputs = starting, intermediate, ending loc
#calles previous_route() TWICE-a. from starting to intermideate loc b. from intermidate loc to ending loc

#defining rewards
R = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,1,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0,1,0],
]) 

## PART 2: BUILDING THE AI SOLUTION WITH Q-LEARNING
#mapping state to index vlaues
state_to_loc = {state : location for location, state in loc_to_state.items()}

#creating function that finds best route between starting location and ending location
def route(starting_location, ending_location):
    R_new = np.copy(R)
    ending_state = loc_to_state[ending_location]
    R_new[ending_state,ending_state] = 1000
    Q = np.array(np.zeros([12,12]))
    for i in range(1000):
        current_state = np.random.randint(0,12)
        playable_actions = []
        for j in range(12):
            if R_new[current_state,j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R_new[current_state,next_state] + gamma * Q[next_state, np.argmax(Q[next_state,])] - Q[current_state,next_state]
        Q[current_state,next_state] =Q[current_state,next_state] + alpha * TD
    route = [starting_location]
    next_location = starting_location
    while (starting_location != ending_location):
        starting_state = loc_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_loc[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

## PART 3 - GOING INTO PRODUCTION

#best_route() function to give route through intermediary location

def best_route(starting_location, intermediary_location, ending_location):
    return route(starting_location,intermediary_location) + route(intermediary_location, ending_location) [1:]

print("Best Route:")
print(best_route("E","K","G"))