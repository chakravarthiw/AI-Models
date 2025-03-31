import numpy as np

#setting constant values for Q-learning algorithm
gamma = 0.75 #discount factor
alpha = 0.9 #learning rate

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
R = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,0,0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0],
    [0,1,0,0,0,0,0,0,0,1,0,0],
    [0,0,1,0,0,0,1000,1,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,1,0,0,1,0,1,0],
    [0,0,0,0,0,0,0,0,0,1,0,1],
    [0,0,0,0,0,0,0,1,0,0,1,0],
]) #location at cell (g,g) = 1000 as it is top priority

#BUILDING AI SOLUTION

#initilizing Q values as a matrix of 12*12 of '0' because at time t=0 all Q values =0
# Every move is equally unknown right now — let’s treat them all as zero-value.
Q = np.array(np.zeros([12,12]))
#Q - learning algorithm implementation
for i in range(1000):
    current_state = np.random.randint(0,12) #Pick a random starting location** (0 to 11).
    playable_actions = []
    for j in range(12): #figure out where the robot can legally go from its current state
        if R[current_state, j] > 0: #check reward matrix
            playable_actions.append(j) #collect all possible valid moves
    next_state = np.random.choice(playable_actions)#Pick a **random action** from the valid ones.
    # TD - learning signal - tells how much better or worse move was compared to expectations
    #R[current_state, next_state] = the reward for taking this step
    #Q[next_state, np.argmax(Q[next_state,])] = the best possible future value from the next state
    #Q[current_state, next_state] = what we currently believe about this move
    #“New Value = reward + value of future moves — old value”
    TD = R[current_state,next_state] + gamma*Q[next_state,np.argmax(Q[next_state,])] - Q[current_state,next_state]
    #Q[current_state,next_state] = Q[current_state,next_state] + alpha*TD -> is the update step:
    #adjust Q value for the move:
    #   a. TD is positive → increase it
    #   b. TD is negative → decrease it
    Q[current_state,next_state] = Q[current_state,next_state] + alpha*TD


#print("Q-Values:")
#print(Q.astype(int))
    
## Part 3: Going into PRODUCTION

#mapping the states to location
state_to_location = {state: location for location, state in loc_to_state.items()}
#defining the route function for the AI
def route(starting_location,ending_location):
    route = [starting_location]#Start route with the starting location
    next_location = starting_location
    while(starting_location != ending_location):#Keep going until we reach the goal
        starting_state = loc_to_state[starting_location]## Convert letter to number
        next_state = np.argmax(Q[starting_state,])#best next state (where Q-value is highest)
        next_location = state_to_location[next_state]#convert state to letter
        route.append(next_location)
        starting_location = next_location
    return(route)

print('Route:')
print(route('E','G'))



