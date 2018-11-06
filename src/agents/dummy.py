import sys
import random

from .agent import Agent

class Dummy(Agent):

    def __init__(self, seed=12345):
        super(Dummy, self).__init__(seed=seed)
        
    def select_action(self, state):
        """ When this method is called, the agent executes an action. """
        act = random.choice(self.environment.all_actions())
        #print(state)
        #print("***Chosen action: "+str(act)+"  Agent: "+str(self.environment.get_unum(self.agentIndex)))
        return act
    
    def observe_reward(self,state,action,statePrime,reward):
        """ After executing an action, the agent is informed about the state-action-reward-state tuple """
        pass  
    def get_used_budget(self):
        return 0
    