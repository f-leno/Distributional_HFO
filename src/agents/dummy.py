import sys
import random

from .agent import Agent

class Dummy(Agent):
    """Random Agent"""
    def __init__(self, seed):
        super(Dummy, self).__init__(seed=seed, port=port,serverPath=serverPath)


    def select_action(self,stateFeatures, state):
        """ When this method is called, the agent executes an action. """
        
        
        return random.choice(self.environment.all_actions(forExploration=True))
    def advise_action(self,uNum,state):
        """Verifies if the agent can advice a friend, and return the action if possible"""
        return None #No advising


    def observe_reward(self,state,action,reward,statePrime):
        """ After executing an action, the agent is informed about the state-reward-state tuple """
        pass
    def setupAdvising(self,agentIndex,allAgents):
        """ This method is called in preparation for advising """
        pass

