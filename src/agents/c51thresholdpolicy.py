"""
    C51Threshold implementation. Almost all code inside c51agent.py is reused, except the codification for
    deriving the policy from the distribution.
    
    @author: Leno    

"""
from agents.c51agent import C51Agent
import numpy as np
import environment.hfoactions as hfoactions
import agents.batch_util as batch_util


class C51ThresholdPolicy(C51Agent):
    
    PROB_SHOOT = 0.5#0.5
    PROB_PASS = 0.5
    
    def __init__(self,seed=12345,saveWeightsInterval = 500,alpha=0.01, epsilon=0.1,Vmin = -1.5,Vmax = 1.5, N=51, loadWeights=False, loadStep = None):
        """
            Creates the C51 agent, initializing the main attributes.
            Some attributes will be initialized only when the connect_env function is called.
            seed: seed for reproducibility
            alpha: Learning rate for the Adam optimizer
            epsilon: parameter for epsilon-greedy exploration
            Vmin, Vmax, and N: parameters for the C51 distribution (see original paper)
            loadWeights: Should the agent load previously saved weights?          
        """
        super(C51ThresholdPolicy, self).__init__(seed=seed,alpha = alpha,saveWeightsInterval = saveWeightsInterval, epsilon = epsilon, Vmin = Vmin, Vmax = Vmax, N=N, loadWeights = loadWeights, loadStep = loadStep)
        self.className = "C51Threshold"
        #self.batch_type = batch_util.BALANCED_ACTIONS
        
        #self.batchController = batch_util.BatchController(self,self.batch_type)
        
        
    def select_action(self,states,multipleOut=False,useNetwork=False):
        """Select the action for the current state, can also be used for multiple states if multipleOut == True
            states: current state or list of states in which the agent should choose an action.
            multipleOut: must be True if a list of states is given
            useNetwork: True if the trainable network should be used, False if the target one should be
        """
        return_act = []
        
        if self.exploring:
            rV = self.rnd.random()
            
            if rV <= self.epsilon and not multipleOut:
                return self.rnd.choice(self.environment.all_actions(states,self.agentIndex))
                        
        if isinstance(states,tuple):
            states = [states]
          
        #with self.session.as_default(): 
        #    with self.graph.as_default():  
        for state in states:
            possibleActions = self.environment.all_actions(state,self.agentIndex)
            if len(possibleActions)==1:
                return_act.append(possibleActions[0])
            else:
                state = np.array(state[1])
                
                chosenAct=self.choose_act(state,possibleActions,useNetwork)
                    
                return_act.append(chosenAct)
        if not multipleOut:
            return return_act[0]
        return return_act
    
    def choose_act(self,state,possibleActions, useNetwork):
        """" 
            Selects the action according to the distribution:
             Shoots when PROB_SHOOT% of probability of receiving a return greater than 0
             Passes when PROB_PASS% of probability of receiving a return greater than 0
             DRIBBLE otherwise
        """
        evaluatePass = True

        #if shooting is a feasible action
        action = hfoactions.get_shoot() 
    
        #Get the probabilitys
        prob_vec = self.get_distrib(state,action,useNetwork)
        prob_greater_zero = sum(np.extract(self.z_vec >= 0., prob_vec))

        if prob_greater_zero > self.PROB_SHOOT:
            return action

    
        if evaluatePass:
            #Do the same for all pass actions
            for act in possibleActions:
                if hfoactions.is_pass_action(act):
                     #Get the probabilitys
                    prob_vec = self.get_distrib(state,act,useNetwork)
                    prob_greater_zero = sum(np.extract(self.z_vec >= 0., prob_vec))
    
                    if prob_greater_zero > self.PROB_PASS:
                        return act
        else:
            action = hfoactions.get_dribble() 

            #Get the probabilitiess
            prob_vec = self.get_distrib(state,action,useNetwork)
            prob_greater_zero = sum(np.extract(self.z_vec >= 0., prob_vec))
    
            if prob_greater_zero < self.PROB_PASS:
                bestPassAction = None
                bestVal = -float('inf')
                for act in possibleActions:
                    if hfoactions.is_pass_action(act):
                        q = self.calc_Q(state,act)
                        if q > bestVal:
                            bestPassAction = act
                            bestVal = q
                if bestPassAction is not None:
                    return bestPassAction
                    
            

        return hfoactions.get_dribble()

