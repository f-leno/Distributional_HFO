"""
    Functions to control experience replay batches storage and mini-batch selection
    author: Leno.

"""


import random
import numpy as np
from agents.memory import PrioritizedReplayMemory
from math import floor,ceil


# 3 main strategies were implemented, FIFO simply keeps in memory the newest samples and selects
# random ones for mini-batchs. BALANCED_ACTIONS tries to store the same number of samples for each action
# and to select a balanced number of samples in the mini-batch. PRIORITIZED is the prioritized experience replay
# implementation given by Ruben Glatt.
FIFO = 0
BALANCED_ACTIONS = 1
PRIORITIZED = 2

class BatchController():
    agent = None
    type_batch = None
    aux_obj = None
    
    def __init__(self,agent,type_batch):
        """
            agent: reference to the agent object (the buffer is in the agent class)
            type_batch: either FIFO, BALANCED_ACTIONS, or PRIORITIZED.
        """
        self.agent = agent
        self.type_batch = type_batch
        
        if type_batch not in [FIFO, BALANCED_ACTIONS, PRIORITIZED]:
            raise Exception("Unknown type of batch: " + str(type_batch))
        if type_batch == PRIORITIZED:
            #Interface for Ruben's implementation
            params = {"mem_max_priority": 2.0,
                      "rng": random.Random(agent.rnd.randint(0,1000)),
                      "memory_size":agent.maxBatchSize,
                      "prioritize_bias":0.
                      }
            self.aux_obj = PrioritizedReplayMemory(params=params)
            agent.replay_memory = [None]*agent.maxBatchSize
        else:
            agent.replay_memory = []
            
        
    def get_mini_batch(self):
        """Selects minibatch samples and return them"""
        if self.type_batch == FIFO:
            return get_batch_fifo(self)
        elif self.type_batch == BALANCED_ACTIONS:
            return self.get_batch_balanced()
        elif self.type_batch == PRIORITIZED:
            return self.get_batch_prioritized()
            
    def get_batch_prioritized(self):
        index = self.aux_obj.get_minibatch_indices(min(self.aux_obj.current_size, self.agent.miniBatchSize))
        batch = [self.agent.replay_memory[i] for i in index]
        return batch,index
    
    def get_batch_balanced(self):
        #If an equal number of examples for each action can be chosen
        n_acts = len(self.agent.environmentActions)
        expectedNumber = int(floor(min(self.agent.miniBatchSize,len(self.agent.replay_memory)) / n_acts))
        actualNActions = np.zeros((n_acts))
        remainingExamples = 0
        for i in range(n_acts):
            if self.agent.countReplayActions[i] < expectedNumber:
                actualNActions[i] = self.agent.countReplayActions[i]
                remainingExamples += expectedNumber - actualNActions[i]
            else:
                actualNActions[i] = expectedNumber
        while remainingExamples > 0:
            #Number of actions that still have remaining examples in the replay buffer
            n_remaining =  sum([1 for i in range(n_acts) if actualNActions[i] < self.agent.countReplayActions[i]])
            distribExamples = ceil(remainingExamples / n_remaining)
            
            for i in range(n_acts):
                if actualNActions[i] < self.agent.countReplayActions[i]:
                    usableEx = min(distribExamples, self.agent.countReplayActions[i] - actualNActions[i])
                    actualNActions[i] += usableEx
                    remainingExamples -= usableEx
        
        indexes = []
        for i in range(n_acts):
            sampAct = [x for x in range(len(self.agent.replay_memory)) if self.agent.replay_memory[x][1]==self.agent.environmentActions[i]]
            indexes.extend(self.agent.rnd.sample(sampAct, int(actualNActions[i])))
        batch = [self.agent.replay_memory[x] for x in indexes]
        
        return batch,indexes 

    def delete_sample(self):
        """ Deletes one example from the bath, self is the agent"""
        if self.type_batch == FIFO:
            return delete_fifo(self.agent)
        elif self.type_batch == BALANCED_ACTIONS:
            return delete_balanced(self.agent)
        
    
    
    def add_sample(self, sample):
        if self.type_batch == PRIORITIZED:
            self.agent.replay_memory[self.aux_obj.current_position] = sample
            self.aux_obj.add()            
        else:
            if len(self.agent.replay_memory) >= self.agent.maxBatchSize:
                    self.delete_sample()#del self.replay_memory[0]
            self.agent.replay_memory.append(sample)
            
            if self.type_batch == BALANCED_ACTIONS:
                actI = self.agent.environmentActions.index(sample[1])
                self.agent.countReplayActions[actI] += 1
    
    def batch_update(self, args):
        """If the batch selection method receives any value after update, it is computed here"""
        if self.type_batch == PRIORITIZED:
            indexes = args[0]
            importance = args[1]
            self.aux_obj.batch_update(indexes, importance)
        
def delete_balanced(self):
    """Deletes one example from the batch and returns the index"""
    maxInd = np.argmax(self.countReplayActions)
    self.countReplayActions[maxInd] -= 1
   
    delInd = next(x[0] for x in enumerate(self.replay_memory) if x[1][1] == self.environmentActions[maxInd])
    del self.replay_memory[delInd]
    return delInd

def delete_fifo(self):
    del self.replay_memory[0]
    return 0

    
    

    
def get_batch_fifo(self):
    indexes = self.rnd.sample(range(len(self.replay_memory)),min(len(self.replay_memory),self.miniBatchSize))
    batch = [self.replay_memory[x] for x in indexes]    
    return batch,indexes 
    
    

    

