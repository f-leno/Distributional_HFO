FIFO = 0
BALANCED_ACTIONS = 1
PRIORITIZED = 2

class BatchController():
    agent = None
    type_batch = None
    aux_obj = None
    
    def __init__(self,agent,type_batch):
        self.agent = agent
        self.type_batch = type_batch
        
        if type_batch not in [FIFO, BALANCED_ACTIONS, PRIORITIZED]:
            raise Exception("Unknown type of batch: " + str(type_batch))
        if type_batch == PRIORITIZED:
            self.aux_obj = ReplayMemory()
        
    def get_mini_batch(self):
        """Selects minibatch samples and return them"""
        if type_batch == FIFO:
            return get_batch_fifo(self)
        elif type_batch == BALANCED_ACTIONS:
            return get_batch_balanced(self)
            
        

    def delete_sample(self):
        """ Deletes one example from the bath, self is the agent"""
        if self.type_batch == FIFO:
            return delete_fifo(self.agent)
        elif self.type_batch == BALANCED_ACTIONS:
            return delete_balanced(self.agent)
    
    
    def add_sample(self, sample):
        if len(self.agent.replay_memory) >= self.agent.maxBatchSize:
                self.delete_sample()#del self.replay_memory[0]
        self.agent.replay_memory.append(sample)
        
        if self.type_batch == BALANCED_ACTIONS:
            actI = self.agent.environmentActions.index(sample[1])
            self.agent.countReplayActions[actI] += 1
        
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
    
    
def get_batch_balanced(self):
    #If an equal number of examples for each action can be chosen
    n_acts = len(self.environmentActions)
    expectedNumber = int(floor(min(self.miniBatchSize,len(self.replay_memory)) / n_acts))
    actualNActions = np.zeros((n_acts))
    remainingExamples = 0
    for i in range(n_acts):
        if self.countReplayActions[i] < expectedNumber:
            actualNActions[i] = self.countReplayActions[i]
            remainingExamples += expectedNumber - actualNActions[i]
        else:
            actualNActions[i] = expectedNumber
    while remainingExamples > 0:
        #Number of actions that still have remaining examples in the replay buffer
        n_remaining =  sum([1 for i in range(n_acts) if actualNActions[i] < self.countReplayActions[i]])
        distribExamples = ceil(remainingExamples / n_remaining)
        
        for i in range(n_acts):
            if actualNActions[i] < self.countReplayActions[i]:
                usableEx = min(distribExamples, self.countReplayActions[i] - actualNActions[i])
                actualNActions[i] += usableEx
                remainingExamples -= usableEx
    
    indexes = []
    for i in range(n_acts):
        sampAct = [x for x in range(len(self.replay_memory)) if self.replay_memory[x][1]==self.environmentActions[i]]
        indexes.extend(self.rnd.sample(sampAct, int(actualNActions[i])))
    batch = [self.replay_memory[x] for x in indexes]
    
    return batch,indexes 
    

