"""
    This is precisely the same implementation as in the C51Agent. However, we change a little the
    procedure to select a mini-batch from the replay memory to make sure that all agents in the system will
    update their policy by selecting samples from the same time steps
    
    @author: Leno    

"""
from agents.c51agent import C51Agent
from time import sleep


class C51SyncAgent(C51Agent):
    
    batchCount = None
    lastBatchIndexes = None
    lastDeleteIndex = None
    s_lockKey,d_lockKey = None,None
    def __init__(self,seed=12345,alpha=0.01, epsilon=0.1,Vmin = -1.5,Vmax = 1.5, N=51, loadWeights=False):
        """
            Creates the C51 agent, initializing the main attributes.
            Some attributes will be initialized only when the connect_env function is called.
            seed: seed for reproducibility
            alpha: Learning rate for the Adam optimizer
            epsilon: parameter for epsilon-greedy exploration
            Vmin, Vmax, and N: parameters for the C51 distribution (see original paper)
            loadWeights: Should the agent load previously saved weights?          
        """
        super(C51SyncAgent, self).__init__(seed=seed,alpha=alpha, epsilon=epsilon,Vmin = Vmin,Vmax = Vmax, N=N, loadWeights=loadWeights)
        self.batchCount = 0      

    def get_mini_batch(self):
        
        #One of the agent will assign itself to select the minibatch and the others will follow
        agentTurn = self.batchCount % len(self.allAgents)
        if agentTurn == self.agentIndex:
            #Selects the batch itself
            batch,indexes = super(C51SyncAgent, self).get_mini_batch()
            self.lastBatchIndexes = indexes
            self.s_lockKey = 0
        else:
            while self.allAgents[agentTurn].s_lockKey is None:
                print("waiting " + str(self.agentIndex)+ " - " + str(self.allAgents[agentTurn].s_lockKey))
                sleep(0.001)
                #pass
            self.allAgents[agentTurn].s_lockKey += 1
            if self.allAgents[agentTurn].s_lockKey == len(self.allAgents)-1:
                self.allAgents[agentTurn].s_lockKey = None
            indexes = self.allAgents[agentTurn].lastBatchIndexes
            batch = [self.replay_memory[i] for i in indexes]
            
        self.batchCount += 1
        return batch,indexes
    def delete_example(self):
        agentTurn = self.batchCount % len(self.allAgents)
        if agentTurn == self.agentIndex:
            #Selects the batch itself
            index = super(C51SyncAgent, self).delete_example()
            self.lastDeleteIndex = index
            self.d_lockKey = 0
        else:
            while self.allAgents[agentTurn].d_lockKey is None:
                print("waiting " + str(self.agentIndex))
                #pass
            self.allAgents[agentTurn].d_lockKey += 1
            if self.allAgents[agentTurn].d_lockKey == len(self.allAgents)-1:
                self.allAgents[agentTurn].d_lockKey = None
            index = self.allAgents[agentTurn].lastDeleteIndex
            act_index = self.environmentActions.index(self.replay_memory[index][1])
            self.countReplayActions[act_index] -= 1
            del self.replay_memory[index]
        return index
      
    def finish_learning(self):
        """Saves the weight after learning finishes"""
        fileFolder = "./agentFiles/C51Sync/"
        if not os.path.exists(fileFolder):
            os.makedirs(fileFolder)
            
        if self.environment.numberFriends == 0:
            filePath = fileFolder + "C51Model.ckpt"
        else:
            filePath = fileFolder + "C51Model" + str(self.agentIndex) + ".ckpt"
        
        self.saver.save(self.session,filePath)
        self.session.close()
                   
    def load_weights(self): 
        """Loads previously saved weight files"""
        fileFolder = "./agentFiles/C51Sync/"
        if self.environment.numberFriends == 0:
            filePath = fileFolder + "C51Model.ckpt"
        else:
            filePath = fileFolder + "C51Model" + str(self.agentIndex) + ".ckpt"
        self.saver.restore(self.session, filePath)  
    
 