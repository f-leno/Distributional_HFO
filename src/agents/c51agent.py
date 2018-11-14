"""
    Agent implementing the distributional RL algorithm C51, 
    described in "A distributional Perspective on RL" by Marc Bellemare and others.
    Some parts were based on the implementation available at:
    https://github.com/flyyufelix/C51-DDQN-Keras/blob/master/networks.py
    
    @author: Leno    

"""
from agents.agent import Agent
import time
import numpy as np
import os
from math import ceil,floor
import keras
import tensorflow as tf
import random
from numpy import float64


class C51Agent(Agent):
    
    Vmin = None
    Vmax = None
    N = None
    alpha = None
    deltaZ = None
    z_vec = None
    environment = None
    
    rnd = None
    
    network = None
    targetNet = None
    
    replay_memory = None
    maxBatchSize = 10000
    miniBatchSize = 100
    learningSteps = None
    countReplayActions = None
    
    learningInterval = 25
    updateTargetInterval = 100
    
    environmentActions = None
    loadWeights = None
    
    useBoltzmann = False
    useThreeNetworks = True
    
    n_hidden = 5
    n_neuronsHidden = 50
    
    def __init__(self,seed=12345,alpha=0.01, epsilon=0.1,Vmin = -1.5,Vmax = 1.5, N=51, loadWeights=False):
        super(C51Agent, self).__init__(seed=seed)
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.N = N
        self.alpha = alpha
        self.deltaZ = (Vmax - Vmin) / (N-1)
        self.z_vec = np.array([Vmin + i*self.deltaZ for i in range(N)])
        self.epsilon = epsilon
        self.loadWeights = loadWeights
        self.gamma = 0.99
        
        
        self.rnd = random.Random(seed)
        np.random.seed(self.rnd.randint(0, 10000))
        tf.set_random_seed(self.rnd.randint(0, 10000))
        
        self.replay_memory = []
        self.learningSteps = 0
        
        
    def connect_env(self,environment,agentIndex):
        """Connects to the domain environment"""
        super(C51Agent, self).connect_env(environment,agentIndex)
        self.environmentActions = environment.possible_actions()
        self.countReplayActions = np.zeros(len(self.environmentActions))
        self.build_network()
        #self.update_target()
        if self.loadWeights:
            self.load_weights()
        
    
    def build_network(self):
        """
            Builds the network to be used by the C51 agent. The network architecture depends on the 
            parameters n_hidden (number of hidden layers) and n_neuronsHidden (number of neurons in
            each hidden layer) as well as on the number of possible actions in the environment.
            This function must be called after the agent has been connected to the environment,
        """
        featureSize = len(self.environment.get_state(self.agentIndex)[1])
        #Input layer
        inputs =  keras.layers.Input(shape = (featureSize,))
        #inputs = keras.layers.Input(shape=(environment.N_INPUTS,))
        
        net = keras.layers.Dense(self.n_neuronsHidden,activation='relu')(inputs)
        
        #Hidden layers
        for i in range(self.n_hidden-1):
            net = keras.layers.Dense(self.n_neuronsHidden, activation='relu')(net)
        n_act = len(self.environmentActions) 
        
        
        if self.useThreeNetworks:
            self.network = []
            self.targetNet = [] 
        
            actLayer = keras.layers.Dense(self.N, activation='softmax')(net)
            
            network = keras.models.Model(inputs = inputs, outputs=actLayer)         
            
            for i in range(n_act):
                self.network.append(keras.models.clone_model(model=network))
                optimizer = keras.optimizers.Adam(lr = self.alpha)
                self.network[i].compile(optimizer = optimizer,
                          loss='categorical_crossentropy'
                          )
                self.targetNet.append(keras.models.clone_model(model=self.network[i]))
                
                
        else:            
            actLayers = []   
            
            
            for i in range(n_act):
                actLayers.append(keras.layers.Dense(self.N, activation='softmax')(net))
            
            self.network = keras.models.Model(inputs = inputs, outputs=actLayers)
            optimizer = keras.optimizers.Adam(lr = self.alpha)
            self.network.compile(optimizer = optimizer,
                          loss='categorical_crossentropy'
                          )
            self.targetNet = keras.models.clone_model(model=self.network)
            #self.targetNet.predict(np.zeros((1,featureSize)))
        
       
        
        
    def update_target(self):
        if self.useThreeNetworks:
            for i in range(len(self.network)):
                self.targetNet[i].set_weights(self.network[i].get_weights())
        else:
            self.targetNet.set_weights(self.network.get_weights())
    
    
    def calc_z_i(self,i):
        return self.Vmin + i*self.deltaZ
    def prob(self,i,state,action):
        """
            check here, how to extract from the NN the parametric model?
        """
        probs = self.network.predict(np.array(state))[0]
        probs_act = probs[self.environmentActions.index(action)]
        
        #assert len(probs_act) == 20
        #assert np.sum(probs_act) < 1.1 and np.sum(probs_act) > 0.9
        
        return probs_act[i]
    
    def delete_example(self):
        #Get the 
        maxInd = np.argmax(self.countReplayActions)
        self.countReplayActions[maxInd] -= 1
        
        delInd = next(x[0] for x in enumerate(self.replay_memory) if x[1][1] == self.environmentActions[maxInd])
        del self.replay_memory[delInd]
    def get_mini_batch(self):
        #If an equal number of examples for each action can be chosen
        n_acts = len(self.environmentActions)
        expectedNumber = int(floor(min(self.miniBatchSize,len(self.replay_memory)) / n_acts))
        actualNActions = np.zeros((n_acts,1))
        remainingExamples = 0
        for i in range(n_acts):
            if self.countReplayActions[i] < expectedNumber:
                actualNActions[i] = self.countReplayActions[i]
                remainingExamples += expectedNumber - actualNActions[i]
            else:
                actualNActions[i] = expectedNumber
        while remainingExamples != 0:
            #Number of actions that still have remaining examples in the replay buffer
            n_remaining =  sum([1 for i in range(n_acts) if actualNActions[i] < self.countReplayActions[i]])
            distribExamples = remainingExamples / n_remaining
            
            for i in range(n_acts):
                if actualNActions[i] < self.countReplayActions[i]:
                    usableEx = min(distribExamples, self.countReplayActions[i] - actualNActions[i])
                    actualNActions[i] += usableEx
                    remainingExamples -= usableEx
        
        batch = []
        for i in range(n_acts):
            sampAct = [x for x in self.replay_memory if x[1]==self.environmentActions[i]]
            batch.extend(random.sample(sampAct, int(actualNActions[i])))
        return batch    
    
    def observe_reward(self,state,action,statePrime,reward):
        if self.exploring:
            self.learningSteps += 1
            if len(self.replay_memory) >= self.maxBatchSize:
                self.delete_example()#del self.replay_memory[0]
            self.replay_memory.append([state,action,statePrime,reward,self.environment.is_terminal_state()])
            actI = self.environmentActions.index(action)
            self.countReplayActions[actI] += 1
            
            if self.learningSteps % self.learningInterval == 0:
                batch = self.get_mini_batch()#random.sample(self.replay_memory, min(self.miniBatchSize,len(self.replay_memory)))
                self.train_network(batch)
            if self.learningSteps % self.updateTargetInterval == 0:
                self.update_target()
            

            
    
    def train_network(self,batch):
        size_batch = len(batch)
        n_act = len(self.environmentActions)
        
        states = []
        statesPrime = []
        
    
        m = [np.zeros((size_batch,self.N)) for i in range(n_act)]  # n_actions x size_batch x self.N
        actions = []
        rewards = []
        terminal = []
        
        for samples in batch:
            states.append(samples[0][1])
            actions.append(self.environmentActions.index(samples[1]))
            statesPrime.append(samples[2])
            rewards.append(samples[3])
            terminal.append(samples[4])
        states = np.array(states)
        #statesPrime = np.array(statesPrime)
        
        next_acts = self.select_action(statesPrime, multipleOut=True, network=self.network) #use self.network
        next_acts = [self.environmentActions.index(a) for a in next_acts]
        
        z_prime = None
        statesPrime = np.array([statep[1] for statep in statesPrime])
        if self.useThreeNetworks:
            z_prime = []
            for target in self.targetNet:
                z_prime.append(target.predict(statesPrime))
        else:
            z_prime = self.targetNet.predict(statesPrime)  #Use target network
        
        #Calculate projection (m)
        for i in range(size_batch):
            if terminal[i]:
                Tz = np.clip(rewards[i], self.Vmin, self.Vmax)
                bj = (Tz - self.Vmin) / self.deltaZ
                m_l = np.floor(bj)
                m_u = np.ceil(bj)
                m[actions[i]][i][int(m_l)] += (m_u - bj)
                m[actions[i]][i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.N):
                    Tz = np.clip(rewards[i] + self.gamma*self.z_vec[j], self.Vmin, self.Vmax)
                    bj = (Tz - self.Vmin) / self.deltaZ
                    m_l = np.floor(bj)
                    m_u = np.ceil(bj)
                    m[actions[i]][i][int(m_l)] += z_prime[next_acts[i]][i][j] * (m_u - bj)
                    m[actions[i]][i][int(m_u)] += z_prime[next_acts[i]][i][j] * (bj - m_l)
                    
        if self.useThreeNetworks:
            for i in range(len(self.network)):
                indexes = np.array(actions) == i
                statesNet = states[indexes]
                if len(statesNet) > 0:
                    self.network[i].fit(statesNet, m[i][indexes][:], batch_size=len(statesNet), verbose=0, epochs=2)
        else:
            self.network.fit(states, m, batch_size=size_batch, verbose=0)
                      
        
        
        
  

    def calc_Q(self,state,action,network=None):
        prob_vec = self.get_distrib(state,action,network)
        value = np.dot(self.z_vec, prob_vec)
        return value
    
    def get_distrib(self,state,action,network=None):
        if network is None:
            network = self.network
        act_idx = self.environmentActions.index(action)
        distrib = None
        if self.useThreeNetworks:
            #print(str(len(state)) + " - " + str(np.array([state]).shape) + str(np.array([state]).dtype))
            distrib = network[act_idx].predict(np.array([state]))
            distrib = distrib[0]
        else:
            distrib = network.predict(np.array([state]))
            distrib = distrib[act_idx][0]
        
        return distrib #np.array([self.prob(i,state,action) for i in range(self.N)])

    def select_action(self,states,multipleOut=False,network=None):
        return_act = []
        
        if self.exploring:
            rV = self.rnd.random()
            
            if rV <= self.epsilon and not multipleOut:
                return random.choice(self.environment.all_actions(states,self.agentIndex))
                
        if network is None:
            network = self.targetNet
        
        if isinstance(states,tuple):
            states = [states]
            
            
        for state in states:
            possibleActions = self.environment.all_actions(state,self.agentIndex)
            if len(possibleActions)==1:
                return_act.append(possibleActions[0])
            else:
                state = np.array(state[1])
                
                if self.useBoltzmann:
                    act_vals = np.array([self.calc_Q(state,act,network) for act in possibleActions])
                    act_vals = act_vals - min(act_vals) + 0.00001 #Avoiding division by 0
                    sum_vals = sum(act_vals)
                    act_vals = act_vals / sum_vals
                    rV = self.rnd.random()
                    summedVal = 0.0
                    currentIndex = 0
                    
                    while summedVal+act_vals[currentIndex] < rV:
                        summedVal += act_vals[currentIndex]
                        currentIndex += 1
                    return_act.append(possibleActions[currentIndex])
                    
                else:
                    maxV = -float('inf')
                    maxAct = None
                    for act in possibleActions:
                        #print(state)
                        qV = self.calc_Q(state,act,network)
                        if qV > maxV:
                            maxV = qV
                            maxAct = [act]
                        elif qV == maxV:
                            maxAct.append(act)
                    return_act.append(random.choice(maxAct))
        if not multipleOut:
            return return_act[0]
        return return_act
    
    def get_used_budget(self):
        return 0.0
           
    def finish_learning(self):
        """Saves the weight after learning finishes"""
        fileFolder = "./agentFiles/C51/"
        if not os.path.exists(fileFolder):
            os.makedirs(fileFolder)
        if self.useThreeNetworks:
            for i in range(len(self.network)):
                self.network[i].save_weights(fileFolder+"C51Model"+str(i)+".h5")
        else:
            self.network.save_weights(fileFolder+"C51Model.h5")           
    def load_weights(self): 
        """Loads previously saved weight files"""
        fileFolder = "./agentFiles/C51/"
        if self.useThreeNetworks:
            for i in range(len(self.network)):
                self.network[i].load_weights(fileFolder+"C51Model"+str(i)+".h5")
        else:
            self.network.load_weights(fileFolder+"C51Model.h5")  
        self.update_target()   
        
        