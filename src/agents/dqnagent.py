"""
    DQN agent that uses the same experience replay as c51
    
    @author: Leno    

"""
from agents.agent import Agent
import time
import numpy as np
import os
from math import ceil,floor
import tensorflow as tf
import random
from numpy import float64


class DQNAgent(Agent):
    
    alpha = None
    environment = None
    
    rnd = None
    
    #Networks
    inputs, inputs_target = None,None
    soft_max, soft_max_target = None, None
    q,q_target = None, None
    q_best_act,q_best_act_target = None, None
    
    #Training variables
    actions = None
    isTerminal = None
    
     
    
    #Tensorflow session
    session = None
    
    #Network Optimizers
    optimizer = None
    
    saver = None
    
    replay_memory = None
    maxBatchSize = 10000
    miniBatchSize = 100
    learningSteps = None
    countReplayActions = None
    
    learningInterval = 25
    updateTargetInterval = 100
    learningEpochs = 1
    
    environmentActions = None
    loadWeights = None
    
    useBoltzmann = False
    
    
    n_hidden = 3
    n_neuronsHidden = 25 #50
    
    def __init__(self,seed=12345,alpha=0.01, epsilon=0.1, loadWeights=False):
        """
            Creates the C51 agent, initializing the main attributes.
            Some attributes will be initialized only when the connect_env function is called.
            seed: seed for reproducibility
            alpha: Learning rate for the Adam optimizer
            epsilon: parameter for epsilon-greedy exploration
            Vmin, Vmax, and N: parameters for the C51 distribution (see original paper)
            loadWeights: Should the agent load previously saved weights?          
        """
        super(DQNAgent, self).__init__(seed=seed)

        self.alpha = alpha
        self.epsilon = epsilon
        self.loadWeights = loadWeights
        self.gamma = 0.99
        
        
        self.rnd = random.Random(seed)
        
        self.replay_memory = []
        self.learningSteps = 0
        
        
        
    def connect_env(self,environment,agentIndex,allAgents):
        """Connects to the domain environment
             environment: HFO environment object
             agentIndex: Agent index in the evnvironment
        """
        super(DQNAgent, self).connect_env(environment,agentIndex,allAgents)
        self.environmentActions = environment.possible_actions()
        self.countReplayActions = np.zeros(len(self.environmentActions))
        #self.graph = tf.Graph()
        #self.session = keras.backend.get_session()
        #with self.session.as_default():
        #    with self.graph.as_default():
        self.build_network()

        self.update_target()

        if self.loadWeights:
            self.load_weights()
        
        
    def build_layers(self,trainable,prefix):
        """Creates a new Neural Network
            trainable: defines if the network is trainable or not (as in the case of the target network)
        """
        featureSize = len(self.environment.get_state(self.agentIndex)[1])
        n_act = len(self.environmentActions)  
        #Input layer
        #inputs =  keras.layers.Input(shape = (environment.N_INPUTS,))
        inputs = tf.placeholder(tf.float32, [None,featureSize], name = prefix+'Input')
        y_net = [None] * n_act
        befSoft = [None] * n_act
        #Hidden layers
        #First hidden Layer
        layerW = tf.get_variable(prefix+'W1', trainable = trainable, shape=(featureSize,self.n_neuronsHidden),
                                 initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                #tf.random_uniform([featureSize,self.n_neuronsHidden],seed = self.rnd.randint(0,1000),minval = 0.0001, maxval=0.1))
        layerB = tf.get_variable(prefix+'b1',trainable = trainable, shape = (self.n_neuronsHidden),
                                 initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                 #tf.random_uniform([self.n_neuronsHidden], seed = self.rnd.randint(0,1000)))
        hiddenL =  tf.add(tf.matmul(inputs,layerW),layerB)
        hiddenL = tf.nn.sigmoid(hiddenL)
        for i in range(1,self.n_hidden):
            layerW = tf.get_variable(prefix+'W'+str(i+1), trainable=trainable, shape=(self.n_neuronsHidden,self.n_neuronsHidden),
                                     initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                     #tf.random_uniform([self.n_neuronsHidden,self.n_neuronsHidden],seed = self.rnd.randint(0,1000),minval = 0.0001, maxval=0.1))
            layerB = tf.get_variable(prefix+'b'+str(i+1), trainable=trainable, shape=(self.n_neuronsHidden),
                                     initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                     #tf.random_uniform([self.n_neuronsHidden], seed = self.rnd.randint(0,1000)))
            hiddenL =  tf.add(tf.matmul(hiddenL,layerW),layerB)
            hiddenL = tf.nn.relu(hiddenL)
            
        #Last Layer, connection with the Actions
        layerW = tf.get_variable(prefix+'W'+str(self.n_hidden+1), trainable=trainable, shape= (self.n_neuronsHidden,n_act),
                                     initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                     #tf.random_uniform([self.n_neuronsHidden,self.N],seed = self.rnd.randint(0,1000),minval = 0.0001, maxval=0.1))
        layerB = tf.get_variable(prefix+'b'+str(self.n_hidden+1), trainable=trainable, shape= (n_act), 
                                     initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                     #tf.random_uniform([self.N], seed = self.rnd.randint(0,1000)))

        befSoft = tf.add(tf.matmul(hiddenL, layerW), layerB)
        y_net = tf.nn.softmax(befSoft)
        action = tf.argmax(y_net)
        
        return inputs,y_net,befSoft,action
    def build_network(self):
        """
            Builds the network to be used by the C51 agent. The network architecture depends on the 
            parameters n_hidden (number of hidden layers) and n_neuronsHidden (number of neurons in
            each hidden layer) as well as on the number of possible actions in the environment.
            This function must be called after the agent has been connected to the environment,
        """
        n_act = len(self.environmentActions)
        
        g = tf.Graph()

        with g.as_default():            
            #Builds both trainable and target networks
            self.inputs, self.soft_max,self.q,self.q_best_act = \
                                    self.build_layers(trainable = True, prefix = "net/")
            self.inputs_target, self.soft_max_target,self.q_target,self.q_best_act_target = \
                                    self.build_layers(trainable = False, prefix = "target/")
            

            # builds the operation to update the target network
            self.update_target_op = []
            trainable_variables = tf.trainable_variables()
            all_variables = tf.global_variables()
            for i in range(0, len(trainable_variables)):
                #print(trainable_variables[i].name + "->" +all_variables[len(trainable_variables) + i].name)
                self.update_target_op.append(all_variables[len(trainable_variables) + i].assign(trainable_variables[i]))

            
            #actions executed
            self.actions = tf.placeholder('int64', [None], name='action_train')
            #Actions suggested by
            self.next_acts = tf.placeholder('int64', [None], name='action_train')   
            self.isTerminal = tf.placeholder(tf.float32, [None], name='terminal_train')
            
            next_acts = tf.one_hot(self.next_acts, n_act, 1.0, 0.0, name='next_action_one_hot')
            target_max_q = tf.reduce_sum(self.q_target * next_acts, reduction_indices=1, name='qt_next')#tf.max(self.q_target)
            
            gamma = tf.convert_to_tensor(self.gamma)
            action_one_hot = tf.one_hot(self.actions, n_act, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
            target_q_t = (tf.convert_to_tensor(1.0)-self.isTerminal)*gamma*target_max_q
            
            delta = target_q_t - q_acted
                #cost[i] = tf.Print(cost[i], [cost[i]], "cost")
            self.cost = tf.reduce_mean(self.clipped_error(delta), name='loss')
                # add an optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.cost)
                
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.session = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.update_target()
            
    def clipped_error(self,x):
      # Huber loss
      try:
        return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
      except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
 
    def update_target(self):

        """Updates the target network with the current network weights"""
        g = self.session.graph

        with g.as_default():
            self.session.run(self.update_target_op)

    
    
    def delete_example(self):
        #Get the 
        maxInd = np.argmax(self.countReplayActions)
        self.countReplayActions[maxInd] -= 1
        
        delInd = next(x[0] for x in enumerate(self.replay_memory) if x[1][1] == self.environmentActions[maxInd])
        del self.replay_memory[delInd]
        return delInd
    def get_mini_batch(self):
      
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
            indexes.extend(random.sample(sampAct, int(actualNActions[i])))
        batch = [self.replay_memory[x] for x in indexes]
        
        return batch,indexes    
    
    def observe_reward(self,state,action,statePrime,reward):
        if self.exploring:
            self.learningSteps += 1
            if len(self.replay_memory) >= self.maxBatchSize:
                self.delete_example()#del self.replay_memory[0]
            self.replay_memory.append([state,action,statePrime,reward,self.environment.is_terminal_state()])
            actI = self.environmentActions.index(action)
            self.countReplayActions[actI] += 1
            
            #with self.session.as_default():
            #    with self.graph.as_default():
            g = self.session.graph

            with g.as_default():
                if self.learningSteps % self.learningInterval == 0:
                    batch,_ = self.get_mini_batch()#random.sample(self.replay_memory, min(self.miniBatchSize,len(self.replay_memory)))
                    self.train_network(batch)
                if self.learningSteps % self.updateTargetInterval == 0:
                    self.update_target()
                

            
    
    def train_network(self,batch):
        size_batch = len(batch)
        n_act = len(self.environmentActions)
        
        states = []
        statesPrime = []
        
    
        states = np.zeros((len(batch),len(batch[0][0][1])))
        actions = np.zeros((len(batch)))
        statesPrime = []
        rewards = np.zeros((len(batch)))
        terminal = np.zeros((len(batch)))
        
        i = 0
        for samples in batch:
            states[i,:] = samples[0][1]
            actions[i] = self.environmentActions.index(samples[1])
            statesPrime.append(samples[2])
            rewards[i] = samples[3]
            terminal[i] = 1. if samples[4] else 0.
            i = i + 1
        
        #statesPrime = np.array(statesPrime)
        next_acts = self.select_action(statesPrime, multipleOut=True, useNetwork=False) #use self.network
        next_acts = np.array([self.environmentActions.index(a) for a in next_acts])
        
        
        statesPrime = np.array([statep[1] for statep in statesPrime])
        
        self.optimizer.run(session=self.session, feed_dict = {self.inputs : states, self.inputs_target : statesPrime,
                                                              self.actions: actions, self.next_acts : next_acts,
                                                              self.isTerminal: terminal } )   
        
        
  

    def calc_Q(self,state,action,useNetwork=True):
        """Calculates the Q-value for a given state and action
            useNetwork: If true, uses the trainable network, otherwise uses the
                target network.
        """
        if useNetwork:
            q_func = self.q
            input = self.inputs
        else:
            q_func = self.q_target
            input = self.inputs_target
            
        act_idx = self.environmentActions.index(action)
        
        g = self.session.graph
        
        with g.as_default():
            qs =q_func.eval(session=self.session, feed_dict = {input : np.array([state])})
        #print("Qs(s) : " + str(qs))
        return qs[0][act_idx] 
    
    
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
                return random.choice(self.environment.all_actions(states,self.agentIndex))
                        
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
                
                if self.useBoltzmann:
                    act_vals = np.array([self.calc_Q(state,act,useNetwork) for act in possibleActions])
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
                        qV = self.calc_Q(state,act,useNetwork)
                        if qV > maxV:
                            maxV = qV
                            maxAct = [act]
                        elif qV == maxV:
                            maxAct.append(act)
                    #print(str(state) + str(possibleActions))
                    
                    return_act.append(random.choice(maxAct))
        if not multipleOut:
            return return_act[0]
        return return_act
    
    def get_used_budget(self):
        return 0.0
           
    def finish_learning(self):
        """Saves the weight after learning finishes"""
        fileFolder = "./agentFiles/DQN/"
        if not os.path.exists(fileFolder):
            os.makedirs(fileFolder)
            
        if self.environment.numberFriends == 0:
            filePath = fileFolder + "DQNModel.ckpt"
        else:
            filePath = fileFolder + "DQNModel" + str(self.agentIndex) + ".ckpt"
        
        self.saver.save(self.session,filePath)
        self.session.close()
                   
    def load_weights(self): 
        """Loads previously saved weight files"""
        fileFolder = "./agentFiles/DQN/"
        if self.environment.numberFriends == 0:
            filePath = fileFolder + "DQNModel.ckpt"
        else:
            filePath = fileFolder + "DQNModel" + str(self.agentIndex) + ".ckpt"
        self.saver.restore(self.session, filePath)  
        
    