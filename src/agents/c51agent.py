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
import tensorflow as tf
import random
from numpy import float64
import errno

import agents.batch_util as batch_util

class C51Agent(Agent):
    
    Vmin = None
    Vmax = None
    N = None
    alpha = None
    deltaZ = None
    z_vec = None
    environment = None
    
    rnd = None
    
    #Networks
    y_hat, y_hat_target = None,None
    inputs, inputs_target = None,None
    
    #Placeholder to use during training
    y = None
    
    #Tensorflow session
    session = None
    
    #Network Optimizers
    optimizers = None
    
    saver = None
    
    batchController = None
    replay_memory = None
    maxBatchSize = 10000
    miniBatchSize = 100
    learningSteps = None
    countReplayActions = None
    batch_type = batch_util.PRIORITIZED
    
    learningInterval = 25
    updateTargetInterval = 100
    learningEpochs = 1
    
    environmentActions = None
    loadWeights = None
    loadStep = None
    
    useBoltzmann = False
    useThreeNetworks = True
    
    n_hidden = 3
    n_neuronsHidden = 25 #50
    
    className = None
     
    
    def __init__(self,seed=12345,saveWeightsInterval = 500, alpha=0.01, epsilon=0.1,Vmin = -1.5,Vmax = 1.5, N=51, loadWeights=False, loadStep = None):
        """
            Creates the C51 agent, initializing the main attributes.
            Some attributes will be initialized only when the connect_env function is called.
            seed: seed for reproducibility
            alpha: Learning rate for the Adam optimizer
            epsilon: parameter for epsilon-greedy exploration
            Vmin, Vmax, and N: parameters for the C51 distribution (see original paper)
            loadWeights: Should the agent load previously saved weights?          
        """
        super(C51Agent, self).__init__(seed=seed,saveWeightsInterval = saveWeightsInterval)
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
        self.replay_memory = []
        self.batchController = batch_util.BatchController(self,self.batch_type)
        self.learningSteps = 0
        self.className = "C51"
        self.loadStep = loadStep
        
        
        
    def connect_env(self,environment,agentIndex,allAgents):
        """Connects to the domain environment
             environment: HFO environment object
             agentIndex: Agent index in the evnvironment
        """
        super(C51Agent, self).connect_env(environment,agentIndex,allAgents)
        self.environmentActions = environment.possible_actions()
        self.countReplayActions = np.zeros(len(self.environmentActions))
        #self.graph = tf.Graph()
        #self.session = keras.backend.get_session()
        #with self.session.as_default():
        #    with self.graph.as_default():
        self.build_network()

        self.update_target()

        if self.loadWeights:
            self.load_weights(self.loadStep)
            print("Weights Loaded " + str(self.loadStep))
        
        
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
        if self.useThreeNetworks:
            for act in range(n_act):
                #Hidden layers
                #First hidden Layer
                layerW = tf.get_variable(prefix+'W1/'+str(act), trainable = trainable,shape=(featureSize,self.n_neuronsHidden),
                                         initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                         #initializer=tf.random_uniform([featureSize,self.n_neuronsHidden],seed = self.rnd.randint(0,1000),minval = 0.0001, maxval=0.1))                
                layerB = tf.get_variable(prefix+'b1/'+str(act), trainable = trainable,shape=(self.n_neuronsHidden),
                                         initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                         #initializer=tf.random_uniform([self.n_neuronsHidden], seed = self.rnd.randint(0,1000)))
                hiddenL =  tf.add(tf.matmul(inputs,layerW),layerB)
                hiddenL = tf.nn.sigmoid(hiddenL)
                #hiddenL = tf.nn.dropout(hiddenL, tf.constant(0.9), seed = self.rnd.randint(0,1000))
                for i in range(1,self.n_hidden):
                    layerW = tf.get_variable(prefix+'W'+str(i+1)+'/'+str(act),trainable = trainable, shape=(self.n_neuronsHidden,self.n_neuronsHidden),
                                             initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                         #initializer=tf.random_uniform([self.n_neuronsHidden,self.n_neuronsHidden],seed = self.rnd.randint(0,1000),minval = 0.0001, maxval=0.1)))
                    layerB = tf.get_variable(prefix+'b'+str(i+1)+'/'+str(act), trainable = trainable, shape = (self.n_neuronsHidden),
                                             initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                        #initializer=tf.random_uniform([self.n_neuronsHidden], seed = self.rnd.randint(0,1000)))
                    hiddenL =  tf.add(tf.matmul(hiddenL,layerW),layerB)
                    hiddenL = tf.nn.relu(hiddenL) 
                #Last Layer, connection with the Actions
                layerW = tf.get_variable(prefix+'W'+str(self.n_hidden+1)+'/'+str(act), trainable = trainable,shape=(self.n_neuronsHidden,self.N),
                                         initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                         #initializer=tf.random_uniform([self.n_neuronsHidden,self.N],seed = self.rnd.randint(0,1000),minval = 0.0001, maxval=0.1))
                layerB = tf.get_variable(prefix+'b'+str(self.n_hidden+1)+'/'+str(act), trainable = trainable, shape=(self.N),
                                         initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                        #initializer=tf.random_uniform([self.N], seed = self.rnd.randint(0,1000)))
                befSoft[act] = tf.add(tf.matmul(hiddenL, layerW), layerB)
                #befSoft[act] += tf.convert_to_tensor(1e-15)
                y_net[act] = tf.nn.softmax(befSoft[act])
        else:
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
                
            for act in range(n_act):
                #Last Layer, connection with the Actions
                layerW = tf.get_variable(prefix+'W'+str(self.n_hidden+1)+'/'+str(act), trainable=trainable, shape= (self.n_neuronsHidden,self.N),
                                         initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                         #tf.random_uniform([self.n_neuronsHidden,self.N],seed = self.rnd.randint(0,1000),minval = 0.0001, maxval=0.1))
                layerB = tf.get_variable(prefix+'b'+str(self.n_hidden+1)+'/'+str(act), trainable=trainable, shape= (self.N), 
                                         initializer = tf.glorot_uniform_initializer(seed=self.rnd.randint(0,1000)))
                                         #tf.random_uniform([self.N], seed = self.rnd.randint(0,1000)))

                befSoft[act] = tf.add(tf.matmul(hiddenL, layerW), layerB)
                y_net[act] = tf.nn.softmax(befSoft[act])
            
        return inputs,y_net,befSoft
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
            #Placeholder for correct predictions (training)
            self.y = tf.placeholder(tf.float32, [None,self.N], name = "y")
            
            #Builds both trainable and target networks
            self.inputs, self.y_hat,befSoft = self.build_layers(trainable = True, prefix = "net/")
            self.inputs_target, self.y_hat_target,befSoft_target = self.build_layers(trainable = False, prefix = "target/")
            

            # builds the operation to update the target network
            self.update_target_op = []
            trainable_variables = tf.trainable_variables()
            all_variables = tf.global_variables()
            for i in range(0, len(trainable_variables)):
                #print(trainable_variables[i].name + "->" +all_variables[len(trainable_variables) + i].name)
                self.update_target_op.append(all_variables[len(trainable_variables) + i].assign(trainable_variables[i]))

               
            #A categorical cross-entropy cost function and an optimizer are defined for each action
            self.cost = [None] * n_act
            self.optimizers = [None] * n_act
            for i in range(n_act):
                #cost[i] = tf.reduce_mean( - tf.reduce_sum(self.y * tf.log(self.y_hat[i] +0.0000000001)))
                #cost[i] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=befSoft[i],labels=self.y))
                #cost[i] = tf.Print(cost[i], [cost[i]], "cost")
                self.cost[i] = self.categorical_crossentropy(target=self.y, output=self.y_hat[i])
                # add an optimizer
                self.optimizers[i] = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.cost[i])
                
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.session = tf.Session(graph=g, config=tf.ConfigProto(gpu_options=gpu_options))
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.update_target()
 
    def update_target(self):

        """Updates the target network with the current network weights"""
        g = self.session.graph

        with g.as_default():
            self.session.run(self.update_target_op)

    
    
    def calc_z_i(self,i):
        return self.Vmin + i*self.deltaZ

    
    def delete_example(self):
        return self.batchController.delete_example()
    def get_mini_batch(self):
        return self.batchController.get_mini_batch()
    
    def observe_reward(self,state,action,statePrime,reward):
        if self.exploring:
            self.learningSteps += 1
            self.batchController.add_sample([state,action,statePrime,reward,self.environment.is_terminal_state()])
            #actI = self.environmentActions.index(action)
            # self.countReplayActions[actI] += 1
            
            #with self.session.as_default():
            #    with self.graph.as_default():
            g = self.session.graph

            with g.as_default():
                if self.learningSteps % self.learningInterval == 0:
                    batch,self.lastIndexes = self.get_mini_batch()#self.rnd.sample(self.replay_memory, min(self.miniBatchSize,len(self.replay_memory)))
                    self.train_network(batch)
                if self.learningSteps % self.updateTargetInterval == 0:
                    self.update_target()
                
                

    def finish_learning(self):
        self.session.close()       
    def finish_episode(self):
        super(C51Agent, self).finish_episode()
        if self.training_episodes_total % self.saveWeightsInterval == 0:
            self.save_weights(self.training_episodes_total)
            
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
        
        next_acts = self.select_action(statesPrime, multipleOut=True, useNetwork=True) #use self.network
        next_acts = [self.environmentActions.index(a) for a in next_acts]
        
        
        statesPrime = np.array([statep[1] for statep in statesPrime])
        z_prime = []
        for targetNet in self.y_hat_target:
            z_prime.append(targetNet.eval(session=self.session,feed_dict={self.inputs_target:statesPrime}))  #Use target network
        
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
                    
        cost_vals = []
        for i in range(len(self.y_hat)):
            indexes = np.array(actions) == i
            statesNet = states[indexes]
            if len(statesNet) > 0:
                for y in range(self.learningEpochs):
                    self.optimizers[i].run(session=self.session, feed_dict = {
                                                                  self.inputs : statesNet,
                                                                  self.y : m[i][indexes][:]            
                                                                              })
                #cost_vals.append(self.session.run(self.cost[i], feed_dict={
                #                                              self.inputs : statesNet,
                #                                              self.y : m[i][indexes][:]            
                #                                                          }))
        #print("Costs: " + str(cost_vals) + " -  shape:" + str([cost.shape for cost in cost_vals]) + str(self.agentIndex))
                
                #self.network[i].fit(statesNet, m[i][indexes][:], batch_size=len(statesNet), verbose=0, epochs=2)
        
        #For batch update
        if self.batch_type == batch_util.PRIORITIZED:
            for i in range(len(actions)):
                actions[i] = self.environmentActions[actions[i]]
            importance = np.zeros((len(states)))
            for i in range(len(states)):
                q = self.calc_Q(states[i], actions[i], True)
                importance[i] = np.absolute(rewards[i] - q)
            
            self.batchController.batch_update([self.lastIndexes,importance])
                      
        
        
        
  

    def calc_Q(self,state,action,useNetwork=True):
        """Calculates the Q-value for a given state and action
            useNetwork: If true, uses the trainable network, otherwise uses the
                target network.
        """
        prob_vec = self.get_distrib(state,action,useNetwork)
        value = np.dot(prob_vec,self.z_vec)#np.dot(self.z_vec, prob_vec)
        return value[0]
    
    def get_distrib(self,state,action,useNetwork=True):
        """ Returns the distribution over possible returns for the current state and action"""
        
        if useNetwork:
            inputs,y_hat = self.inputs, self.y_hat
        else:
            inputs,y_hat = self.inputs_target, self.y_hat_target
            
        act_idx = self.environmentActions.index(action)
        
        g = self.session.graph
        
        with g.as_default():
            distrib = y_hat[act_idx].eval(session=self.session, feed_dict = {inputs : np.array([state])})
        
        return distrib #np.array([self.prob(i,state,action) for i in range(self.N)])

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
                    
                    return_act.append(self.rnd.choice(maxAct))
        if not multipleOut:
            return return_act[0]
        return return_act
    
    def get_used_budget(self):
        return 0.0
           
        
                   
    def load_weights(self,step): 
        """Loads previously saved weight files"""
        fileFolder = "./agentFiles/" + self.className + "/" + str(step) + "/"
        if self.environment.numberFriends == 0:
            filePath = fileFolder + "C51Model.ckpt"
        else:
            filePath = fileFolder + "C51Model" + str(self.agentIndex) + ".ckpt"
        self.saver.restore(self.session, filePath)  
        
    def save_weights(self,step):
        "Saves the weights when desired"
        fileFolder = "./agentFiles/" + self.className +"/" + str(step) + "/"
        try:
            os.makedirs(fileFolder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            
        if self.environment.numberFriends == 0:
            filePath = fileFolder + "C51Model.ckpt"
        else:
            filePath = fileFolder + "C51Model" + str(self.agentIndex) + ".ckpt"
        
        self.saver.save(self.session,filePath)
        
        
    def categorical_crossentropy(self,target, output, from_logits=False, axis=-1):
        """Categorical crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor of the same shape as `output`.
            output: A tensor resulting from a softmax
                (unless `from_logits` is True, in which
                case `output` is expected to be the logits).
            from_logits: Boolean, whether `output` is the
                result of a softmax, or is a tensor of logits.
            axis: Int specifying the channels axis. `axis=-1`
                corresponds to data format `channels_last`,
                and `axis=1` corresponds to data format
                `channels_first`.
        # Returns
            Output tensor.
        # Raises
            ValueError: if `axis` is neither -1 nor one of
                the axes of `output`.
        """
        output_dimensions = list(range(len(output.get_shape())))
        if axis != -1 and axis not in output_dimensions:
            raise ValueError(
                '{}{}{}'.format(
                    'Unexpected channels axis {}. '.format(axis),
                    'Expected to be -1 or one of the axes of `output`, ',
                    'which has {} dimensions.'.format(len(output.get_shape()))))
        # Note: tf.nn.softmax_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # scale preds so that the class probas of each sample sum to 1
            #output /= tf.reduce_sum(output, axis, True)
            # manual computation of crossentropy
            _epsilon = tf.convert_to_tensor(1e-15, output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
            return tf.reduce_mean(-tf.reduce_sum(target * tf.log(output), axis),axis)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                           logits=output)
   
