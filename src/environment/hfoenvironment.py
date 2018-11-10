"""
Environment Class for preparing HFO experiments
@author: Felipe Leno
"""
import subprocess
import environment.hfoactions as hfoactions

from environment.hfostate import HFOStateManager

import hfo

import time,sys,math
from threading import Thread

class HFOEnvironment(object):
    serverPort = None
    hfoObj = None
    #Path for the bin folder
    serverPath = "../HFO/bin/"
    #Server subprocess to be finished later
    serverProcess = None
    clientProcess = None

    #Status of the last step
    lastStatus = None
    #Total number of episodes
    totalEpisodes = None
    #Total number of goals
    goals = None
    #Number of friendly agents
    numberFriends = None
    #Number of Opponents
    numberOpponents = None
    #Utilities for state space variables
    stateSpaceManager = None
    #Last applied actions
    lastActions = None
    #Terminate Server thread?
    terminateThread = None
    #Action and parameter  to be sent to HFO server
    applyAction = None
    actionParameter = None
    #Variable to request step processing to another thread
    stepRequest = None
    #Variable to control when to erase other threads
    clearServer = None
    #Number of learning agents 
    numberLearning = None


    
    
    
    def __init__(self,numberLearning=2,cooperative=0,numberOpponents=2,port=1234,limitFrames = 200):
        """Initiates the HFO environment
            numberLearning = Number of Learning Agents
            cooperative = Number of non-learning cooperative agents
            numberOpponents = Number of agents in the other side of the field
            limitFrames = limit in the number of frames before the game is interrupted        
        """
        
        self.serverPort = port
        self.lastStatus = hfo.IN_GAME
        
        self.numberLearning = numberLearning
        self.numberFriends = numberLearning-1 + cooperative
        self.numberOpponents =  numberOpponents

        self.applyAction = [None]*numberLearning
        self.actionParameter = [None]*numberLearning

        #self.agentsControl = agentsControl
        self.lastActions = [None]*numberLearning
        self.hfoObj = [None]*numberLearning
        #for i in range(numberLearning):
        #    self.hfoObj.append(hfo.HFOEnvironment())
        #self.hfoObj = hfo.HFOEnvironment()

        
        self.stepRequest = [None]*numberLearning
        self.clearServer = [None]*numberLearning
        
        
        
        #Initiates a new thread only to avoid an error when loading the strategy.cpp file



        #Initiates one thread for each agent controlled by learning algorithms
        #for i in range(self.numberLearning):
        #    t = Thread(target=connect_server, args=(self, i))
            #t.start()
            #time.sleep(2)
        #t = Thread(target=connect_server, args=(self,))
        #t.start()
        #The connection with the server is OK after here.
        #time.sleep(3)
        self.totalEpisodes = 0
        self.goals = 0
        
        self.stateSpaceManager = HFOStateManager(numberLearning-1+cooperative,self.numberOpponents)
 
        
    def clean_connections(self,serverProcess):
        """Cleans all the initiated services and files"""
        #self.clearServer = [True]*self.numberLearning
        #Wait until another thread finishes the HFO client
        #while True in self.clearServer:
        #    pass
        
        #Kill the HFO server
        #subprocess.call("kill -9 -"+str(self.serverProcess.pid), shell=True)
        #for proc in self.clientProcess:
        #    subprocess.call("kill -9 -" + str(proc.pid), shell=True)
        subprocess.call("kill -9 -"+str(serverProcess.pid), shell=True)

        time.sleep(2)
        #portmanager.release_port(self.serverPort)
        
    def finish_learning(self):
        pass
        #self.clean_connections(serverProcess)
        

    #def get_state(self,agentIndex=0):
    #    return self.hfoObj[agentIndex].getState()
    def all_actions(self,state,agentIndex,forExploration=False):
        """Returns the set of applicable actions for the agent
           in case the agent has the ball, a PASS for each friend, DRIBBLE and SHOOT
           are applicable. Otherwise, only MOVE is applicable
        """
        #fullState = self.hfoObj[agentIndex].getState()
        #withBall = fullState[self.stateSpaceManager.ABLE_KICK] == 1.0
        withBall = state[0]
                
        return hfoactions.all_actions(self.numberFriends, withBall, forExploration) 

    def possible_actions(self):
        """ Return all possible actions, regardless of if they are applicable in the current state """
        return hfoactions.possible_actions(self.numberFriends)
        
    def act(self,action,agentIndex=0):
        """Performs the agent action"""
        #Transforms the action in the agent's point of view to the correct HFO server format
        self.lastActions[agentIndex] = action
        self.applyAction[agentIndex], self.actionParameter[agentIndex] = self.translate_action(action, self.hfoObj[agentIndex].getState())
        #self.applyAction, self.actionParameter = self.translate_action(action, self.hfoObj.getState())
        #Wait for another thread
        #while not self.applyAction[agentIndex] is None:
        #    pass

        
    def translate_action(self, action, stateFeatures):
        """Translates the action to one that is understandable in the HFO server"""
        #If the agent chooses a pass action, a translation is needed
        if hfoactions.is_pass_action(action):
            #According to the chosen action, defines which of the agents is the destination of the pass
            # index=0 corresponds to the closest agent, while index=1 to the second closest, and etc.
            indexAction = hfoactions.pass_index(action)
            f = self.stateSpaceManager
            #Sort Friends by euclidian distance
            listProx = [math.hypot(stateFeatures[f.FRIEND1_X]- stateFeatures[f.X_POSITION], 
                                   stateFeatures[f.FRIEND1_Y]- stateFeatures[f.Y_POSITION])]
            listIDs =  [stateFeatures[f.FRIEND1_NUMBER]]
            
            if self.numberFriends > 1:
                listProx.append(math.hypot(stateFeatures[f.FRIEND2_X]- stateFeatures[f.X_POSITION], 
                                   stateFeatures[f.FRIEND2_Y]- stateFeatures[f.Y_POSITION]))
                listIDs.append(stateFeatures[f.FRIEND2_NUMBER])
                
                if self.numberFriends > 2:
                    listProx.append(math.hypot(stateFeatures[f.FRIEND3_X]- stateFeatures[f.X_POSITION], 
                                   stateFeatures[f.FRIEND3_Y]- stateFeatures[f.Y_POSITION]))
                    listIDs.append(stateFeatures[f.FRIEND3_NUMBER])
                    
                    if self.numberFriends > 3:
                        listProx.append(math.hypot(stateFeatures[f.FRIEND4_X]- stateFeatures[f.X_POSITION], 
                                   stateFeatures[f.FRIEND4_Y]- stateFeatures[f.Y_POSITION]))
                        listIDs.append(stateFeatures[f.FRIEND4_NUMBER])
            #Get list of friends' indexes in descending order according to proximity
            idsOrder = sorted(range(len(listProx)), key=lambda k: listProx[k])
            #To whom the agent should pass
            indexFriend = idsOrder[indexAction]
            #Id according to HFO internal code
            friendUNum = listIDs[indexFriend]
            actionRet = hfo.PASS
            argument = friendUNum

            #print("####### PASS: " + str(friendUNum))
        else:
            actionRet = action
            argument = None
        if hfo.PASS == actionRet and argument is None or argument == 0:
            print(action)
            print(stateFeatures)
            print(self.numberFriends)
        #print("############# Action Ret: " + str(actionRet) + " , arg: " + str(argument))
        return actionRet, argument  
      
    def step(self,agentIndex=0):
        """Performs the state transition and returns (statePrime.action,reward)"""   
        #for i in range(len(self.stepRequest)):
        #    self.stepRequest[i] = True
        #self.stepRequest[agentIndex] = True
        #Wait until another thread completes the step
        #while self.stepRequest[agentIndex]:
        #while self.stepRequest:
        #    pass
        #statePrime = []
        #action = []
        #for agentIndex in range(self.numberLearning):#range(self.agentsControl):
        #    statePrime.append(self.get_state(agentIndex))
        #    action.append(self.lastAction[agentIndex])
        #statePrime = self.get_state()
        #action = self.lastAction
        #reward = [self.observe_reward()]*self.numberLearning
        #return (statePrime,action,reward)
        #if agentIndex == 0:
        #    print("**Point 1")
        if self.actionParameter[agentIndex] is None:
            self.hfoObj[agentIndex].act(self.applyAction[agentIndex])
                #self.hfoObj.act(self.applyAction)
        else:
            self.hfoObj[agentIndex].act(self.applyAction[agentIndex], self.actionParameter[agentIndex])
                #self.hfoObj.act(self.applyAction, self.actionParameter)
        
        #if agentIndex == 0:
        #    print("**Point 2")
        #self.stepRequest[agentIndex] = True
        #Wait until another thread completes the step
        #while self.stepRequest[agentIndex]:
        #    time.sleep(0.0001)
            
        self.lastStatus = self.hfoObj[agentIndex].step()
        
        #if agentIndex == 0:
        #    print("**Point 3")
        
        statePrime = self.get_state(agentIndex)
        action = self.lastActions[agentIndex]
        reward = self.observe_reward()
        return (statePrime,action,reward)
        
    def check_terminal(self):
        """Checks if the current state is terminal and processes the reward"""
        #Here, there is no need to check the environment status. Then, we use the method
        #to count the number of goals
        if self.lastStatus != hfo.IN_GAME:
            self.totalEpisodes += 1
            if self.lastStatus == hfo.GOAL:
                self.goals += 1
        
    def get_state(self, agentIndex=0):
        """Returns the state in the point of view of the agent. 
        The state features are filtered from the full set of features in the HFO server.
        """
        fullState = self.hfoObj[agentIndex].getState()
        withBall = fullState[self.stateSpaceManager.ABLE_KICK] == 1.0
        filtered = self.filter_features(fullState)
        compositeState = (withBall,filtered)
        return compositeState

 
        #return self.filter_features(self.hfoObj.getState())
    
    def filter_features(self,stateFeatures):
        """Removes the irrelevant features from the HFO standard feature set"""   
        stateFeatures = self.stateSpaceManager.reorderFeatures(stateFeatures)
        return self.stateSpaceManager.filter_features(stateFeatures)


    def observe_reward(self):
        """Returns the reward for the agent"""
        if(self.lastStatus == hfo.IN_GAME):
            return 0.0
        elif(self.lastStatus == hfo.CAPTURED_BY_DEFENSE):
             return -1.0
        elif(self.lastStatus == hfo.OUT_OF_BOUNDS):
             return -1.0
        elif(self.lastStatus == hfo.OUT_OF_TIME):
             return 0.0
        elif(self.lastStatus == hfo.GOAL):
             return 1.0
        else:
            print("%%%%% Strange HFO STATUS: "+hfo.statusToString(self.lastStatus))
        
        return 0.0
    
    def is_terminal_state(self):
        """Returns if the current state is terminal"""
        return not self.lastStatus == hfo.IN_GAME
    
    def start_episode(self):
        """Start next evaluation episode"""
        self.lastStatus = hfo.IN_GAME
        self.applyAction = [None]*self.numberLearning
        self.actionParameter = [None]*self.numberLearning
        
    def load_episode(self,episodeInfo):
        """For this domain the server performs the reset
        """
        pass
    def state_transition(self):
        """Executes the state transition""" 
        pass
    
    def get_unum(self,agentIndex=0):
        return self.hfoObj[agentIndex].getUnum()
    
    def connect_server(self,agentIndex):
        """Connects the client subprocess in the hfo server
            The learning process should be all executed in here because of strange
            errors in the HFO server when executing more than one client at the same time
        """
        #Path with formations file
        connectPath = self.serverPath+'teams/base/config/formations-dt'
        self.hfoObj[agentIndex] = hfo.HFOEnvironment()
        #Connecting in the server
        serverResponse = self.hfoObj[agentIndex].connectToServer(
                feature_set= hfo.HIGH_LEVEL_FEATURE_SET,
                config_dir=connectPath,
                server_port=self.serverPort,
                server_addr='localhost',
                team_name='base_left',
                play_goalie=False)
        print("%%%% Server connection FeedBack:    " + str(serverResponse))
        
        
    
"""class ClientConnection():
    hfoObj = None
    main = None
    def __init__(self,main):
        self.main = main
        self.hfoObj = self.main.hfoObj
        
    def clear(self):
        self.hfoObj.act(hfo.QUIT)
    def action(self,action,parameter):
        if parameter is None:
                self.hfoObj.act(action)
        else:
                self.hfoObj.act(action, parameter)
    def step(self):
        self.main.lastStatus = self.hfoObj.step()"""
    
def connect_server(self,agentIndex):
        """Connects the client subprocess in the hfo server
            The learning process should be all executed in here because of strange
            errors in the HFO server when executing more than one client at the same time
        """
        #Path with formations file
        connectPath = self.serverPath+'teams/base/config/formations-dt'
        
        #Connecting in the server
        serverResponse = self.hfoObj[agentIndex].connectToServer(
                feature_set= hfo.HIGH_LEVEL_FEATURE_SET,
                config_dir=connectPath,
                server_port=self.serverPort,
                server_addr='localhost',
                team_name='base_left',
                play_goalie=False)
        print("%%%% Server connection FeedBack:    " + str(serverResponse))
        while not self.clearServer[agentIndex]:
            if self.stepRequest[agentIndex]:
                self.lastStatus = self.hfoObj[agentIndex].step()
                self.stepRequest[agentIndex] = False
            if(self.lastStatus == hfo.SERVER_DOWN):
                self.hfoObj[agentIndex].act(hfo.QUIT)
                print("%%%%%%% HFO Server Down, Ending Environment")
                sys.exit(0)  
            else:
                time.sleep(0.0001)
        self.hfoObj[agentIndex].act(hfo.QUIT)
        self.clearServer[agentIndex] = False
                
       
        """while not self.clearServer[agentIndex]:
            #Wait until one action is chosen
            while self.applyAction[agentIndex] is None and not self.clearServer[agentIndex]:
            #while self.applyAction is None and not self.clearServer:
                #print("Waiting action")
                time.sleep(0.0001)
            #Verifies if the agent should stop learning
            if True in self.clearServer:
                continue
                    
            print("Act -- agent:" +str(agentIndex) + "  action:" + str(self.applyAction[agentIndex]))   
            #Send action to HFO server.
            if self.actionParameter[agentIndex] is None:
                self.hfoObj[agentIndex].act(self.applyAction[agentIndex])
                #self.hfoObj.act(self.applyAction)
            else:
                self.hfoObj[agentIndex].act(self.applyAction[agentIndex], self.actionParameter[agentIndex])
                #self.hfoObj.act(self.applyAction, self.actionParameter)
             
            self.applyAction[agentIndex] = None
            #self.applyAction = None
            self.actionParameter[agentIndex] = None
            #self.actionParameter = None
            #Perform HFO step
            while not self.stepRequest[agentIndex] and not self.clearServer[agentIndex]:
            #while not self.stepRequest and not self.clearServer:
                time.sleep(0.0001)
            #Should the agent stop learning?
            if True in self.clearServer:
                continue
                
            self.lastStatus = self.hfoObj[agentIndex].step()
            #self.lastStatus = self.hfoObj.step()#[agentIndex].step()
            if(self.lastStatus == hfo.SERVER_DOWN):
                self.hfoObj[agentIndex].act(hfo.QUIT)
                print("%%%%%%% HFO Server Down, Ending Environment")
                sys.exit(0)  
            self.stepRequest[agentIndex] = False
        #When the clearServer is set as true, it is time to close the connection
        self.hfoObj[agentIndex].act(hfo.QUIT)
        self.clearServer[agentIndex] = False"""
                
            
            
        
def init_server(serverPath,serverPort,numberLearning,cooperative,numberOpponents,limitFrames,serverPid,seed=None,ballInOffense=True):
        """Initiates the server process.             
        """
        
       
        #Build all commands correspondent to parameters
        #agentsParam = " --offense-agents 1 --offense-npcs "+str(numberFriends)
        agentsParam = " --offense-agents "+str(numberLearning)+" --offense-npcs " + str(cooperative)
        opponentsParam = " --defense-npcs "+str(numberOpponents)
        #opStrategy = " --offense-team base --defense-team " + opStrategy
        #initDist = " --ball-x-min "+str(xMin) + " --ball-x-max "+str(xMax)
        if seed is None:
            seedParam = ""
        else:
            seedParam = " --seed "+str(seed)
        framesParam = " --frames-per-trial "+ str(limitFrames)
        
        
            
              
        

        #Including the name of the executable, default parameters, and the port in the command
        #serverCommand = self.serverPath + "HFO --fullstate --offense-on-ball 12" \
        serverCommand = serverPath + "HFO --fullstate" \
                                          " --no-logging --headless " + \
             "--port " +str(serverPort)
        if ballInOffense:
            #Ball starts with a random attacking agent
            serverCommand += " --offense-on-ball 12"
                         
        #Joining all the commands
        serverCommand += framesParam + agentsParam + opponentsParam + seedParam + " --verbose >> testlog.log"
        print(serverCommand)
        
        #Starting the server
        serverPid[0] = subprocess.Popen(serverCommand, shell=True)
        
        
        

        #self.clientProcess = []
        #for i in range(numberNpcs):
            #After starting server, starts friends subprocess
        #    friendCommand = "python domain/mock_agent.py -p " + str(self.serverPort) + " -o " + str(numberOpponents) + " -f " +str(numberFriends)
        #    print(friendCommand)
        #    self.clientProcess.append(subprocess.Popen(friendCommand, shell=True))
        #    time.sleep(1)
        
        
        
def clean_connections(serverProcess):
        """Cleans all the initiated services and files"""
        #self.clearServer = [True]*self.numberLearning
        #Wait until another thread finishes the HFO client
        #while True in self.clearServer:
        #    pass
        
        #Kill the HFO server
        #subprocess.call("kill -9 -"+str(self.serverProcess.pid), shell=True)
        #for proc in self.clientProcess:
        #    subprocess.call("kill -9 -" + str(proc.pid), shell=True)
        subprocess.call("kill -9 -"+str(serverProcess.pid), shell=True)

        time.sleep(2)
        #portmanager.release_port(self.serverPort)