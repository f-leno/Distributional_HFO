from environment.hfoenvironment import HFOEnvironment
from agents.c51agent import C51Agent
from agents.dqnagent import DQNAgent
from agents.c51syncagent import C51SyncAgent
from agents.c51thresholdpolicy import C51ThresholdPolicy
from graphs.graphs import GraphBuilder
import matplotlib.pyplot as plt
from threading import Thread,Condition
from time import sleep

loadStep = 10000
if True:  #C51Threshold
    agentClass = C51ThresholdPolicy
elif False: #C51Average
    agentClass = C51Agent
elif False: #DQN
    agentClass = DQNAgent
    
number_agents = 2
interruptEnd = True
port = 12345

def thread_agent(agentClass,agentIndex,port,number_agents):
    environment =HFOEnvironment(numberLearning=number_agents,cooperative=0,
                    numberOpponents=2,port=port,limitFrames = 200)

    agent = agentClass(loadWeights = True,loadStep = loadStep) #Initiating agent   
    agent.connect_env(environment,agentIndex, [agent]*number_agents)
   
    agent.exploring = False
    recordVideo = agentIndex == 0 and False     
       
    if recordVideo:     
        graphB = GraphBuilder(agent,environment)


    episodes = 20
    intervalVideo = 5
    
    step = 0


    for epi in range(episodes):
        environment.start_episode()   
        while not environment.is_terminal_state():
            #Reverse range to finish with agent 0
            state = environment.get_state(agentIndex)
            act = agent.select_action(state)
            environment.act(act,agentIndex)
            statePrime,action,reward = environment.step(agentIndex)
            step+=1
            #if step % intervalVideo == 0 and len(environment.all_actions(statePrime,0)) > 1:
            #if len(environment.all_actions(statePrime,0)) > 1:
            if recordVideo:
                graphB.update_graph(state,step,act)
                print(state)
                print("Q 11")
                print(agent.calc_Q(state[1], 11,False))
                print("Q 9")
                print(agent.calc_Q(state[1], 9,False))
    
    if recordVideo:
        graphB.finish()
        

        




agentThreads = []
#Initiating agent
for i in range(number_agents):
    agentThreads.append(Thread(target = thread_agent, args=(agentClass,i,port,number_agents)))
    agentThreads[i].start()
    sleep(2)
            
            
#Waiting for program termination
for i in range(number_agents):
    agentThreads[i].join()







#plt.show()
    
print("Done")
    