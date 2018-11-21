from environment.hfoenvironment import HFOEnvironment
from agents.c51agent import C51Agent
from graphs.graphs import GraphBuilder
import matplotlib.pyplot as plt



number_agents = 1
port = 12345
environment = HFOEnvironment(numberLearning=number_agents,cooperative=0,
                    numberOpponents=1,port=port,limitFrames = 200)
agent = C51Agent(environment,loadWeights = True)
agent.connect_env(environment,0)
graphB = GraphBuilder(agent,environment)
agent.exploring = False

episodes = 1
intervalVideo = 5

step = 0


for epi in range(episodes):
    environment.start_episode()   
    while not environment.is_terminal_state():
        state = environment.get_state()
        act = agent.select_action(state)
        environment.act(act)
        statePrime,action,reward = environment.step()
        step+=1
        #if step % intervalVideo == 0 and len(environment.all_actions(statePrime,0)) > 1:
        #if len(environment.all_actions(statePrime,0)) > 1:
        graphB.update_graph(state,step,act)
        print(state)
        print("Q 11")
        print(agent.calc_Q(state[1], 11,False))
        print("Q 9")
        print(agent.calc_Q(state[1], 9,False))
            
    #graphB.build_graph(env.S1)

#graphB.build_graph(env.S2,'S2')
#graphB.build_graph(env.S3,'S3')
#graphB.build_graph(env.S4,'S4')
graphB.finish()

#plt.show()
    
print("Done")
    