"""Used to generate graphs for specific states"""
from environment.hfoenvironment import HFOEnvironment
from agents.c51agent import C51Agent
from graphs.graphs import GraphBuilder
import matplotlib.pyplot as plt


statesEval = [  (True, (-0.6956999897956848, -0.45806145668029785, 0.0, -0.6834777593612671, -0.45806145668029785, 0.6346031427383423, 0.10877668857574463, -0.9040104150772095, 0.5318907499313354, 0.7561300992965698, -0.07641655206680298)),
                (True, (-0.12747615575790405, -0.2801550626754761, 0.19253051280975342, -0.0974000096321106, -0.2865908741950989, 0.02410578727722168, 0.10608696937561035, -0.8506987690925598, -0.0708267092704773, 0.7614221572875977, -0.07234388589859009)),
                (True, (0.5830222368240356, 0.042909979820251465, -0.05944555997848511, 0.6065587997436523, 0.04204404354095459, -0.7427107095718384, -0.06391400098800659, -0.5353899002075195, -0.8396923542022705, 0.7421746253967285, 0.04398000240325928)),
              ]


number_agents = 1
port = 12345
environment = HFOEnvironment(numberLearning=number_agents,cooperative=0,
                    numberOpponents=1,port=port,limitFrames = 200)

agent = [None] * number_agents
for i in range(number_agents):
    agent[i] = C51Agent(environment,loadWeights = True,loadStep=".")
    agent[i].connect_env(environment,i,agent)
    agent[i].exploring = False

graphB = GraphBuilder(agent[0],environment)


rept_state = 20
index=1

for state in statesEval:   
    print(state)
    print("Q 11")
    print(agent[0].calc_Q(state[1], 11,False))
    print("Q 9")
    print(agent[0].calc_Q(state[1], 9,False))
    
    for i in range(rept_state):
         graphB.update_graph(state,index)

graphB.finish()

#plt.show()
    
print("Done")
    