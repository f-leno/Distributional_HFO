import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as manimation

class GraphBuilder():
    agent = None
    environment = None
    movieWriter = None
    
    graphData = None
    
    colors = [(0.2588,0.4433,1.0),
              (1.0,0.5,0.62),
              (0.,0.,0.)
              ]
    
    window = None
    
    def __init__(self,agent,environment):
        self.agent = agent
        self.environment = environment
        graph, ax = plt.subplots()
        self.window = ax
        
        
        
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
        self.movieWriter = FFMpegWriter(fps=15, metadata=metadata)
        self.movieWriter.setup(graph,'./outputVideo.mp4',100)
        
    
    def update_graph(self,state,step=None):
        N = self.agent.N
        actions = self.environment.all_actions(state,0)
        
        #graph, ax = plt.subplots()
        ax = self.window
        width = 0.35
        
        index = np.arange(N)
        
        sum_distrib = np.zeros((1,N))
        acc_probs = np.zeros((len(actions),N))
        for i in range(len(actions)):
            distrib = self.agent.get_distrib(state[1],actions[i])
            sum_distrib += distrib
            acc_probs[i,:] = np.copy(sum_distrib)[:]
        
        if self.graphData is not None:
             for bars in self.graphData:
                 bars.remove()
        
        self.graphData = [None]*len(actions)
        index = self.agent.z_vec
        for i in reversed(range(len(actions))):
            self.graphData[i] = ax.bar(index,acc_probs[i],width,color=self.colors[i],label=str(actions[i]))
        ax.set_ylabel('Prob')
        ax.set_xlabel('V')
        #ax.set_xticklabels([str(int(x*self.agent.deltaZ)) for x in index])
        #ax.set_xticks([str(int(x)) for x in self.agent.z_vec])
        ax.legend()
        
        
        if step is not None:
            ax.set_title(str(step))
        #ax.set_xticks(index)
        self.movieWriter.grab_frame()
        
    def finish(self):
        self.movieWriter.finish()    
        #graph.xticks(np.arange(min(self.agent.z_vec), max(self.agent.z_vec)+1, 5.0))
        
       
            
        #plt.show()
        
        
        
        
        
        