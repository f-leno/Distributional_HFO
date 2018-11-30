import matplotlib.pyplot as plt
import numpy as np


data = np.array([ [0., 2. , 0. , 4. ], [0., 4., 0., 2.] ])
                 
colors = [(0.2588,0.4433,1.0),(1.0,0.5,0.62)]       

graph, ax = plt.subplots()

for i in range(len(data)):
    ax.bar(np.arange(data.shape[1]),data[i,:],0.35,color=colors[i],label=str(i))
    
    
plt.show()