import os
import numpy as np
import pandas as pd

class Memory(object):
    #def __init__(self, params, state_shape, action_shape):
    def __init__(self, params):
        self.params = params
        self.max_size = self.params["memory_size"]
        shape = [self.max_size]
        self.current_position = 0
        self.current_size = 0
        
    def add(self):
        """
            Adds a new experience to the replay memory.
        """
        # make new entry
        #print(state, action, reward, state_)
        # update memory size and current position
        self.current_size = max(self.current_size, self.current_position + 1)
        self.current_position = (self.current_position + 1) % self.max_size
    
    def get_minibatch_indices(self, batchsize):
        """
            Just a dummy. Should be implemented for each kind of replay memory separately.
        """
        return np.asarray([])
    
    def extract_batch_from_memory(self, indices):
        """
            Returns states, actions, rewards, and followup states from a given
            range of indices from the memory.
        """
        #return np.take(self.states, indices), \
        #    np.take(self.actions, indices), \
        #    np.take(self.rewards, indices), \
        #    np.take(self.states_, indices), \
        #    np.take(self.terminals, indices)
        return self.states[indices], \
            self.actions[indices], \
            self.rewards[indices], \
            self.states_[indices], \
            self.terminals[indices], \
            self.targets[indices]
    
    def get_minibatch (self, batchsize):
        """
            Returns a whole set of samples from the memory as np arrays for
            states, actions, rewards, and followup states.
        """
        #print("################ get_minibatch")
        return self.extract_batch_from_memory(self.get_minibatch_indices(batchsize))

        
class ReplayMemory(Memory):
    # def __init__(self, params, state_shape, action_shape):
    #     super(ReplayMemory, self).__init__(params, state_shape, action_shape)
    def __init__(self, params):
        super(ReplayMemory, self).__init__(params)
    
    def get_minibatch_indices(self, batchsize):
        """
            Returns the indices for a given batch size.
        """
        indices = []
        while len(indices) < batchsize:
            while True:
                # sample one index (ignore states wrapping over)
                index = self.params["rng"].randint(self.current_size - 1)
                # if wraps over current pointer, then get new one
                if index >= self.current_size and index - 1 < self.current_position:
                    continue
                # if wraps over episode end, then get new one
                # last state could be terminal!
                if self.terminals[index - 1]:
                    continue
                # otherwise use this index
                break
            # NB! having index first is fastest in C-order matrices
            # states[len(indices), ...] = self.states[index - 1]
            # states_[len(indices), ...] = self.states[index]
            indices.append(index)
        return indices
    '''
    def get_minibatch (self, batchsize):
        # indices = self.get_minibatch_indices(batchsize)
        # copy actions, rewards and terminals with direct slicing
        # actions = self.actions[indices]
        # rewards = self.rewards[indices]
        # terminals = self.terminals[indices]
        # return states, actions, rewards, states_
        return extract_batch_from_memory(self.get_minibatch_indices(batchsize))
    '''
  
class PrioritizedReplayMemory(Memory):
    # def __init__(self, params, state_shape, action_shape):
    #     super(ProritizedReplayMemory, self).__init__(params, state_shape, action_shape)
    def __init__(self, params, env):
        super(PrioritizedReplayMemory, self).__init__(params, env)
        # Priorities are saved in a SumTree structure
        #      0        Parentnode of 1 and 2          --\
        #     / \                                         > max_size - 1 = 3
        #    1   2      Parentnodes of 3&4 and 5&6     --/
        #   /\  /\
        #  3 4 5 6      Actual Priorities (leafes)      --> max_size = 4
        self.prio_size = 2*self.max_size - 1
        self.priorities = np.zeros(self.prio_size, dtype = np.float32)
        # [----parent nodes (max_size-1)----][----leafes (max_size)----]
        
    def add(self):
        index = self.current_position + self.max_size - 1
        # initially set to highest priority value
        update_value = max(self.priorities[-self.max_size:])
        if update_value == 0.0:
            update_value = self.params["mem_max_priority"]
        self.update_priorities([index], [update_value])
        #self.update_priorities([index], [priority]) 
        super(PrioritizedReplayMemory, self).add()
        
    def update_priorities(self, indices, priorities):
        i = 0
        for index in indices:
            #print("Update", index, priorities[i])
            # calculate change for each index
            change = priorities[i] - self.priorities[index]
            # set new priority
            self.priorities[index] = priorities[i]
            # propagate change to all parents recursively
            while index != 0:   
                index = (index - 1) // 2
                self.priorities[index] += change
            i += 1
    
    def get_sample(self, priority):
        prio = priority
        # explore the tree from top to botton
        parent = 0
        while (True):
            left_child = 2 * parent + 1
            right_child = left_child + 1
            #print("###############################", priority, parent, left_child, right_child)
            #print("###############################", self.priorities.shape[0], 
            # self.priorities[left_child], self.priorities[right_child])
            # if bottom of tree has been reached
            if left_child >= self.prio_size:
                sample = parent
                break
            # otherwise keep exploring downwards
            else:
                if priority <= self.priorities[left_child]:
                    parent = left_child
                else:
                    priority -= self.priorities[left_child]
                    parent = right_child
        #print("Sample", sample, prio, priority)
        return sample
    
    def get_minibatch_indices(self, batchsize):
        """
            Returns the indices for a given batch size.
        """
        indices = []
        segment = self.priorities[0] / batchsize
        for i in range(batchsize):
            range_left = segment * i
            range_right = segment * (i + 1)
            target_priority = self.params["rng"].uniform(range_left, range_right)
            #print("Segment", segment, "Range", range_left, range_right, "Target", target_priority)
            indices.append(self.get_sample(target_priority))
        #print("Indices", np.clip(np.asarray(indices) - self.max_size + 1, 0, self.current_size-1), self.current_size)
        return np.clip(np.asarray(indices) - self.max_size + 1, 0, self.current_size-1)


    def batch_update(self, indices, new_values):
        self.update_priorities(indices + self.max_size - 1,
                               np.squeeze(np.clip(new_values+self.params["prioritize_bias"], 0., 1.)))

    