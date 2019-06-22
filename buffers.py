from collections import deque
import random
import torch
import numpy as np
from collections import namedtuple, deque

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size


# https://www.freecodecamp.org/news/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682/
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    

class PER_ReplayBuffer:
    priority_epsilon = 0.001 # ensure that all experience with positive probability
    max_error = 1
    def __init__(self, device, buffer_size, batch_size, alpha, beta):
     # alpha is the power of the probability to address over selection problem
     # beta is the power used when update the weights.
        self.buffer = SumTree(buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.alpha = alpha
        self.beta = beta
        self.device = device

    def get_priorities(self):
        # all the list of priorities for all data/experience in the buffer
        return self.buffer.tree[self.buffer.branches:]

    def add(self, state, action, reward, next_state):
        """
        Store an experience in the SumTree, new experience stored with the maximum priority until they have been replayed at least once so all experience has the chance to be used for training. (This is not in the lecture?)
        """
        max_priority = np.max(self.get_priorities())
        e = self.experience(state, action, reward, next_state) 
        if max_priority == 0: # when there is no data in the Sumtree, all priorities are 0. 
            max_priority = 1
        self.buffer.add(max_priority, e)

    def sample(self, n):
        # to do
        """
        Sample with size n (batch size for example) using prioritized experience replay
        """
        batch = []
        idxs = []
        priorities = []
        segment_size = self.tree.total_priority / n

        for i in range(n):
            low = segment_size * i
            high = segment_size * (i + 1)

            value = np.random.uniform(low, high) #sample a value from the segment

            index, priority, data = self.tree.get(value) #retrieve corresponding experience
            idxs.append(index)
            priorities.append(priority)
            batch.append(data)
            # p = priority / self.tree.total_priority
            # is_weights[i, 0] = np.power(n * p, -self.beta) / max_weight
            # indexes[i]= index
        
        #setup weights
        probabilities = priorities / self.tree.total_priority

        # weights are more important in the end of learning when our q values begin to converge, so increase beta.
        self.beta = np.min([self.beta + self.beta_incremement, 1.0])  # anneal Beta to 1

        #Dividing by the max of the batch is a bit different than the original
        #paper, but should accomplish the same goal of always scaling downwards
        #but should be slightly faster than calculating on the entire tree
        is_weights = np.power(self.tree.num_entries * probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(is_weights).type(torch.FloatTensor).to(self.device)
        return self.experience(*zip(*batch)), is_weights, idxs

    def batch_update(self, tree_idx, td_errors):
        """
        Update the priorities on the tree
        """
        #error should be provided to this function as ABS()
        td_errors += self.priority_epsilon  # Add small epsilon to error to avoid ~0 probability
        clipped_errors = np.minimum(td_errors, self.max_error) # No error should be weight more than a pre-set maximum value
        priorities = np.power(clipped_errors, self.alpha) # Raise the TD-error to power of Alpha to tune between fully weighted or fully random

        for idx, priority in zip(tree_idx, priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.tree_size


          




    
#https://github.com/jaromiru/AI-blog/blob/master/SumTree.py

class SumTree(object):
    """
    SumTree for holding Prioritized Replay information in an efficiently to
    sample data structure. Uses While-loops throughought instead of commonly
    used recursive function calls, for small efficiency gain.
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of memories to store
        self.branches = capacity - 1 # Branches above the leafs that store sums
        self.tree_size = self.capacity + self.branches # Total tree size
        self.tree = np.zeros(self.tree_size) # Create SumTree array
        self.data = np.zeros(capacity, dtype=object)  # Create array to hold memories corresponding to SumTree leaves
        self.data_pointer = 0
        self.num_entries = 0

    def add(self, priority, data):
        """
        Add the memory in DATA
        Add priority score in the TREE leaf
        """
        idx = self.data_pointer % self.capacity
        tree_index = self.branches + idx # Update the leaf

        self.data[idx] = data # Update data frame
        self.update(tree_index, priority) # Indexes are added sequentially
        # Incremement
        self.data_pointer += 1
        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, tree_index, new_priority):
        """
        Update the leaf priority score and propagate the change through tree
        """
        # Change = new priority score - former priority score
        change = new_priority - self.tree[tree_index]
        self.tree[tree_index] = new_priority
        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, v):
        """
        Retrieve the index, values at that index, and replay memory, that is
        most closely associated to sample value of V.
        """
        current_idx = 0
        while True:
            left = 2 * current_idx + 1
            right = left + 1

            # If we reach bottom, end the search
            if left >= self.tree_size:
                leaf_index = current_idx
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left]:
                    current_idx = left

                else:
                    v -= self.tree[left]
                    current_idx = right

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node   