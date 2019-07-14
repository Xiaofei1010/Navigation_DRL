import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class WeightedLoss(nn.Module):
    """
    Returns Huber Loss with importance sampled weighting.
    """

    def __init__(self):
        super(WeightedLoss, self).__init__()

    def huber(self, values, targets, weights):
        errors = torch.abs(values - targets)
        loss = (errors<1).float()*(errors**2) + (errors>=1).float()*(errors - 0.5)
        weighted_loss = (weights * loss).sum()
        return weighted_loss, errors.detach().cpu().numpy()