
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DEVICE, DEFAULT_CONFIG, Actions
from ..utils import vectors


class ActionChooserNetwork(nn.Module):

    def __init__(self, input_dim):
        super(ActionChooserNetwork, self).__init__()

        self.hidden_dim = input_dim
        self.linear1 = nn.Linear(input_dim, self.hidden_dim).to(DEVICE)
        self.linear2 = nn.Linear(self.hidden_dim, Actions.NUM_ACTIONS).to(DEVICE)

    def forward(self, inputs):
        input_vec = vectors.concat_and_flatten(inputs)
        temp_vec = self.linear1(input_vec)
        temp_vec = F.relu(temp_vec).to(DEVICE)
        result = self.linear2(temp_vec)
        return result
