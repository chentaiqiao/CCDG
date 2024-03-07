import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CCDGComm(nn.Module):
    def __init__(self, input_shape, args):
        super(CCDGComm, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.value = nn.Linear(input_shape + args.rnn_hidden_dim, args.comm_embed_dim)
        self.signature = nn.Linear(input_shape + args.rnn_hidden_dim, args.signature_dim)
        self.query = nn.Linear(input_shape + args.rnn_hidden_dim, args.signature_dim)

    def forward(self, inputs):
        massage = self.value(inputs)
        signature = self.signature(inputs)
        query = self.query(inputs)
        
        return massage, signature, query

class Gated_Net(nn.Module):
    def __init__(self, input_shape, args, rnn_hidden_dim=32, num_layers=2):
        super(Gated_Net, self).__init__()
        #print('input_shape+ args.rnn_hidden_dim',input_shape+ args.rnn_hidden_dim)#106
        
        # Define the MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_shape+ args.rnn_hidden_dim, rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim,rnn_hidden_dim )
        )
        
        # Define the gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 1),#32*1
            nn.Sigmoid()
        )
        
        # Initialize the weights and biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Pass the input through the MLP layers
        x = self.mlp(x)
        #print('x',x.shape)#[num,32]
        
        # Apply the gating mechanism
        gate = self.gate(x)
        #x = gate * x
        
        return gate#n_agents

