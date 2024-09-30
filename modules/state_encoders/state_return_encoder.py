import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch as th

class statereturnencoder(nn.Module):
    def __init__(self,args):
        super(statereturnencoder,self).__init__()
        self.args=args
        self.n_agents=args.n_agents
        self.n_actions=args.n_actions
        self.state_latent_dim=args.state_latent_dim
        self.state_dim=int(np.prod(args.state_shape))
        self.rnn = nn.GRUCell(args.rnn_state_hidden_dim, args.rnn_state_hidden_dim)
        self.fc1= nn.Linear(self.state_dim, args.rnn_state_hidden_dim)
        self.state_encoder_avg=nn.Sequential(
            nn.Linear(self.state_dim,args.state_latent_dim*2),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim*2,args.state_latent_dim)
        )
        self.action_encoder_avg=nn.Sequential(
            nn.Linear((self.n_agents)*(self.n_actions),args.action_latent_dim*2),
            nn.ReLU(),
            nn.Linear(args.action_latent_dim*2,args.action_latent_dim)
        )

        self.return_decoder_avg=nn.Sequential(
            nn.Linear(args.action_latent_dim+args.state_latent_dim,args.state_latent_dim),
            nn.ReLU(),
            nn.Linear(args.state_latent_dim,1)
        )
        self.next_latent_state=nn.Sequential(
             nn.Linear(args.action_latent_dim+args.state_latent_dim,args.state_latent_dim),
             nn.ReLU(),
             nn.Linear(args.state_latent_dim,args.state_latent_dim))

    def forward(self,states):
        states=states.to(self.args.device)
        states_latent_avg=self.state_encoder_avg(states)
        return states_latent_avg
    def predict(self,states,actions_onehot):
        states_latent_avg=self.forward(states)
        #获取动作的onehot形式，以word vector的形式展开
        actions=actions_onehot.view(-1,self.n_agents*self.n_actions)
        actions_latent=self.action_encoder_avg(actions)
        inputs = th.cat([states_latent_avg, actions_latent], dim=-1).to(self.args.device)
        return_decoder=self.return_decoder_avg(inputs)
        next_latent_state=self.next_latent_state(inputs)
        return return_decoder,next_latent_state



 #######以下部分为以GRU去提取时间序列上轨迹中的状态表征###################
    def GRU_forward(self,states,hidden_state):
        states = states.to(self.args.device)
        x = F.relu(self.fc1(states))
        h_in = hidden_state.reshape(-1, self.args.rnn_state_hidden_dim).to(self.args.device)
        h = self.rnn(x, h_in)
        return  h
    def GRU_predict(self,states,actions_onehot,hidden_state):
        states_latent_avg=self.GRU_forward(states,hidden_state)
        actions = actions_onehot.view(-1, self.n_agents * self.n_actions)
        actions_latent=self.action_encoder_avg(actions)
        inputs = th.cat([states_latent_avg, actions_latent], dim=-1).to(self.args.device)
        return_decoder = self.return_decoder_avg(inputs)
        next_state_latent = self.next_latent_state(inputs)
        return states_latent_avg,return_decoder, next_state_latent



    def init_hidden(self):
        return self.fc1.weight.new(self.args.batch_size, self.args.rnn_state_hidden_dim).zero_()





























        
