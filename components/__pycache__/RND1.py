import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RandomNetwork(nn.Module):
    def __init__(self,input_shape,output_shape,hidden_size):
        super(RandomNetwork,self).__init__()
        self.fc1=nn.Linear(input_shape,hidden_size)
        self.fc2=nn.Linear(hidden_size,output_shape)
    def forward(self,state):
        state_latent=torch.relu(self.fc1(state))
        state_latent=self.fc2(state_latent)
        return state_latent

class RND:
    def __init__(self, input_shape, output_shape, hidden_size, learning_rate):
        self.input_shape=input_shape
        self.ouput_shape=output_shape
        self.hidden_size=hidden_size
        self.learning_rate=learning_rate
        self.predict_net=RandomNetwork(input_shape,output_shape,hidden_size)
        self.target_net=RandomNetwork(input_shape,output_shape,hidden_size)
        for param in self.target_net.parameters():
            nn.init.normal_(param,mean=0,std=0.1)
        self.optimizer=optim.Adam(self.predict_net.parameters(),lr=self.learning_rate)
    def predict_reward(self,state):
        predict=self.predict_net(state)
        target=self.target_net(state)
        return torch.mean(torch.square(predict-target),dim=-1)
    # update暂时用不上
    def update_target(self):
        self.target_net.load_state_dict(self.predict_net.state_dict())
    def train(self,state):
        state=state.reshape(state.size()[0]*state.size()[1],-1)
        self.optimizer.zero_grad()
        reward=self.predict_reward(state)
        loss=torch.mean(reward)
        loss.backward()
        self.optimizer.step()

if __name__ == '__main__':
    RND1=RND(1,2,3,4)


