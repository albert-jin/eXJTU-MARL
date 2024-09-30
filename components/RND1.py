import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RandomNetwork(nn.Module):
    def __init__(self,input_shape,output_shape,hidden_size):
        super(RandomNetwork,self).__init__()
        self.fc1=nn.Linear(input_shape,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,output_shape)
    def forward(self,state):
        state_latent=torch.relu(self.fc1(state))
        state_latent=self.fc2(state_latent)
        state_latent=self.fc3(state_latent)
        state_latent=self.fc4(state_latent)
        return state_latent
class RandomNetwork1(nn.Module):
    def __init__(self,input_shape,output_shape,hidden_size):
        super(RandomNetwork1,self).__init__()
        self.fc1=nn.Linear(input_shape,hidden_size)
        self.fc2=nn.Linear(hidden_size,hidden_size)
        self.fc3=nn.Linear(hidden_size,hidden_size)
        self.fc4=nn.Linear(hidden_size,output_shape)
    def forward(self,state):
        state_latent=torch.relu(self.fc1(state))
        state_latent=self.fc2(state_latent)
        state_latent=self.fc3(state_latent)
        state_latent=self.fc4(state_latent)
        return state_latent

class RND:
    def __init__(self, input_shape, output_shape, hidden_size, learning_rate):
        self.input_shape=input_shape
        self.ouput_shape=output_shape
        self.hidden_size=hidden_size
        self.learning_rate=learning_rate
        self.predict_net=RandomNetwork1(input_shape,output_shape,hidden_size)
        self.target_net=RandomNetwork(input_shape,output_shape,hidden_size)
        for param in self.target_net.parameters():
            param.stop_gradient=True
            param.data=param.sign()*param.abs().sqrt()
        self.optimizer=optim.Adam(self.predict_net.parameters(),lr=self.learning_rate)
    def predict_reward(self,state):
        predict=self.predict_net(state)
        target=self.target_net(state)
        return torch.mean(torch.square(predict-target),dim=-1)
    # update暂时用不上
    def update_target(self):
        self.target_net.load_state_dict(self.predict_net.state_dict())
    def train(self,state):
        for i in range(state.size()[0]):
             self.optimizer.zero_grad()
             reward=self.predict_reward(state)[i,:]
             loss=torch.mean(reward)
             loss.backward()
             self.optimizer.step()
       
    def cuda(self):
        self.predict_net.cuda()
        self.target_net.cuda()  


if __name__ == '__main__':
    RND1=RND(1,2,3,4)


