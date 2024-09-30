import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class state_encoder(nn.Module):
    def __init__(self,args):
        super(state_encoder, self).__init__()
        self.args=args
        self.state_dim = int(np.prod(args.state_shape))
        self.n_agents = args.n_agents
        self.state_latent_dim=args.state_latent_dim
        self.state_encoder_avg=nn.Sequential(nn.Linear(self.state_dim,2*self.state_latent_dim),
                                             nn.ReLU(),
                                             nn.Linear(2*self.state_latent_dim,self.state_latent_dim))
        self.transition_model=nn.Sequential(nn.Linear(self.state_latent_dim+self.n_agents*args.n_actions,2*self.state_latent_dim),
                                           nn.ReLU(),
                                           nn.Linear(2*self.state_latent_dim,self.state_latent_dim))
        self.reward_decoder=nn.Sequential(nn.Linear(self.state_latent_dim,self.state_latent_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.state_latent_dim,1))
        self.discount=0.5


    def update_transition_reward_model(self, states, action, next_states, reward):
        h = self.state_encoder_avg(states)
        action1=action.reshape(-1,self.n_agents*self.args.n_actions)
        pred_next_latent_states= self.transition_model(torch.cat([h, action1], dim=1))
        next_h = self.state_encoder_avg(next_states)
        diff = (pred_next_latent_states - next_h.detach())
        loss = torch.mean(0.5 * diff.pow(2))
        pred_next_reward = self.reward_decoder(pred_next_latent_states)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        return total_loss

    def update_encoder_loss(self,states,batch_size,actions,reward):
            h = self.state_encoder_avg(states)

            # Sample random states across episodes at random
            batch_size = batch_size
            perm = np.random.permutation(batch_size)
            h2 = h[perm]

            with torch.no_grad():
                # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
                actions=actions.reshape(-1,self.n_agents*self.args.n_actions)
                pred_next_states_latent = self.transition_model(torch.cat([h, actions], dim=1))
                # reward = self.reward_decoder(pred_next_latent_mu1)
                reward2 = reward[perm]

            z_dist = F.smooth_l1_loss(h, h2, reduction='none')
            r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
            pred_next_states_latent2 = pred_next_states_latent[perm]

            transition_dist = F.smooth_l1_loss(pred_next_states_latent, pred_next_states_latent2 , reduction='none')

            bisimilarity = r_dist + 5*self.discount * transition_dist
            loss = (z_dist - bisimilarity).pow(2).mean()
            return loss
    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()

        transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, L, step)
        encoder_loss = self.update_encoder(obs, action, reward, L, step)
        total_loss = self.bisim_coef * encoder_loss + transition_reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

