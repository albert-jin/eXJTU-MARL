import copy

import numpy as np
import torch
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from torch.optim import Adam 
import torch.nn.functional as F


class QLearner:
    def __init__(self, mac, scheme, logger, args,intricRND):
        self.args = args
        self.mac = mac
        self.RND=intricRND
        self.logger = logger
        self.n_agents=args.n_agents
        self.bisim_coef=0.5

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # self.optimiser = Adam(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.optimiser = Adam(params=self.params)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        #state encoder
        if args.using_state_encoder:
            self.state_encoder_params = list(mac.state_encoder_params())
            self.state_encoder_optimiser = Adam(params=self.state_encoder_params, lr=args.splr)

        ##动作隐空间
        if args.using_action_encoder:
            self.action_encoder_params = list(self.mac.action_encoder_params())
            self.action_encoder_optimiser = Adam(params=self.action_encoder_params, lr=args.lr)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        state=batch["state"][:,:-1]
        rewards = batch["reward"][:, :-1]
        feature=batch["feature"][:,:-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)


        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)

        mac_out = th.stack(mac_out, dim=1)  # Concat over time


        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999


        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # feature1=torch.zeros_like(feature)
        # feature1[:,1:]=feature[:,0:-1]
        # change=feature-feature1

        # change=torch.dist(feature[:,0],feature[:,int(batch.max_seq_length/2)],2)
        # # 鼓励change越小越好
        # change=F.cosine_similarity(feature[:,0],feature[:,int(batch.max_seq_length/2)],dim=-1)

        # change_mean=change.mean()
        # feature_sample=feature[:,int(batch.max_seq_length/2)]
        # ##鼓励这个抽取的特征尽量均匀分布######
        # d=torch.zeros((batch.batch_size,batch.batch_size))


        # for i in range(batch.batch_size):
        #     for j in range(i+1,batch.batch_size):
        #         d[i,j]=d[j,i]=torch.dist(feature_sample[i],feature_sample[j],p=2)
        #
        # d=d.mean(dim=-1).to(self.args.device)
        # d=d.view(32,1,1)




        # Calculate 1-step Q-Learning targets
        # inreward=torch.norm(change,p=2,dim=2,keepdim=True)
        # b=0.99


        # rewards=rewards+0.1*b**(t_env/10)*inreward
        inreward_ls=[]
        with torch.no_grad():
            for t in range(batch.max_seq_length - 1):
                inreward = self.RND.predict_reward(state[:, t])
                inreward_ls.append(inreward.reshape(-1, 1))
            inreward = torch.stack(inreward_ls, dim=1)
        self.RND.train(state)
        # rewards=rewards+0.001*inreward

        targets = rewards+self.args.gamma * (1 - terminated) * target_max_qvals

        # feature1=torch.zeros_like(feature)
        # feature1[:,1:]=feature[:,0:-1]
        # change=feature-feature1
        # change=torch.sum(change**2,dim=-1,keepdim=True)
        # change[:,0]=0
        # # Td-error
        # b=0.99**(episode_num/50)
        # change=b*torch.log(change+1)
      
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask




        # Normal L2 loss, take mean over actual data
        # loss = (d*0.5*masked_td_error ** 2).sum() / mask.sum()+0.1*change_mean
        # loss = (d * 0.5 * masked_td_error ** 2).sum() / mask.sum()
        # loss = (masked_td_error ** 2).sum() / mask.sum() + 0.1 * change_mean
        # loss = (masked_td_error ** 2).sum() / mask.sum()



        loss = 0.5*(masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        if t_env<0:
            #####状态表征训练方法######
            rt_pred = []
            ns_pred = []
            n_s_l = []
            if self.args.using_action_encoder:
                pred_obs_loss = None
                pred_r_loss = None
                pred_grad_norm = None
                if t_env>0:
                    # train action encoder
                    no_pred = []
                    r_pred = []
                    for t in range(batch.max_seq_length):
                        no_preds, r_preds = self.mac.action_repr_forward(batch, t=t)
                        no_pred.append(no_preds)
                        r_pred.append(r_preds)
                    no_pred = th.stack(no_pred, dim=1)[:, :-1]  # Concat over time
                    r_pred = th.stack(r_pred, dim=1)[:, :-1]
                    no = batch["obs"][:, 1:].detach().clone()
                    repeated_rewards = batch["reward"][:, :-1].detach().clone().unsqueeze(2).repeat(1, 1, self.n_agents,
                                                                                                    1)

                    pred_obs_loss = th.sqrt(((no_pred - no) ** 2).sum(dim=-1)).mean()
                    pred_r_loss = ((r_pred - repeated_rewards) ** 2).mean()

                    pred_loss = pred_obs_loss + 10 * pred_r_loss
                    self.action_encoder_optimiser.zero_grad()
                    pred_loss.backward()
                    pred_grad_norm = th.nn.utils.clip_grad_norm_(self.action_encoder_params, self.args.grad_norm_clip)
                    self.action_encoder_optimiser.step()





            elif (self.args.using_GRU):
                self.mac.init_hidden_GRU(batch.batch_size)
                n_s=torch.zeros([32,15])
                for t in range(batch.max_seq_length):
                    with torch.no_grad():
                        n_s = self.mac.state_encoder.GRU_forward(states[:, t], n_s)
                    h, rt_preds, ns_preds = self.mac.state_GRU__repr_forward(batch, t=t)
                    ns_pred.append(ns_preds)
                    rt_pred.append(rt_preds)
                    n_s_l.append(n_s)

                ns1 = th.stack(n_s_l, dim=1)[:, :-2]
                ns_pred = th.stack(ns_pred, dim=1)[:, :-2]  # Concat over time
                rt_pred = th.stack(rt_pred, dim=1)[:, :-1]


            ###不使用GRU情况下##############
            else:
                transition_reward = []
                encoder = []
                total = []
                self.states_encoder = self.mac.state_encoder
                for t in range(batch.max_seq_length - 1):
                    states = batch["state"][:, t]
                    actions_onehot = batch["actions_onehot"][:, t]
                    next_states = batch["state"][:, t + 1]
                    rewards = batch["reward"][:, t]
                    transition_reward_loss = self.states_encoder.update_transition_reward_model(states, actions_onehot,
                                                                                                next_states, rewards)
                    transition_reward.append(transition_reward_loss)
                    encoder_loss = self.states_encoder.update_encoder_loss(states,self.args.batch_size, actions_onehot, rewards)
                    encoder.append(encoder_loss)
                    total_loss = 5* encoder_loss + transition_reward_loss
                    total.append(total_loss)

                # transition_rew = th.stack(transition_reward, dim=0)
                # encoder_l = th.stack(transition_reward, dim=0)
                total = th.stack(total, dim=0)
                total1 = ((total) ** 2).mean()
                self.state_encoder_optimiser.zero_grad()
                total1.backward()
                self.state_encoder_optimiser.step()
                # state_latent_encoder = self.mac.state_encoder.state_encoder_avg
                # for t in range(int(batch.max_seq_length-1)):
                #     ns = batch["state"][:, t + 1]
                #     with torch.no_grad():
                #         n_s_latent = state_latent_encoder(ns)
                #     n_s_l.append(n_s_latent)
                #     rt_preds, ns_preds = self.mac.state_repr_forward(batch, t=t)
                #     ns_pred.append(ns_preds)
                #     rt_pred.append(rt_preds)
                
                # ns_pred = th.stack(ns_pred, dim=1)[:, :-1]  # Concat over time
                # rt_pred = th.stack(rt_pred, dim=1)[:, :]
                # ns1 = th.stack(n_s_l, dim=1)[:, :-1]  # Concat over time
                
                # # ns1 = batch["feature"][:, 1:-1].detach().clone()
                # repeated_returns = batch["episode_return"][:, :-1].detach().clone()
                
                # pred_states_loss = th.sqrt(((ns_pred - ns1) ** 2).sum(dim=-1)).mean()
                
                
                # pred_return_loss = ((rt_pred - repeated_returns) ** 2).mean()
                
                
                # pred_loss = pred_states_loss + 10* pred_return_loss
                # self.state_encoder_optimiser.zero_grad()
                # pred_loss.backward()
                # pred_grad_norm = th.nn.utils.clip_grad_norm_(self.state_encoder_params, self.args.grad_norm_clip)
                # self.state_encoder_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # self.logger.log_stat("pred_obs_loss", pred_obs_loss.item(), t_env)
            # self.logger.log_stat("pred_r_loss", pred_r_loss.item(), t_env)
            # self.logger.log_stat("pred_loss", pred_loss.item(), t_env)

            self.logger.log_stat("loss", loss.item(), t_env)


            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
