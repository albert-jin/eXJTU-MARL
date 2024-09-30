from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from modules.state_encoder import REGISTRY as state_encoder_REGISTRY
from modules.state_encoders import REGISTRY as state_return_encoder_REGISTRY
from modules.action_encoders import REGISTRY as action_encoder_REGISTRY
import torch as th
from modules.state_encoder.state_encoder import state_encoder

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args,bis=False,using_state_return=False,using_state_reward=True,using_seq=False):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        ###加入状态表征与动作表征等######

        if using_state_return:
            self.state_encoder = state_return_encoder_REGISTRY[args.state_return_encoder](args)
        if using_state_reward:
            self.state_encoder = state_encoder_REGISTRY[args.state_encoder](args)
            # self.encoder=state_encoder(args)
        if args.using_action_encoder:
            self.action_encoder= action_encoder_REGISTRY[args.action_encoder](args)


        self.hidden_states = None

        if args.using_seq:
            self.GRU_hidden_states = None


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def state_encoder_params(self):
        return list(self.state_encoder.parameters())

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.state_encoder.cuda()
        if self.args.using_action_encoder:
            self.action_encoder.cuda()


    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
#######输入包括观测空间大小，上一个动作，允许的话利用onehot去表示智能体的身份#######
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
########状态表征###################
    def state_repr_forward(self,ep_batch,t):
        return self.state_encoder.predict(ep_batch["state"][:, t], ep_batch["actions_onehot"][:, t])


               ####基于GRU的状态表征#######

    def state_GRU__repr_forward(self, ep_batch, t):
        self.GRU_hidden_states, r, ns=self.state_encoder.GRU_predict(ep_batch["state"][:, t],
                                                                       ep_batch["actions_onehot"][:, t],
                                                                       self.GRU_hidden_states)
        return self.GRU_hidden_states,r,ns


    def init_hidden_GRU(self):
        self.GRU_hidden_states = self.state_encoder.init_hidden()


    def action_encoder_params(self):
        return list(self.action_encoder.parameters())

    def action_repr_forward(self, ep_batch, t):
        return self.action_encoder.predict(ep_batch["obs"][:, t], ep_batch["actions_onehot"][:, t])




