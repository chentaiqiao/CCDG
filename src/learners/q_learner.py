import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from collections import deque  


class CCDGQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.Pi_com_list = deque(maxlen=args.H)  #改写,初始化Pi_com_list

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

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
    def cal_q(self,action,batch,t):
        # Max over target Q-Values
        max_qvals = action.max(dim=3)[0]
        # Mix
        if self.mixer is not None:
            target_max_qvals = self.target_mixer(max_qvals, batch["state"][t, 1:])
        return target_max_qvals

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        args=self.args
        obs=batch["obs"][:, :-1]
        p_i=batch["p_i"][:, :-1]
        rewards = batch["reward"][:, :-1]
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
            #改写,添加gated loss
            # Construct gate for each agent
            gate = []
            for i in range(args.n_agents):
                g = [1] * args.n_agents
                g[i] = 0
                gate.append(g)
            
            agent_outs_com = self.mac.forward(batch, t=t, gate=[1] * args.n_agents)
            a_com = agent_outs_com  # Result for gate=[1,1,1,...,1]
            
            a_ncom = []
            for i, g in enumerate(gate):
                if g != [1] * args.n_agents:
                    agent_outs_ncom = self.mac.forward(batch, t=t, gate=g)
                    a_ncom.append(agent_outs_ncom)
            
            # Calculate q_com and q_ncom
            q_com = self.cal_q(a_com,batch,t) # ,实现cal_q
            q_ncom = []
            for a in a_ncom:
                q_ncom.append(self.cal_q(a,batch,t))  
            
            # Calculate Pi_com
            Pi_com = q_com - q_ncom[args.n_agents - 1]
            
            # Expand Pi_com for each time step
            if len(self.Pi_com_list) == args.H:#满了
                self.Pi_com_list.popleft()  #删除过期元素
                
            self.Pi_com_list.append(Pi_com)
            # Gated end
            
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
            mac_out[avail_actions == 0] = -9999999
            cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        
        #改写
        # Sort Pi_com_list
        Pi_com_list.sort()#L_Pi_com[],升序
        
        # Compare Pi_com_list with current Pi_com_list for each agent
        Y_o = []
        for i in range(args.n_agents):
            threshold = args.prob * len(Pi_com_list)#T=threshold
            if Pi_com_list[i] > threshold:
                Y_o.append(1)
            else:
                Y_o.append(0)
        
        # Compute cross-entropy loss between p_i and Y_o
        Gated_loss = -th.mean(p_i * th.log(Y_o) + (1 - p_i) * th.log(1 - Y_o))
        loss=loss+Gated_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        #self.logger.console_logger.info("Updated target network")

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
