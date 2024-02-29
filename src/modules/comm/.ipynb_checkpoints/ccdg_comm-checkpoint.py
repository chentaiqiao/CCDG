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

class Gating(nn.Module):
    def __init__(self, make_obs_ph_n, weights_ph, p_W, act_space_n, before_com_func, channel, after_com_func,
                 q_func, optimizer, com_type, capacity=2, grad_norm_clipping=None, dim_message=4, num_units=64,
                 scope="trainer", reuse=None, beta=0.01):
        super(Gating, self).__init__()

        if com_type == 'full':
            gate = -0.1
        elif com_type == 'no':
            gate = 1.1
        else:
            gate = 0.5

        num_agents = len(make_obs_ph_n)

        act_pdtype_n = [self.make_pdtype(act_space) for act_space in act_space_n]
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(num_agents)]

        self.hiddens_n = [before_com_func(obs_ph_n[i], dim_message, scope="before_com_{}".format(i), num_units=num_units) for
                     i in range(num_agents)]
        before_com_vars_n = [self.scope_vars(self.absolute_scope_name("before_com_{}".format(i))) for i in range(num_agents)]

        hiddens_n_for_message = torch.stack(
            [before_com_func(obs_ph_n[i], dim_message, scope="before_com_{}".format(i), reuse=True, num_units=num_units)
             for i in range(num_agents)], dim=0)  # [num_agent, batch_size, dim_message]

        hiddens_n_for_message = hiddens_n_for_message.permute(1, 0, 2)  # [batch_size, num_agent, dim_message]
        hiddens_n_for_message = hiddens_n_for_message.detach()

        weights = [p_W(obs_ph_n[i], 1, scope="weights_{}".format(i), num_units=num_units) for i in range(num_agents)]
        weights_vars = [self.scope_vars(self.absolute_scope_name("weights_{}".format(i))) for i in range(num_agents)]

        weights2 = torch.cat(weights, dim=1)
        weights2 = weights2.permute(1, 0)
        X = (weights2 >= 0.0).float()
        Z = X.unsqueeze(2)
        matrix = Z.repeat(1, 1, dim_message)

        p_n = []
        a_n = []
        for i in range(num_agents):

            X_new = (weights[i][0] >= gate).float()  # 0.5 as gate

            if np.random.random() < 0.25:
                matrix2 = torch.zeros(dim_message)
            else:
                matrix2 = torch.ones(dim_message)

            matrix2 = matrix2.float()
            matrix_new = matrix_new * matrix2

            a_n.append(self.multiply(hiddens_n[3], matrix_new))

            # --------improve communication-------- #
            p = after_com_func(torch.cat([hiddens_n[i], a_n[i]], 1), act_pdtype_n[i].param_shape()[0],
                               scope="p_func_{}".format(i), num_units=num_units)

            p_n.append(p)

        channel_vars_n = [self.scope_vars(self.absolute_scope_name("channel" + str(i))) for i in range(num_agents)]
        p_func_vars = [self.scope_vars(self.absolute_scope_name("p_func_{}".format(i))) for i in range(num_agents)]

        act_pd_n = [act_pdtype_n[i].pdfromflat(p_n[i]) for i in range(num_agents)]

        act_sample_n = [act_pd.sample() for act_pd in act_pd_n]
        p_reg_n = [torch.mean(torch.square(act_pd.flatparam())) for act_pd in act_pd_n]

        act_input_n_n = [act_ph_n + [] for _ in range(num_agents)]
        weigths_input_n_n = [weights_ph + [] for _ in range(num_agents)]
        for i in range(num_agents):
            act_input_n_n[i][i] = act_pd_n[i].sample()
            weigths_input_n_n[i][i] = weights[i]

        q_input_n = [torch.cat(obs_ph_n + act_input_n_n[i], 1) for i in range(num_agents)]
        q_n = [q_func(q_input_n[i], 1, scope="q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0] for i in
               range(num_agents)]
        pg_loss_n = [-torch.mean(q) for q in q_n]

        pg_loss = torch.sum(pg_loss_n)
        p_reg = torch.sum(p_reg_n)
        loss = pg_loss + p_reg * 1e-3

        w_q_input_n = [torch.cat(obs_ph_n + weigths_input_n_n[i], 1) for i in range(num_agents)]
        w_q_n = [q_func(w_q_input_n[i], 1, scope="w_q_func_{}".format(i), reuse=True, num_units=num_units)[:, 0] for i
                 in range(num_agents)]
        w_pg_loss_n = [-torch.mean(q) for q in w_q_n]
        w_pg_loss = torch.sum(w_pg_loss_n)
        w_loss = w_pg_loss

        var_list = []
        var_list.extend(before_com_vars_n)
        var_list.extend(p_func_vars)
        var_list = list(itertools.chain(*var_list))

        optimizer.zero_grad()
        loss.backward()
        if grad_norm_clipping:
            torch.nn.utils.clip_grad_norm_(var_list, grad_norm_clipping)
        optimizer.step()

        weigths_vars = list(itertools.chain(*weights_vars))
        optimizer_w = optim.Adam(weigths_vars, lr=optimizer.learning_rate)

        optimizer_w.zero_grad()
        w_loss.backward()
        if grad_norm_clipping:
            torch.nn.utils.clip_grad_norm_(weigths_vars, grad_norm_clipping)
        optimizer_w.step()

    def forward(self, obs_ph_n, act_ph_n):
        act = [act_sample_n, weigths, X]
        p_values = p_n

        # target network
        target_hiddens_n = [
            before_com_func(obs_ph_n[i], dim_message, scope="target_before_com_{}".format(i), num_units=num_units) for i
            in range(num_agents)]

        target_before_com_vars = [self.scope_vars(self.absolute_scope_name("target_before_com_{}".format(i))) for i in
                                  range(num_agents)]

        target_hiddens_n_for_message = torch.stack([before_com_func(obs_ph_n[i], dim_message,
                                                                 scope="target_before_com_{}".format(i), reuse=True,
                                                                 num_units=num_units) for i in range(num_agents)],
                                                dim=0)  # [num_agent, batch_size, num_unints]
        target_hiddens_n_for_message = target_hiddens_n_for_message.permute(1, 0, 2)  # [batch_size, num_agent, num_unints]
        target_hiddens_n_for_message = target_hiddens_n_for_message.detach()

        target_weights = [p_W(obs_ph_n[i], 1, scope="target_weights_{}".format(i), num_units=num_units) for i in
                          range(num_agents)]
        target_weights_vars = [self.scope_vars(self.absolute_scope_name("target_weights_{}".format(i))) for i in
                               range(num_agents)]

        weights2_ = torch.cat(target_weights, dim=1)
        weights2_ = weights2_.permute(1, 0)
        X_ = (weights2_ >= 1.1).float()
        Z_ = X_.unsqueeze(2)
        target_matrix = Z_.repeat(1, 1, dim_message)

        target_p_n = []
        target_a_n = []
        for i in range(num_agents):
            X_new_ = (target_weights[i][0] >= gate).float()  # 0.5 as gate
            matrix_new_ = X_new_.repeat(dim_message)

            if np.random.random() < 0.25:
                matrix2 = torch.zeros(dim_message)
            else:
                matrix2 = torch.ones(dim_message)

            matrix2 = matrix2.float()
            matrix_new_ = matrix_new_ * matrix2

            target_a_n.append(matrix_new_)

            # --------improve communication-------- #
            p_ = after_com_func(torch.cat([hiddens_n[i], target_a_n[i]], 1), act_pdtype_n[i].param_shape()[0],
                               scope="p_func_{}".format(i), num_units=num_units)

            target_p_n.append(p_)

        target_p_func_vars = [self.scope_vars(self.absolute_scope_name("target_p_func_{}".format(i))) for i in
                              range(num_agents)]

        target_var_list = []
        target_var_list.extend(target_before_com_vars)
        target_var_list.extend(target_p_func_vars)
        target_var_list = list(itertools.chain(*target_var_list))

        update_target_p = self.make_update_exp(var_list, target_var_list)
        update_target_p_w = self.make_update_exp(list(itertools.chain(*weights_vars)),
                                            list(itertools.chain(*target_weights_vars)))

        target_act_sample_n = [act_pdtype_n[i].pdfromflat(target_p_n[i]).sample() for i in range(num_agents)]
        target_act = [target_act_sample_n, target_weights]

        return act, loss, w_loss, update_target_p, update_target_p_w, {'p_values': p_values, 'target_act': target_act}

