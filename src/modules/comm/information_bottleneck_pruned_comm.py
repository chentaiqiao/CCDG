import torch.nn as nn
import torch.nn.functional as F
class IBPComm(nn.Module):
    def __init__(self, input_shape, args):
        super(IBPComm, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.comm_embed_dim * 2)
        self.inference_model = nn.Sequential(
            nn.Linear(args.comm_embed_dim, 2*args.comm_embed_dim),
            nn.ReLU(True),
            nn.Linear(2*args.comm_embed_dim, 2 * args.comm_embed_dim),
            nn.ReLU(True),
            nn.Linear(2 * args.comm_embed_dim, args.atom)
        )
        self.gate = nn.Sequential(
            nn.Linear(args.comm_embed_dim, 16),
            nn.ReLU(True),
            nn.Linear(16, args.comm_embed_dim)
        )
    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        gaussian_params = self.fc2(x)
        mu = gaussian_params[:, :self.args.comm_embed_dim]
        sigma = F.softplus(gaussian_params[:, self.args.comm_embed_dim:])
        return mu, sigma
'''
“_communicate”方法的门控部分是使用基于 sigmoid 函数的门控机制实现的。 以下是门控执行方式的细分：
1. 该方法接收输入，其中包括来自其他代理的消息。
2. 计算消息的平均值，然后通过绝对函数(mu_d = th.abs(mu.view(bs*self.n_agents, -1).detach()))进行处理。
3. 处理后的平均值通过 sigmoid 门控网络传递，该网络由一个线性层和后跟 ReLU 激活 (self.comm.gate) 组成。
4. 使用与处理平均值相同形状的均匀分布生成随机值 (mask = th.rand(mu_d.shape).cuda())。
5. 将 sigmoid 门值与随机值进行比较。 如果门值大于随机值，则认为是激活的； 否则，它被设置为零 (g = th.where(g > mask, g, th.zeros(mu_d.shape).cuda()))。
6. 消息乘以门值 (ms = ms * g)。
7. 返回生成的门控消息。
门机制根据 sigmoid 门确定的激活级别过滤消息。 具有较高激活值的消息通过，而具有较低激活值的消息通过将其值设置为零来有效地过滤掉。 这种门控机制允许在多代理通信网络中的代理之间进行选择性通信。
门控机制中使用的激活值是根据来自其他代理的消息的平均值计算的。 计算方法如下：
1. 消息的平均值是通过沿消息张量的第二维取平均值计算的 (mu_d = th.abs(mu.view(bs*self.n_agents, -1).detach()))。
2. 然后使用绝对函数 (mu_d = th.abs(mu.view(bs*self.n_agents, -1).detach())) 处理平均值。
    - 此步骤确保激活值基于平均值的大小，而忽略方向或符号。
    - 通过取绝对值，门控机制关注消息的整体强度，而不是分别考虑正值或负值。
然后将处理后的平均值 (mu_d) 用作 sigmoid 门控网络的输入以获得激活值。 门控网络由一个线性层和一个 ReLU 激活层 (self.comm.gate) 组成。 该网络的输出是一个激活值向量，用于确定每条消息的强度或影响。 较高的激活值表示较强的消息，而较低的激活值表示较弱的消息。
激活值随后用于门控过程，在门控过程中将它们与使用均匀分布生成的随机值进行比较。 激活值大于相应随机值的消息被认为是活跃的并通过，而激活值小于随机值的消息通过将它们的值设置为零来有效地过滤掉。
'''