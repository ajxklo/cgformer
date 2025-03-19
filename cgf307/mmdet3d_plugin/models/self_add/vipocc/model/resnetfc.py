from torch import nn  # 导入 PyTorch 的神经网络模块
import torch  # 导入 PyTorch 库

import torch.autograd.profiler as profiler  # 导入 PyTorch 的自动求导分析器，用于性能分析

from mmdet3d_plugin.models.self_add.vipocc.util import util


# from models.common import util  # 导入一个常见的工具模块（假设在你的项目中存在）


class ResnetBlockFC(nn.Module):
    """
    全连接的 ResNet 块类，用于构建深度神经网络中的残差连接。
    该块参考自 DVR 代码（深度体积重建）。

    :param size_in (int): 输入维度
    :param size_out (int): 输出维度
    :param size_h (int): 隐藏层维度
    :param beta (float): Softplus 激活函数的 beta 参数（如果 <= 0，使用 ReLU 激活）
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0):
        super().__init__()

        # 如果没有指定输出维度，默认为输入维度
        if size_out is None:
            size_out = size_in

        # 如果没有指定隐藏层维度，使用输入和输出维度的最小值
        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in  # 保存输入维度
        self.size_h = size_h  # 保存隐藏层维度
        self.size_out = size_out  # 保存输出维度

        # 定义第一个全连接层（fc_0），将输入映射到隐藏层
        self.fc_0 = nn.Linear(size_in, size_h)
        # 定义第二个全连接层（fc_1），将隐藏层映射到输出层
        self.fc_1 = nn.Linear(size_h, size_out)

        # 初始化全连接层的偏置和权重
        nn.init.constant_(self.fc_0.bias, 0.0)  # fc_0 的偏置初始化为 0
        nn.init.kaiming_normal_(self.fc_0.weight, a=0, mode="fan_in")  # fc_0 的权重使用 He 初始化方法
        nn.init.constant_(self.fc_1.bias, 0.0)  # fc_1 的偏置初始化为 0
        nn.init.zeros_(self.fc_1.weight)  # fc_1 的权重初始化为 0

        # 选择激活函数，如果 beta > 0，使用 Softplus，否则使用 ReLU 激活函数
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        # 如果输入和输出维度相同，则跳过连接不需要改变维度，否则需要定义一个额外的线性层（shortcut）
        if size_in == size_out:
            self.shortcut = None  # 不需要额外的线性层
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)  # 定义一个用于调整维度的线性层
            nn.init.constant_(self.shortcut.bias, 0.0)  # shortcut 的偏置初始化为 0
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")  # shortcut 的权重使用 He 初始化方法

    def forward(self, x):
        """
        前向传播函数，计算残差块的输出。

        :param x: 输入张量
        :return: 输出张量
        """
        with profiler.record_function("resblock"):  # 使用性能分析器来分析此函数的执行
            net = self.fc_0(self.activation(x))  # 先通过第一个全连接层并应用激活函数
            dx = self.fc_1(self.activation(net))  # 再通过第二个全连接层并应用激活函数

            # 如果有 shortcut（跳跃连接），则计算 shortcut 的输出
            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x  # 如果没有 shortcut，直接使用输入

            return x_s + dx  # 返回残差连接结果（输入与卷积结果相加）


class ResnetFC(nn.Module):
    """
    全连接的 ResNet 网络，包含多个残差块（ResnetBlockFC），适用于结构化数据的处理。

    :param d_in: 输入的维度
    :param d_out: 输出的维度
    :param n_blocks: 残差块的数量
    :param d_latent: 潜在维度，添加到每个残差块中（0 表示禁用）
    :param d_hidden: 网络中使用的隐藏层维度
    :param beta: Softplus 激活函数的 beta 参数（如果 <= 0，使用 ReLU 激活）
    :param combine_layer: 用于组合内层维度的层数
    :param combine_type: 组合类型，通常是 "average" 或 "max"
    :param use_spade: 是否使用 SPADE（一个条件化网络的技术）
    """

    def __init__(
            self,
            d_in,
            d_out=4,
            n_blocks=5,
            d_latent=0,
            d_hidden=128,
            beta=0.0,
            combine_layer=1000,
            combine_type="average",
            use_spade=False,
    ):
        super().__init__()

        # 如果输入维度大于 0，定义输入全连接层，将输入映射到隐藏层
        if d_in > 0:
            self.lin_in = nn.Linear(d_in, d_hidden)  # 例如，将 103 的输入维度映射到 64 的隐藏层维度
            nn.init.constant_(self.lin_in.bias, 0.0)  # 输入层的偏置初始化为 0
            nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")  # 使用 He 初始化方法

        # 定义输出层，将隐藏层映射到输出
        self.lin_out = nn.Linear(d_hidden, d_out)  # 将 64 的隐藏层映射到输出维度（例如 1）
        nn.init.constant_(self.lin_out.bias, 0.0)  # 输出层的偏置初始化为 0
        nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")  # 输出层的权重初始化为 He 初始化

        # 网络的其他参数
        self.n_blocks = n_blocks  # 残差块的数量
        self.d_latent = d_latent  # 潜在维度
        self.d_in = d_in  # 输入维度
        self.d_out = d_out  # 输出维度
        self.d_hidden = d_hidden  # 隐藏层维度

        self.combine_layer = combine_layer  # 组合层，控制何时组合内层维度
        self.combine_type = combine_type  # 组合方式，通常为 "average" 或 "max"
        self.use_spade = use_spade  # 是否使用 SPADE

        # 定义多个残差块
        self.blocks = nn.ModuleList(
            [ResnetBlockFC(d_hidden, beta=beta) for i in range(n_blocks)]  # 创建 n_blocks 个残差块
        )

        # 如果潜在维度大于 0，定义潜在层和调整维度的线性层
        if d_latent != 0:
            n_lin_z = min(combine_layer, n_blocks)  # 选择最小值作为潜在层的数量
            self.lin_z = nn.ModuleList(
                [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]  # 创建与潜在维度相关的线性层
            )
            for i in range(n_lin_z):
                nn.init.constant_(self.lin_z[i].bias, 0.0)  # 初始化线性层的偏置为 0
                nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")  # 使用 He 初始化方法

            # 如果使用 SPADE，则需要额外定义 scaling 层
            if self.use_spade:
                self.scale_z = nn.ModuleList(
                    [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]  # 为每个潜在维度创建一个 scaling 层
                )
                for i in range(n_lin_z):
                    nn.init.constant_(self.scale_z[i].bias, 0.0)  # 初始化 scaling 层的偏置为 0
                    nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")  # 使用 He 初始化方法

        # 根据 beta 参数选择激活函数（Softplus 或 ReLU）
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, zx, combine_inner_dims=(1,)):
        """
        前向传播函数，计算整个全连接 ResNet 网络的输出。

        :param zx: 输入张量，包含了潜在空间和输入特征
        :param combine_inner_dims: 用于多视图输入的维度组合方式
        :return: 输出张量
        """
        assert zx.size(-1) == self.d_latent + self.d_in  # 确保输入的维度大小正确
        if self.d_latent > 0:
            z = zx[..., : self.d_latent]  # 提取潜在变量 z
            x = zx[..., self.d_latent:]  # 提取输入变量 x
        else:
            x = zx  # 如果没有潜在维度，直接使用输入

        if self.d_in > 0:
            x = self.lin_in(x)  # 如果输入维度大于 0，经过输入层全连接
        else:
            x = torch.zeros(self.d_hidden, device=zx.device)  # 否则，初始化为全 0 张量

        # 通过多个残差块
        for blkid in range(self.n_blocks):
            if blkid == self.combine_layer:
                x = util.combine_interleaved(
                    x, combine_inner_dims, self.combine_type  # 根据需要组合内层维度
                )

            if self.d_latent > 0 and blkid < self.combine_layer:
                tz = self.lin_z[blkid](z)  # 通过潜在层线性映射潜在变量
                if self.use_spade:
                    sz = self.scale_z[blkid](z)  # 通过 scaling 层调整潜在变量
                    x = sz * x + tz  # 应用 scaling 和潜在变量
                else:
                    x = x + tz  # 直接加上潜在变量

            x = self.blocks[blkid](x)  # 通过残差块

        out = self.lin_out(self.activation(x))  # 通过输出层并应用激活函数
        return out  # 返回最终输出

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        """
        根据配置文件创建 ResnetFC 网络实例

        :param conf: 配置字典
        :param d_in: 输入维度
        :return: ResnetFC 网络实例
        """
        return cls(
            d_in,
            n_blocks=conf.get("n_blocks", 5),  # 从配置中获取残差块数量
            d_hidden=conf.get("d_hidden", 128),  # 从配置中获取隐藏层维度
            beta=conf.get("beta", 0.0),  # 从配置中获取 beta 值
            combine_layer=conf.get("combine_layer", 1000),  # 从配置中获取组合层
            combine_type=conf.get("combine_type", "average"),  # 从配置中获取组合方式
            use_spade=conf.get("use_spade", False),  # 从配置中获取是否使用 SPADE
            **kwargs  # 处理其他额外参数
        )
