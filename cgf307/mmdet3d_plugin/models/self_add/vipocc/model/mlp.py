import torch  # 导入 PyTorch 库
from torch import nn  # 导入 PyTorch 的神经网络模块
import numpy as np  # 导入 NumPy 用于数值计算

from mmdet3d_plugin.models.self_add.vipocc.util import util


# from models.common import util  # 导入一个实用工具模块（假定包含一些辅助函数）


class ImplicitNet(nn.Module):
    """
    表示一个多层感知机（MLP）。
    这段代码来自 IGR（隐式神经表示）。
    """

    def __init__(
            self,
            d_in,
            dims,
            skip_in=(),
            d_out=4,
            geometric_init=True,
            radius_init=0.3,
            beta=0.0,
            output_init_gain=2.0,
            num_position_inputs=3,
            sdf_scale=1.0,
            dim_excludes_skip=False,
            combine_layer=1000,
            combine_type="average",
    ):
        """
        初始化 ImplicitNet 模型。

        :param d_in: 输入的维度大小
        :param dims: 隐藏层的维度列表
        :param skip_in: 从输入层到某些层的跳跃连接（残差连接）
        :param d_out: 输出的维度大小
        :param geometric_init: 是否使用几何初始化
        :param radius_init: 如果使用几何初始化，SDF（有符号距离函数）球体的半径
        :param beta: Softplus 激活函数的 beta 值
        :param output_init_gain: 输出层的标准差初始化
        :param num_position_inputs: 位置输入的特征数量
        :param sdf_scale: 有符号距离函数的缩放因子
        :param dim_excludes_skip: 是否排除跳跃连接的维度计算
        :param combine_layer: 用于组合特征的层
        :param combine_type: 特征组合的方法（'average' 或 'max'）
        """
        super().__init__()

        dims = [d_in] + dims + [d_out]  # 构建包含输入、隐藏层和输出层的维度列表
        if dim_excludes_skip:
            for i in range(1, len(dims) - 1):
                if i in skip_in:
                    dims[i] += d_in  # 如果使用跳跃连接，调整隐藏层的维度

        self.num_layers = len(dims)  # 层数等于维度列表的长度
        self.skip_in = skip_in  # 保存跳跃连接的层信息
        self.dims = dims  # 保存每一层的维度信息
        self.combine_layer = combine_layer  # 保存用于组合特征的层号
        self.combine_type = combine_type  # 保存特征组合的方法（'average' 或 'max'）

        # 遍历所有层，定义每一层
        for layer in range(0, self.num_layers - 1):  # 遍历所有除最后一层之外的层
            if layer + 1 in skip_in:  # 如果当前层存在跳跃连接
                out_dim = dims[layer + 1] - d_in  # 减去输入维度大小
            else:
                out_dim = dims[layer + 1]  # 否则使用当前层的维度大小

            lin = nn.Linear(dims[layer], out_dim)  # 定义一个全连接层

            # 如果启用了几何初始化，进行初始化
            if geometric_init:
                if layer == self.num_layers - 2:  # 如果是倒数第二层（输出层之前的隐藏层）
                    # 对最后一层进行几何初始化，使用 SDF 半径
                    nn.init.normal_(
                        lin.weight[0],
                        mean=-np.sqrt(np.pi) / np.sqrt(dims[layer]) * sdf_scale,
                        std=0.00001,
                    )
                    nn.init.constant_(lin.bias[0], radius_init)  # 设置偏置
                    if d_out > 1:  # 如果输出维度大于 1，初始化其它部分
                        nn.init.normal_(lin.weight[1:], mean=0.0, std=output_init_gain)
                        nn.init.constant_(lin.bias[1:], 0.0)
                else:
                    # 对其他层进行普通的权重和偏置初始化
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if d_in > num_position_inputs and (layer == 0 or layer in skip_in):
                    # 特殊处理输入层，允许位置编码
                    nn.init.constant_(lin.weight[:, -d_in + num_position_inputs:], 0.0)
            else:
                # 如果没有启用几何初始化，使用 Kaiming 正态分布初始化
                nn.init.constant_(lin.bias, 0.0)
                nn.init.kaiming_normal_(lin.weight, a=0, mode="fan_in")

            setattr(self, "lin" + str(layer), lin)  # 将层添加到模型中

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)  # 如果 beta 大于 0，使用 Softplus 激活函数
        else:
            # 默认使用 ReLU 激活函数
            self.activation = nn.ReLU()

    def forward(self, x, combine_inner_dims=(1,)):
        """
        模型的前向传播函数。

        :param x: 输入张量，形状为 (..., d_in)
        :param combine_inner_dims: 用于多视角输入的维度组合
        :return: 经过网络层处理后的输出张量
        """
        x_init = x  # 保存原始输入，用于跳跃连接
        for layer in range(0, self.num_layers - 1):  # 遍历所有层
            lin = getattr(self, "lin" + str(layer))  # 获取当前层

            if layer == self.combine_layer:  # 如果是用于组合特征的层
                x = util.combine_interleaved(x, combine_inner_dims, self.combine_type)  # 合并多视角特征
                x_init = util.combine_interleaved(x_init, combine_inner_dims, self.combine_type)  # 同样处理原始输入

            if layer < self.combine_layer and layer in self.skip_in:
                x = torch.cat([x, x_init], -1) / np.sqrt(2)  # 拼接跳跃连接的输入，并缩放

            x = lin(x)  # 应用线性变换（全连接层）
            if layer < self.num_layers - 2:
                x = self.activation(x)  # 除了最后一层外，应用激活函数

        return x  # 返回最终的输出张量

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        """
        从配置中创建模型的类方法。

        :param conf: 配置对象，包含模型的参数
        :param d_in: 输入维度
        :return: 返回创建的 ImplicitNet 模型实例
        """
        return cls(
            d_in,
            conf.get_list("dims"),  # 隐藏层的维度列表
            skip_in=conf.get_list("skip_in"),  # 跳跃连接的层
            beta=conf.get_float("beta", 0.0),  # Softplus 激活函数的 beta 值
            dim_excludes_skip=conf.get_bool("dim_excludes_skip", False),  # 是否排除跳跃连接的维度计算
            combine_layer=conf.get_int("combine_layer", 1000),  # 用于组合特征的层
            combine_type=conf.get_string("combine_type", "average"),  # 特征组合的方法（'average' 或 'max'）
            **kwargs  # 其他附加参数
        )
