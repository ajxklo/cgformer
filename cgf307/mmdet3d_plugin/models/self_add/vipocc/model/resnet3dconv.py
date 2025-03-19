import torch  # 导入 PyTorch 库
import torch.autograd.profiler as profiler  # 导入 PyTorch 的自动求导分析器，用于性能分析
from torch import nn  # 导入 PyTorch 的神经网络模块


class ResnetBlock3DConv(nn.Module):
    """
    这是一个 3D 卷积的 ResNet 块类。该块用于构建深层网络中的残差连接。
    代码参考自 DVR（深度体积重建）算法。

    :param size_in (int): 输入维度
    :param size_out (int): 输出维度
    :param size_h (int): 隐藏层维度
    :param beta (float): Softplus 激活函数的 beta 参数（如果 <= 0，使用 ReLU 激活）
    :param kernel_size (int): 卷积核大小
    :param stride (int): 卷积步幅
    :param padding (int): 卷积的填充
    """

    def __init__(self, size_in, size_out=None, size_h=None, beta=0.0, kernel_size=1, stride=1, padding=0):
        super().__init__()

        # 初始化输入、隐藏和输出维度
        if size_out is None:
            size_out = size_in  # 如果没有指定输出维度，则使用输入维度作为输出维度

        if size_h is None:
            size_h = min(size_in, size_out)  # 如果没有指定隐藏层维度，则使用输入和输出维度的最小值

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # 定义 3D 卷积层
        self.conv_0 = nn.Conv3d(size_in, size_h, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_1 = nn.Conv3d(size_h, size_out, kernel_size=kernel_size, stride=stride, padding=padding)

        # 初始化卷积层的权重和偏置
        nn.init.constant_(self.conv_0.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_0.weight, a=0, mode="fan_in")  # 使用 He 初始化方法
        nn.init.constant_(self.conv_1.bias, 0.0)
        nn.init.zeros_(self.conv_1.weight)  # 第二个卷积层的权重初始化为 0

        # 根据 beta 参数选择激活函数（Softplus 或 ReLU）
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        # 如果输入和输出维度相同，跳过连接不需要改变维度，否则需要定义一个额外的卷积层
        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv3d(size_in, size_out, bias=False, kernel_size=kernel_size, stride=stride,
                                      padding=padding)
            nn.init.constant_(self.shortcut.bias, 0.0)
            nn.init.kaiming_normal_(self.shortcut.weight, a=0, mode="fan_in")

    def forward(self, x):
        """
        前向传播函数，计算残差块的输出。

        :param x: 输入张量
        :return: 输出张量
        """
        with profiler.record_function("resblock"):  # 使用性能分析器来分析此函数的执行
            net = self.conv_0(self.activation(x))  # 先通过第一个卷积层并应用激活函数
            dx = self.conv_1(self.activation(net))  # 再通过第二个卷积层并应用激活函数

            # 如果有 shortcut（跳跃连接），则计算 shortcut 的输出
            if self.shortcut is not None:
                x_s = self.shortcut(x)
            else:
                x_s = x  # 如果没有 shortcut，直接使用输入

            return x_s + dx  # 返回残差连接结果（输入与卷积结果相加）


class Resnet3DConv(nn.Module):
    """
    这是一个 3D 卷积的 ResNet 网络，包含多个残差块（ResnetBlock3DConv）。

    :param d_in: 输入的维度
    :param d_out: 输出的维度
    :param n_blocks: 残差块的数量
    :param d_hidden: 隐藏层的维度
    :param beta: Softplus 激活函数的 beta 参数（如果 <= 0，使用 ReLU 激活）
    :param kernel_size: 卷积核大小
    :param stride: 卷积步幅
    :param padding: 卷积的填充
    """

    def __init__(
            self,
            d_in,
            d_out=4,
            n_blocks=5,
            d_hidden=128,
            beta=0.0,
            kernel_size=1,
            stride=1,
            padding=0,
    ):
        super().__init__()

        # 如果输入维度大于 0，定义输入卷积层
        if d_in > 0:
            self.conv_in = nn.Conv3d(d_in, d_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
            nn.init.constant_(self.conv_in.bias, 0.0)
            nn.init.kaiming_normal_(self.conv_in.weight, a=0, mode="fan_in")

        # 定义输出卷积层
        self.conv_out = nn.Conv3d(d_hidden, d_out, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.constant_(self.conv_out.bias, 0.0)
        nn.init.kaiming_normal_(self.conv_out.weight, a=0, mode="fan_in")

        self.n_blocks = n_blocks  # 残差块的数量
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        # 定义多个残差块
        self.blocks = nn.ModuleList(
            [ResnetBlock3DConv(d_hidden, beta=beta, kernel_size=kernel_size, stride=stride, padding=padding) for _ in
             range(n_blocks)]
        )

        # 根据 beta 参数选择激活函数（Softplus 或 ReLU）
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

    def forward(self, zx):
        """
        前向传播函数，计算整个 3D ResNet 网络的输出。

        :param zx: 输入张量，包含了潜在空间和输入特征
        :return: 输出张量
        """
        with profiler.record_function("resnet3dconv_infer"):  # 使用性能分析器来分析此函数的执行
            x = zx
            if self.d_in > 0:
                x = self.conv_in(x)  # 如果输入维度大于 0，经过输入卷积层
            else:
                x = torch.zeros(self.d_hidden, device=zx.device)  # 如果输入维度为 0，初始化为全 0 的张量

            # 通过多个残差块
            for blkid in range(self.n_blocks):
                x = self.blocks[blkid](x)

            # 最后通过输出卷积层
            out = self.conv_out(self.activation(x))
            return out

    @classmethod
    def from_conf(cls, conf, d_in, d_out, **kwargs):
        """
        从配置文件构建模型的类方法。

        :param conf: 配置对象，包含网络的各种参数
        :param d_in: 输入维度
        :param d_out: 输出维度
        :return: 返回构建的 Resnet3DConv 实例
        """
        return cls(
            d_in,
            d_out,
            n_blocks=conf.get("n_blocks", 2),  # 残差块的数量，默认值为 2
            d_hidden=conf.get("d_hidden", 128),  # 隐藏层的维度，默认值为 128
            beta=conf.get("beta", 0.0),  # Softplus 激活函数的 beta 参数
            kernel_size=conf.get("kernel_size", 1),  # 卷积核大小
            stride=conf.get("stride", 1),  # 卷积步幅
            padding=conf.get("padding", 0),  # 卷积填充
            **kwargs  # 其他附加参数
        )
