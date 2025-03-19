import torch
import numpy as np
import torch.autograd.profiler as profiler


class PositionalEncoding(torch.nn.Module):
    """
    实现 NeRF (Neural Radiance Fields) 的位置编码功能
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        """
        初始化位置编码模块
        :param num_freqs: 频率数量，控制编码的复杂程度（默认为6）
        :param d_in: 输入维度（默认为3，例如3D坐标 x, y, z）
        :param freq_factor: 频率因子，用于生成频率序列（默认为π）
        :param include_input: 是否包括原始输入作为输出的一部分（默认为True）
        """
        super().__init__()  # 调用父类的初始化方法
        self.num_freqs = num_freqs  # 编码频率数量
        self.d_in = d_in  # 输入维度
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)  # 生成频率序列，按指数增长
        self.d_out = self.num_freqs * 2 * d_in  # 编码后的输出维度，不包括原始输入部分
        self.include_input = include_input and self.d_in > 1  # 如果指定并且输入维度大于1，则包括原始输入
        if include_input:
            self.d_out += d_in  # 如果包括原始输入，则输出维度增加 d_in

        # 将频率序列重复两次，用于生成正弦和余弦分量
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)  # 重复每个频率两次，并调整形状
        )

        # 设置相位偏移，交替为0和π/2，使生成的正弦和余弦分量交替排列
        _phases = torch.zeros(2 * self.num_freqs)  # 初始化相位为0
        _phases[1::2] = np.pi * 0.5  # 每隔一个元素设置为 π/2
        self.register_buffer("_phases", _phases.view(1, -1, 1))  # 注册为不可训练的缓冲区

    def forward(self, x):
        """
        对输入应用位置编码
        :param x: 输入张量，形状为 (batch_size, self.d_in)
        :return: 编码后的张量，形状为 (batch_size, self.d_out)
        """
        with profiler.record_function("positional_enc"):  # 使用 profiler 记录操作时间
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)  # 在频率维度上重复输入
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))  # 计算正弦编码，addcmul 是 x + α * y 的快捷方式
            embed = embed.view(x.shape[0], -1)  # 展平编码结果
            if self.include_input:  # 如果包括原始输入
                embed = torch.cat((x, embed), dim=-1)  # 将原始输入拼接到编码结果上
            return embed  # 返回编码后的结果

    @classmethod
    def from_conf(cls, conf, d_in=3):
        """
        从配置字典创建位置编码模块
        :param conf: 配置字典，包含编码的超参数
        :param d_in: 输入维度
        :return: 实例化的 PositionalEncoding 模块
        """
        return cls(
            conf.get("num_freqs", 6),  # 从配置中获取 num_freqs，默认为6
            d_in,  # 输入维度
            conf.get("freq_factor", np.pi),  # 从配置中获取 freq_factor，默认为π
            conf.get("include_input", True),  # 从配置中获取 include_input，默认为True
        )
