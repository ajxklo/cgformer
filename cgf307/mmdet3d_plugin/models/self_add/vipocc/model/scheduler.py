from torch import optim  # 导入 PyTorch 的优化器模块
from torch.optim.lr_scheduler import StepLR  # 导入 StepLR 学习率调度器


class FixLR(optim.lr_scheduler._LRScheduler):
    """
    固定学习率调度器，继承自 PyTorch 的 _LRScheduler 基类。
    该调度器用于保持学习率不变，返回与基础学习率相同的学习率值。
    """

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        """
        初始化方法，继承了 PyTorch 的调度器初始化方法。

        :param optimizer: 用于训练的优化器
        :param last_epoch: 上一个 epoch 的索引，默认为 -1
        :param verbose: 是否打印调度器信息，默认为 False
        """
        super(FixLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        获取当前的学习率，返回固定的学习率。

        :return: 返回一个包含所有参数学习率的列表，值为基础学习率
        """
        return [base_lr for base_lr in self.base_lrs]  # 固定学习率不变化，直接返回基础学习率列表

    def _get_closed_form_lr(self):
        """
        获取闭式的学习率，这里也返回固定的学习率。

        :return: 返回一个包含所有参数学习率的列表，值为基础学习率
        """
        return [base_lr for base_lr in self.base_lrs]  # 同样是固定学习率，返回基础学习率列表


def make_scheduler(config, optim):
    """
    根据配置文件创建并返回学习率调度器。

    :param config: 配置字典，包含调度器类型及其参数
    :param optim: 优化器，用于调整学习率
    :return: 返回相应的学习率调度器实例
    """
    # 获取配置中的调度器类型，默认为 'fix'
    type = config.get("type", "fix")

    # 如果选择的是固定学习率调度器
    if type == "fix":
        scheduler = FixLR(optim)  # 创建 FixLR 调度器实例
        return scheduler

    # 如果选择的是 StepLR 学习率调度器
    elif type == "step":
        scheduler = StepLR(
            optim,  # 传入优化器
            config["step_size"],  # 获取配置中的步长
            config["gamma"]  # 获取配置中的 gamma 值，用于衰减学习率
        )
        return scheduler

    # 如果配置中的调度器类型不支持，抛出异常
    else:
        raise NotImplementedError(f"Unknown learning rate scheduler type: {type}")
