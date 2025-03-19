from .mlp import ImplicitNet
from .resnetfc import ResnetFC


def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet":  # for mlp_coarse
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "empty" and allow_empty:  # for mlp_fine
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net
