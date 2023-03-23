from torch.utils.collect_env import get_pretty_env_info


def collect_env_info():
    env_str = get_pretty_env_info()
    return env_str
