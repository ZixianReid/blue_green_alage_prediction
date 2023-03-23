import torch


def make_optimizer(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, eps=cfg.SOLVER.EPSILON)


def make_lr_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPS,
                                                            gamma=cfg.SOLVER.GAMMA)
