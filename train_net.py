import argparse

import torch

from pred_core.config import cfg
from pred_core.utils.logger import setup_logger
from pred_core.utils.comm import get_rank
from pred_core.utils.collect_env import collect_env_info
from pred_core.data import make_data_loader
from pred_core.engine.trainer import do_train
from pred_core.modeling import build_prediction_model
from pred_core.solver.build import make_optimizer
from pred_core.solver.build import make_lr_scheduler
from pred_core.utils.comm import load_graph_data
from pred_core.utils.checkpoint import Checkpointer


def train(cfg, adj_mx):
    model = build_prediction_model(adj_mx, cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    #
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkPointer = Checkpointer(model, optimizer, scheduler, output_dir, save_to_disk)
    arguments = {}
    arguments['iteration'] = 0
    arguments['num_epoch'] = cfg.SOLVER.EPOCH
    arguments['epochs'] = cfg.SOLVER.EPOCHS
    arguments['test_period'] = cfg.SOLVER.TEST_PERIOD

    data_loader = make_data_loader(
        cfg
    )

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        device,
        checkPointer,
        arguments
    )


def main():
    parser = argparse.ArgumentParser(description="blue green alage prediction")
    parser.add_argument(
        "--config-file",
        default="/mnt/develop/PycharmProjects/blue_green_alage_prediction/pred_core/config/blue_green_alage.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        '--graph_pkl_filename',
        default="/media/reid/ext_disk1/blue_alage/dushu/adj_mat.pkl",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR

    logger = setup_logger("pred_core", output_dir, get_rank())
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    _, _, adj_mx = load_graph_data(args.graph_pkl_filename)

    train(cfg, adj_mx)


if __name__ == '__main__':
    main()
