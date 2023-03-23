import argparse
import torch

from pred_core.config import cfg
from pred_core.modeling import build_prediction_model
from train_net import load_graph_data
from pred_core.data import make_data_loader
from pred_core.engine.trainer import evaluate
from pred_core.utils.comm import load_graph_data
from pred_core.utils.checkpoint import Checkpointer


def main():
    parser = argparse.ArgumentParser(description="blue green alage prediction test")
    parser.add_argument(
        "--config-file",
        default="/mnt/develop/PycharmProjects/blue_green_alage_prediction/pred_core/config/blue_green_alage.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--model_location",
        default="/mnt/develop/PycharmProjects/blue_green_alage_prediction/models/epo199.tar",
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

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    _, _, adj_mx = load_graph_data(args.graph_pkl_filename)

    model = build_prediction_model(adj_mx, cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    # model.to(device)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = Checkpointer(model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    data_loader = make_data_loader(
        cfg
    )
    mean_loss, dict_result = evaluate(model, data_loader, data_loader['scaler'], 'cuda')
    pass


if __name__ == '__main__':
    main()
