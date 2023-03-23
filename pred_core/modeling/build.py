from .dcrnn_model import DCRNNModel


def build_prediction_model(adj_mx, cfg):
    return DCRNNModel(adj_mx, cfg)
