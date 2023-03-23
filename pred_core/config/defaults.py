from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 2
_C.DATA.DATASET_DIR = "/mnt/develop/PycharmProjects/blue_green_alage_prediction/pred_core/data"
_C.DATA.TEST_BATCH_SIZE = 2
_C.DATA.GRAPH_PKL_FILENAME = "/media/reid/ext_disk1/blue_alage/dushu/adj_mat.pkl"

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 200
_C.SOLVER.EPOCH = 0
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.EPSILON = 1e-8
_C.SOLVER.STEPS = [20, 30, 40, 50]
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.TEST_PERIOD = 50
# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = "/mnt/develop/PycharmProjects/blue_green_alage_prediction/models"
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = None
_C.MODEL.MAX_DIFFUSION_STEP = 2
_C.MODEL.CL_DECAY_STEPS = 1000
_C.MODEL.FILTER_TYPE = 'laplacian'
_C.MODEL.NUM_NODES = 888
_C.MODEL.NUM_RNN_LAYERS = 1
_C.MODEL.RNN_UNITS = 64
_C.MODEL.INPUT_DIM = 4
_C.MODEL.OUTPUT_DIM = 1
_C.MODEL.HORIZON = 1
_C.MODEL.SEQ_LEN = 5
_C.MODEL.CL_DECAY_STEPS = 1000
_C.MODEL.USE_CURRICULUM_LEARNING = False
