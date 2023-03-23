# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# df = pd.read_csv("/media/reid/ext_disk1/blue_alage/dushu/krige_data/20210901.csv")
#
# pred = np.load("/mnt/develop/PycharmProjects/blue_green_alage_prediction/output/pred.npy")
#
# # pred_plot = pred[12]
#
# truth = np.load('/mnt/develop/PycharmProjects/blue_green_alage_prediction/output/truth.npy')
#
# # truth_plot = truth[12]
#
# plt.scatter(df['lon'], df['lat'], c=truth_plot)
# plt.colorbar()
# plt.show()