import pandas as pd
import matplotlib.pyplot as plt

path = "/media/reid/ext_disk1/blue_alage/dushu/krige_data/20210918.csv"

df = pd.read_csv(path)

plt.scatter(df['lon'], df['lat'], c=df['pc'], cmap='cool')
plt.colorbar()

# Set plot title and axis labels
plt.title('Scatter Plot with Color Depth')
plt.xlabel('x')
plt.ylabel('y')

# Display the plot
plt.show()