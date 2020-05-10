import pandas as pd
import scipy
import scipy.spatial.distance as dist
from scipy.stats import multivariate_normal, pearsonr, f_oneway


d = pd.read_csv("bads_maps.csv", sep = "\t")

#Box ploting
d.boxplot(column=['A', 'B', 'C', 'D'], grid=False)
#Loading data file
