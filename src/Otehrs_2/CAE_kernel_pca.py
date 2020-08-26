print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import mne

from sklearn.decomposition import PCA, KernelPCA
from mne import read_evokeds


def read_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    matrix = []
    for line in lines:
        res =[]
        temp = line.split(' ')
        for num in temp:
            if num:
                res.append(float(num))
        matrix.append(res)

    return np.asarray(matrix)

def read_pos_data(path=None):
    if path is None:
        path = 'Cap63.locs'
    with open(path, 'r') as f:
        lines = f.readlines()
    res = []
    for line in lines:
        temp = line.split('\t')
        row = []
        if len(temp) >1:
            row.append(float(temp[1].strip()))
            row.append(float(temp[2].strip()))
        if row:
            res.append(row)
    return np.asarray(res)

data_folder = "C:\\Users\\Lab\\.PyCharmCE2019.1\\config\\scratches\\"
t_path = data_folder + "test.mul"
# print(t_path.format())
X1 = read_data(t_path)
print("The shape of the original matrix is {}".format(X1.shape))

pca = PCA(n_components=2)
TranX_pca = pca.fit_transform(X1)
# kpca = KernelPCA(n_components=2, kernel ="poly", fit_inverse_transform=True)
# X_kpca = kpca.fit_transform(X1)
# X_back = kpca.inverse_transform(X_kpca)


print(pca.explained_variance_)
print(pca.components_.shape)
print("The shape of the projected matrix is {}".format(TranX_pca.shape))
t_pos = read_pos_data()
# print(t_pos.shape)
n= len(t_pos[:,1])
for i in range(n):
    t_pos[i, 1] = int(t_pos[i,1]*100)


t_pca = np.transpose(pca.components_)
# print(t_pca.shape)
# print("the length of data {} == the length of position array {}".format(len(t_pca), len(t_pos)))

# mne.viz.plot_topomap(t_pca, t_pos[0:62, :])






