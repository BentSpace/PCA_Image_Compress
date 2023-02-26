import pca
import numpy as np

#test PCA
#X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
#X = np.array([[-2,-2],[-1,1],[2,-2],[1,1]])
#X = np.array([[-2,-6],[5,1],[5,2],[-1,-7]])
#X = np.array([[0,6],[1, 5],[0,-9],[-1,-7]])
X = np.array([[-1, -1],[-1,1],[1,-1],[1,1],[2,1]])

Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z)
L, PCS = pca.find_pcs(COV)
Z_star = pca.project_data(Z, PCS, L, 1, 0)


#test compression  using PCA
import compress

X = compress.load_data('Data/Train/')
compress.compress_images(X,100)
