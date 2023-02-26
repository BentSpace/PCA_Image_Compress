import numpy as np

def compute_Z(X, centering=True, scaling=True):
    num_rows, num_cols = X.shape
    num_features = num_cols
    num_samples = num_rows
    if centering:
        Z = np.zeros((num_samples, num_features))
        mean = find_mean(X)
        i = 0
        while i < num_features:
            j = 0 
            while j < num_samples:
                Z[j, i] = X[j, i] - mean[i]
                j += 1
            i += 1
    else:
        Z = X
    if scaling:
        std = np.zeros(num_features)
        i = 0
        while i < num_features:
            std[i] = np.std(X[:, i])
            i += 1
            
        i = 0
        while i < num_features:
            j = 0 
            while j < num_samples:
                Z[j, i] = Z[j, i] / std[i]
                j += 1
            i += 1
        print ("std: \n", std)
    return Z
                
#return matrix of means for each feature / column in X
def find_mean(X):
    num_rows, num_cols = X.shape
    num_features = num_cols
    num_samples = num_rows
    mean = np.zeros(num_features)
    i = 0
    while i < num_features:
        j = 0
        sum = 0
        while j < num_samples:
            sum += X[j, i]
            j += 1
        mean[i] = sum / num_samples
        i += 1
    print("mean: \n", mean)
    return mean

def compute_covariance_matrix(Z):
    covar = np.matmul(np.transpose(Z), Z)
    print("covar: \n", covar)
        