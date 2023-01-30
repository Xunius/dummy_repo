'''Example to reproduce a BisectingKMeans bug'''

import numpy as np
from sklearn.cluster import BisectingKMeans

npz = np.load('./bisectkmeans.npz')

km = BisectingKMeans(n_clusters=9, init='random', max_iter=400,
                     n_init=10, random_state=10,
                     bisecting_strategy='largest_cluster')
X = npz['data']
weights = npz['weights']
km.fit(X, None, sample_weight=weights)
