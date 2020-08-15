"""
Unsupervised learning: Hierarchical clustering: Bottom up approach.
Experimental code to create clusters and their dendrograms. Assumes data is already prepared in the tabular format.

Performance metric: Silhouette score & Cophenet distance.
"""

import os
import sqlalchemy
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc

# Feat. specific imports
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Cluster closeness/score measure
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

%matplotlib inline
warnings.filterwarnings('ignore')

# Setup for saving dendograms to pdf
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('dendo_vis.pdf')


def measure_cluster_accuracy(hier, data):
    """
    Generate score for Hierarchy clusters.
    The closer the value is to 1, the better the clustering preserves the original distances
    """
    score, coph_dists = cophenet(hier, pdist(data))
    print('\n', 'Cophenet distance for ', cat, '==> ', round(score, 2))

def save_dendogram(hier, le_name_mapping, cat):
    """
    Plot & save dendogram, to identify clusters for stores.
    """
    plt.figure(figsize=(6, 4))
    plt.title("Walmart Stores Dendograms for '{cat}'".format(cat=cat))
    dend = shc.dendrogram(hier, orientation='right', labels=le_name_mapping)
    pp.savefig(plt.savefig('temp.jpg'), bbox_inches = 'tight', frameon=True)

def get_ideal_clustersize_category(data, cat):
    clusters = [i+2 for i in range(data.shape[1]-2)]
    print('\n', cat, '  ', '*'*10)

    clusters_sorted_by_values = dict()
    for cluster in clusters:
        model = AgglomerativeClustering(linkage='average', n_clusters=cluster, affinity='euclidean')
        cluster_labels = model.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        clusters_sorted_by_values[cluster] =  silhouette_avg

    for key, value in sorted(clusters_sorted_by_values.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        print("For cluster =", key, ". The average silhouette_score is :", value)


# Pull data # --------
engine = sqlalchemy.create_engine('oracle+cx_oracle://[hitesh]:@localhost')
table_data =  engine.execute('select * from all_raw_data')
raw_data = pd.DataFrame(table_data.fetchall())
raw_data.columns = table_data.keys()
print(raw_data['pct_overlap'].describe())

for cat in np.sort(raw_data['item_group'].unique()):
	"""
	Run for each category
	"""
    # Format data
    data = raw_data[raw_data['item_group'].eq(cat)].copy()
    le = preprocessing.LabelEncoder()
    le.fit(data['item_group'].unique())
    data.drop(columns='item_group', axis=1, inplace=True)

    le.fit(data['first_group'].unique())
    data['first_group'] = le.transform(data['first_group'])
    data['second_group'] = le.transform(data['second_group'])

    le_name_mapping = list(zip(le.classes_, le.transform(le.classes_)))
    data = data.pivot(index='first_group', columns='second_group', values='pct_overlap')

    # replace NaNs for self group joins
    data.fillna(100, inplace=True)

    hier = shc.linkage(data, method='average')
    measure_cluster_accuracy(hier, data)
    save_dendogram(hier, le_name_mapping, cat)
    get_ideal_clustersize_category(data, cat)


pp.close()
os.remove('temp.jpg')
print('Setup complete. Check the visualisations in PDF!')

