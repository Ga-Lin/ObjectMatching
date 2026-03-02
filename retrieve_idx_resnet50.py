import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query_feature, feature):
  query_feature = query_feature.reshape(query_feature.shape[0], -1)
  feature = feature.reshape(feature.shape[0], -1)
  similarity = cosine_similarity(query_feature, feature)
  similarity = np.squeeze(similarity)

  return similarity

def retrieve_idx(query_feature_path, gallery_feature_dir):
  query_feature = np.load(query_feature_path)

  idx_sim = {}
  for feature_file in os.listdir(gallery_feature_dir):
    gallery_idx = feature_file.split('.')[0]
    feature = np.load(os.path.join(gallery_feature_dir, feature_file))

    similarity = []
    # For a query image
    # Different feature with different bounding box
    for query_sub_feature in query_feature:
      similarity.append(compute_similarity(query_sub_feature, feature))

    # Select the most significant feature represented by similarity
    idx_sim[gallery_idx] = max(similarity)

  # Retrieve the top ten
  return sorted(idx_sim, key=idx_sim.get, reverse=True)[:10]


query_feature_dir = './query_features'
gallery_feature_dir = './gallery_features'

with open('./rank_list_resnet50.txt', 'w') as rank_list:
  for query_feature_file in tqdm(os.listdir(query_feature_dir)):
    query_feature_path = os.path.join(query_feature_dir, query_feature_file)
    rank = retrieve_idx(query_feature_path, gallery_feature_dir)
    rank_list.write('Q{}: '.format(query_feature_file.split('.')[0]) + ' '.join(rank) + '\n')