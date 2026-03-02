import os
import numpy as np
import cv2
from tqdm import tqdm

query_dir = './query'
gallery_dir = './gallery'

with open('./rank_list_orb.txt', 'w') as rank_list:
    # Create SIFT feature detector
    # sift = cv2.SIFT_create()
    # Create ORB feature detector
    orb = cv2.ORB_create()

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)

    search_params = dict(checks=50)
    # FLANN-based matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for query_img_file in tqdm(os.listdir(query_dir)):
        query_img_path = os.path.join(query_dir, query_img_file)
        query_img = cv2.cvtColor(cv2.imread(query_img_path), cv2.COLOR_BGR2GRAY)
        # kp_query, des_query = sift.detectAndCompute(query_img, None)
        kp_query, des_query = orb.detectAndCompute(query_img, None)

        idx_dis = {}
        for gallery_img_file in os.listdir(gallery_dir):
            gallery_idx = gallery_img_file.split('.')[0]
            gallery_img_path = os.path.join(gallery_dir, gallery_img_file)
            gallery_img = cv2.cvtColor(cv2.imread(gallery_img_path), cv2.COLOR_BGR2GRAY)
            # kp_gallery, des_gallery = sift.detectAndCompute(gallery_img, None)
            kp_gallery, des_gallery = orb.detectAndCompute(gallery_img, None)

            # K-nearest neighbors matching
            matches = flann.knnMatch(des_query, des_gallery, k=2)
            # Apply ratio test as per Lowe's paper to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # Compute the homography matrix using RANSAC
            if len(good_matches) > 4:
                src_pts = np.float32([kp_query[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_gallery[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = mask.ravel().tolist()
                num_inliers = inliers.count(1)
                idx_dis[gallery_idx] = num_inliers
            else:
                idx_dis[gallery_idx] = 0

        # Retrieve the top ten
        rank = sorted(idx_dis, key=idx_dis.get, reverse=True)[:10]
        rank_list.write('Q{}: '.format(query_img_file.split('.')[0]) + ' '.join(rank) + '\n')
