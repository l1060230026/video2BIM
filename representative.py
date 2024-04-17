import os
import shutil
import scripts.read_write_model as model
import scripts.read_write_fused_vis as vis
import numpy as np
from sklearn import decomposition
from sklearn.cluster import KMeans

if __name__ == "__main__":
    images = model.read_images_binary("dense/sparse/images.bin")
    cameras = model.read_cameras_binary("dense/sparse/cameras.bin")
    mesh_points = vis.read_fused("dense/fused.ply", "dense/fused.ply.vis")
    images = list(images.values())
    cameras = list(cameras.values())
    images_name = []
    for image in images:
        images_name.append(image.name)
    label = np.zeros((len(mesh_points), len(images)))
    for i in range(len(mesh_points)):
        pos = mesh_points[i].visible_image_idxs
        label[i][pos] = 1
    pca = decomposition.PCA()
    label = label.T
    rate = 0.1
    reduced_x = pca.fit_transform(label)
    n_clusters = round(len(images_name)*rate)
    estimator = KMeans(n_clusters=n_clusters,).fit(reduced_x)
    centers = estimator.labels_
    distance = estimator.fit_transform(reduced_x)
    idx = []

    for i in range(n_clusters):
        pos = np.where(centers==i)[0]
        min = 1000
        img_id = 0
        for j in pos:
            if np.min(distance[j]) < min:
                img_id = j
                min = np.min(distance[j])
        idx.append(img_id)
    if os.path.exists('knn' + str(rate)):
        shutil.rmtree('knn' + str(rate))
    os.mkdir('knn' + str(rate))
    for i in idx:
        shutil.copy('dense/images/'+images_name[i], 'knn' + str(rate) + '/' +images_name[i])