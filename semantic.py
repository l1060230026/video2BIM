import copy
import os

import cv2

import scripts.read_write_model as model
import scripts.read_write_fused_vis as vis
import glob
import imgviz
import numpy as np
from pyntcloud import PyntCloud
from numba import cuda
import cupy as cp
from scipy.spatial import KDTree
import math
from scipy import stats
from sklearn.cluster import DBSCAN


@cuda.jit
def process_gpu(xy, count, img, colors):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    if pos < count.shape[1]:
        for i in range(max(0, xy[0][pos]-1), min(img.shape[0], xy[0][pos]+2), 1):
            for j in range(max(0, xy[1][pos]-1), min(img.shape[1], xy[1][pos]+2), 1):
                for k in range(len(colors)):
                    if (colors[k][0] == img[i][j][0]) & (colors[k][1] == img[i][j][1]) & (
                            colors[k][2] == img[i][j][2]):
                        count[k][pos] += 1
                        break


if __name__ == "__main__":
    images = model.read_images_binary("dense/sparse/images.bin")
    cameras = model.read_cameras_binary("dense/sparse/cameras.bin")
    images = list(images.values())
    cameras = list(cameras.values())
    fold = "knn0.1"
    with open("labels.txt") as f:
        label_txt = f.readlines()
    images_name = []
    for image in images:
        images_name.append(image.name)
    labels = glob.glob(fold + "/*.png")

    labels_cameras = []
    for i in range(len(labels)):
        labels[i] = labels[i].split("\\")[1][:-3] + 'jpg'
        labels[i] = images_name.index(labels[i])
        params = cameras[labels[i]].params
        intrinsic = np.array([
            [params[0], 0, params[2]],
            [0, params[1], params[3]],
            [0, 0, 1]
        ])
        extrinsic = np.hstack((images[labels[i]].qvec2rotmat(), images[labels[i]].tvec.reshape(-1, 1)))
        labels_cameras.append(np.dot(intrinsic, extrinsic))
    colormap = imgviz.label_colormap()
    mesh_points = vis.read_fused("dense/fused.ply", "dense/fused.ply.vis")
    filters = [[] for _ in labels]
    filters_mesh = [[] for _ in labels]
    for i in range(len(mesh_points)):
        for j in range(len(labels)):
            if labels[j] in mesh_points[i].visible_image_idxs:
                filters[j].append(mesh_points[i].position)
                filters_mesh[j].append(i)
    filters_mesh = [np.array(_) for _ in filters_mesh]
    total = np.zeros((len(mesh_points))).astype(np.int_)
    total_count = 2
    for i in range(len(filters)):
        filters[i] = np.vstack((np.array(filters[i]).T, np.ones(len(filters[i]))))
        xyz = np.dot(labels_cameras[i], filters[i])
        xy = xyz[:2, :] / xyz[2, :]
        xy = xy.astype(np.int32)
        xy = np.vstack((xy[1], xy[0]))
        count = cuda.to_device(np.zeros((len(label_txt), xy.shape[1])))
        dxy = cuda.to_device(xy)
        img = cv2.cvtColor(cv2.imread(os.path.join(fold, images_name[labels[i]][:-3] + 'png')), cv2.COLOR_BGR2RGB)
        dImg = cuda.to_device(img)
        dColormap = cuda.to_device(colormap[:len(label_txt), :])

        threadsperblock = 256
        blockspergrid = (xy.shape[1] + (threadsperblock - 1)) // threadsperblock

        cuda.synchronize()
        process_gpu[blockspergrid, threadsperblock](dxy, count, dImg, dColormap)
        cuda.synchronize()
        count = count.copy_to_host()
        index = np.argmax(count, axis=0)
        index += 1
        for j in range(1, len(label_txt)+1):
            if j == 1:
                total[np.squeeze(np.where(total[filters_mesh[i][np.squeeze(np.where(index == j))]] == 0))] = j
                continue
            if len(np.squeeze(np.where(index == j))) == 0:
                continue
            if np.vstack((total[filters_mesh[i][np.squeeze(np.where(index == j))]]==0,
                          total[filters_mesh[i][np.squeeze(np.where(index == j))]]==1)).any(axis=0).all():
                total[filters_mesh[i][np.squeeze(np.where(index == j))]] = total_count
                total_count += 1
            else:
                if np.max(np.bincount(total[filters_mesh[i][np.squeeze(np.where(index == j))]])[2:]) >= 10:
                    try:
                        p1 = np.where(np.bincount(total[filters_mesh[i][np.squeeze(np.where(index == j))]]) >= 10)[0][-1]
                    except:
                        print('a')
                    for k in np.where(np.bincount(total[filters_mesh[i][np.squeeze(np.where(index == j))]]) >= 10)[0]:
                        if (k != 0) and (k != 1):
                            total[np.squeeze(np.where(total == k))] = p1
                    total[filters_mesh[i][np.squeeze(np.where(index == j))]] = p1
                else:
                    total[filters_mesh[i][np.squeeze(np.where(index == j))]] = total_count
                    total_count += 1
    final_count = 0
    temps = []
    for i in range(2, total_count):
        temp = []
        index = np.squeeze(np.where(total == i))
        if len(index) == 0:
            continue
        else:
            # if len(index) < 50:
            #     for j in index:
            #         mesh_points[j].color[0], mesh_points[j].color[1], mesh_points[j].color[2] = \
            #             colormap[0]
            final_count += 1
            for j in index:
                mesh_points[j].color[0], mesh_points[j].color[1], mesh_points[j].color[2] = \
                    colormap[final_count]
                temp.append(mesh_points[j])
            temps.append(temp)
            vis.write_fused_ply(temp, "points/{0}.ply".format(str(final_count),))

    #knn
    no_label = np.squeeze(np.where(total == 0))
    no_xyz = []
    label = np.squeeze(np.where(total != 0))
    xyz = []
    for i in no_label:
        no_xyz.append(mesh_points[i].position)
    for i in label:
        xyz.append(mesh_points[i].position)
    no_xyz = np.array(no_xyz)
    xyz = np.array(xyz)
    kdtree = KDTree(xyz)
    query = kdtree.query(no_xyz, 5)[1]
    result = stats.mode(total[label[query]], axis=1)[0].reshape(-1,)
    total[no_label] = result

    temp = []
    for i in range(2, total_count):
        index = np.squeeze(np.where(total == i))
        if len(index) == 0:
            continue
        else:
            final_count += 1
            for j in index:
                mesh_points[j].color[0], mesh_points[j].color[1], mesh_points[j].color[2] = \
                    colormap[final_count]
                temp.append(mesh_points[j])
    vis.write_fused_ply(temp, "{0}.ply".format('final_knn', ))

    temps[0].extend(temps[1])
    temps.pop(1)

    points = []
    for i in range(len(temps)):
        xyz = []
        for x in temps[i]:
            xyz.append(x.position)
        xyz = np.array(xyz)
        db = DBSCAN(eps=0.05, min_samples=50, metric='euclidean')
        y_db = db.fit_predict(xyz)
        index = np.squeeze(np.where(y_db == 0))
        for j in index:
            temps[i][j].color[0], temps[i][j].color[1], temps[i][j].color[2] = \
                colormap[i+1]
            points.append(temps[i][j])
        vis.write_fused_ply(points, "points/{0}.ply".format(i,))