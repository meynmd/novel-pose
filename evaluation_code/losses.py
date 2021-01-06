import tensorflow as tf
import numpy as np
import scipy.io as sio

from config import opts
import procrustes



def unit_norm(mat, dim=1):
    norm = (np.sqrt(np.sum(mat ** 2, dim)) + 1e-9)
    norm = np.expand_dims(norm, dim)
    mat = mat / norm
    return mat

def normalize_3d_pose(pred_3d):
    pelvis = pred_3d[:,0,:]
    bone_lengths = [138.1844999592938,
     138.18484529064688,
     494.82600130000327,
     166.3306689820011,
     283.17116499566146,
     248.31486052389192,
     166.33057668390387,
     283.17231241065934,
     248.3112906273091,
     185.48292281492064,
     461.9404741879626,
     460.22362705971585,
     163.88421144896782,
     461.94021690365923,
     460.2238050572889,
     233.46550818986296]
    rhip = unit_norm(pred_3d[:,9,:], dim=0) * bone_lengths[0] + pelvis
    lhip = unit_norm(pred_3d[:,13,:], dim=0) * bone_lengths[1] + pelvis
    neck = unit_norm(pred_3d[:,1,:], dim=0) * bone_lengths[2] + pelvis
    rs = unit_norm(pred_3d[:,2,:]-pred_3d[:,1,:], dim=0) * bone_lengths[3] + neck
    re = unit_norm(pred_3d[:,3,:]-pred_3d[:,2,:], dim=0) * bone_lengths[4] + rs
    rh = unit_norm(pred_3d[:,4,:]-pred_3d[:,3,:], dim=0) * bone_lengths[5] + re
    ls = unit_norm(pred_3d[:,5,:]-pred_3d[:,1,:], dim=0) * bone_lengths[6] + neck
    le = unit_norm(pred_3d[:,6,:]-pred_3d[:,5,:], dim=0) *bone_lengths[7] + ls
    lh = unit_norm(pred_3d[:,7,:]-pred_3d[:,6,:], dim=0) * bone_lengths[8] + le
    head = unit_norm(pred_3d[:,8,:]-pred_3d[:,1,:], dim=0) * bone_lengths[9] + neck
    rk = unit_norm(pred_3d[:,10,:]-pred_3d[:,9,:], dim=0) * bone_lengths[10] + rhip
    ra = unit_norm(pred_3d[:,11,:]-pred_3d[:,10,:], dim=0) * bone_lengths[11] + rk
    rf = unit_norm(pred_3d[:,12,:]-pred_3d[:,11,:], dim=0) *bone_lengths[12] + ra
    lk = unit_norm(pred_3d[:,14,:]-pred_3d[:,13,:], dim=0) * bone_lengths[13] + lhip
    la = unit_norm(pred_3d[:,15,:]-pred_3d[:,14,:], dim=0) * bone_lengths[14] + lk
    lf = unit_norm(pred_3d[:,16,:]-pred_3d[:,15,:], dim=0) *bone_lengths[15] + la

    skeleton = np.concatenate([np.expand_dims(pelvis, axis=1), np.expand_dims(neck, axis=1), 
                           np.expand_dims(rs, axis=1), np.expand_dims(re, axis=1),
                           np.expand_dims(rh, axis=1), np.expand_dims(ls, axis=1),
                           np.expand_dims(le, axis=1), np.expand_dims(lh, axis=1),
                           np.expand_dims(head, axis=1), np.expand_dims(rhip, axis=1),
                           np.expand_dims(rk, axis=1), np.expand_dims(ra, axis=1),
                           np.expand_dims(rf, axis=1), np.expand_dims(lhip, axis=1), 
                           np.expand_dims(lk, axis=1), np.expand_dims(la, axis=1), 
                           np.expand_dims(lf, axis=1)], axis=1)
    return skeleton

def compute_mpjpe_summary(gt_pose, pred_pose):
    gt_pose = normalize_3d_pose(gt_pose)
    pred_pose = normalize_3d_pose(pred_pose)

    pa_pose = pred_pose.copy()
    try:
        if opts['procustes']:
            for i in range(opts['batch_size']):
                gt = gt_pose[i]
                out = pred_pose[i]
                _, Z, T, b, c = procrustes.compute_similarity_transform(gt,out,compute_optimal_scale=True)
                out = (b*out.dot(T))+c
                pa_pose[i, :, :] = out
    except:
        #in case of an error, dump the pose batch and exit
        sio.savemat("poses.mat", {"gt" : gt_pose, "pred" : pred_pose})
        print("error in mpjpe")
        exit()
    return np.mean(np.sqrt(np.sum((pa_pose - gt_pose)**2, axis = 2)))


