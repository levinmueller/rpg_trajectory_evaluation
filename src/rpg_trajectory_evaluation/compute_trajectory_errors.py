#!/usr/bin/env python2

import os
import numpy as np

import trajectory_utils as tu
import transformations as tf


def compute_relative_error(p_es, q_es, p_gt, q_gt, T_cm, dist, max_dist_diff,
                           accum_distances=[],
                           scale=1.0):

    if len(accum_distances) == 0:
        accum_distances = tu.get_distance_from_start(p_gt)
    comparisons = tu.compute_comparison_indices_length(
        accum_distances, dist, max_dist_diff)

    n_samples = len(comparisons)
    print('number of samples = {0} '.format(n_samples))
    if n_samples < 2:
        print("Too few samples! Will not compute.")
        return {k: np.array([]) for k in ['trans', 'trans_perc', 'trans_xy', 'trans_xy_perc', 'trans_z', 'trans_z_perc', 'yaw', 'pitch', 'roll', 'gravity', 'rot', 'rot_deg_per_m', 'rot_yaw_per_m', 'rot_pitch_per_m', 'rot_roll_per_m']}

    T_mc = np.linalg.inv(T_cm)
    errors = []
    for idx, c in enumerate(comparisons):
        if not c == -1:
            T_c1 = tu.get_rigid_body_trafo(q_es[idx, :], p_es[idx, :])
            T_c2 = tu.get_rigid_body_trafo(q_es[c, :], p_es[c, :])
            T_c1_c2 = np.dot(np.linalg.inv(T_c1), T_c2)
            T_c1_c2[:3, 3] *= scale

            T_m1 = tu.get_rigid_body_trafo(q_gt[idx, :], p_gt[idx, :])
            T_m2 = tu.get_rigid_body_trafo(q_gt[c, :], p_gt[c, :])
            T_m1_m2 = np.dot(np.linalg.inv(T_m1), T_m2)

            T_m1_m2_in_c1 = np.dot(T_cm, np.dot(T_m1_m2, T_mc))
            T_error_in_c2 = np.dot(np.linalg.inv(T_m1_m2_in_c1), T_c1_c2)
            T_c2_rot = np.eye(4)
            T_c2_rot[0:3, 0:3] = T_c2[0:3, 0:3]
            T_error_in_w = np.dot(T_c2_rot, np.dot(
                T_error_in_c2, np.linalg.inv(T_c2_rot)))
            errors.append(T_error_in_w)

    error_trans_norm = []
    error_trans_perc = []
    error_trans_xy = []
    error_trans_xy_perc = []
    error_trans_z = []
    error_trans_z_perc = []
    error_yaw = []
    error_pitch = []
    error_roll = []
    error_gravity = []
    e_rot = []
    e_rot_deg_per_m = []
    e_yaw_deg_per_m = []
    e_pitch_deg_per_m = []
    e_roll_deg_per_m = []

    for e in errors:
        # translation error
            # full 3D translation error
        tn = np.linalg.norm(e[0:3, 3])
        error_trans_norm.append(tn)
        error_trans_perc.append(tn / dist * 100)
            # 2D translation error
        tn_xy = np.linalg.norm(e[0:2, 3])
        error_trans_xy.append(tn_xy)
        error_trans_xy_perc.append(tn_xy / dist * 100)
            # z translation error
        tn_z = abs(e[2, 3])
        error_trans_z.append(tn_z)
        error_trans_z_perc.append(tn_z / dist * 100)

        # orientation error
            # yaw, pitch, roll angles from rotation matrix
        ypr_angles = tf.euler_from_matrix(e, 'rzyx')
            # rotation error in degrees
        e_rot.append(tu.compute_angle(e))
            # separate yaw, pitch, roll errors in degrees
        error_yaw.append(abs(ypr_angles[0])*180.0/np.pi)
        error_pitch.append(abs(ypr_angles[1])*180.0/np.pi)
        error_roll.append(abs(ypr_angles[2])*180.0/np.pi)
            # gravity error in degrees (pitch and roll)
        error_gravity.append(np.sqrt(ypr_angles[1]**2+ypr_angles[2]**2)*180.0/np.pi)
            # rotation error in degrees per meter
        e_rot_deg_per_m.append(e_rot[-1] / dist)
        e_yaw_deg_per_m.append(error_yaw[-1] / dist)
        e_pitch_deg_per_m.append(error_pitch[-1] / dist)
        e_roll_deg_per_m.append(error_roll[-1] / dist)

    return {
        'errors': errors,
        'trans': np.array(error_trans_norm),
        'trans_perc': np.array(error_trans_perc),
        'trans_xy': np.array(error_trans_xy),
        'trans_xy_perc': np.array(error_trans_xy_perc),
        'trans_z': np.array(error_trans_z),
        'trans_z_perc': np.array(error_trans_z_perc),
        'yaw': np.array(error_yaw),
        'pitch': np.array(error_pitch),
        'roll': np.array(error_roll),
        'gravity': np.array(error_gravity),
        'rot': np.array(e_rot),
        'rot_deg_per_m': np.array(e_rot_deg_per_m),
        'rot_yaw_per_m': np.array(e_yaw_deg_per_m),
        'rot_pitch_per_m': np.array(e_pitch_deg_per_m),
        'rot_roll_per_m': np.array(e_roll_deg_per_m),
    }


def compute_absolute_error(p_es_aligned, q_es_aligned, p_gt, q_gt):
    # translation error
    e_trans_vec = (p_gt-p_es_aligned)
    e_trans = np.sqrt(np.sum(e_trans_vec**2, 1))
        # 2D and z translation error
    e_trans_xy = np.sqrt(np.sum(e_trans_vec[:, :2]**2, 1))
    e_trans_z = np.abs(e_trans_vec[:, 2])

    # orientation error
    e_rot = np.zeros((len(e_trans,)))
    e_ypr = np.zeros(np.shape(p_es_aligned))
    for i in range(np.shape(p_es_aligned)[0]):
        R_we = tf.matrix_from_quaternion(q_es_aligned[i, :])
        R_wg = tf.matrix_from_quaternion(q_gt[i, :])
        e_R = np.dot(R_we, np.linalg.inv(R_wg))
        e_ypr[i, :] = tf.euler_from_matrix(e_R, 'rzyx')
        e_rot[i] = np.rad2deg(np.linalg.norm(tf.logmap_so3(e_R[:3, :3])))
        # separate yaw, pitch, roll errors in degrees
    e_yaw = np.abs(e_ypr[:, 0]) * 180.0 / np.pi
    e_pitch = np.abs(e_ypr[:, 1]) * 180.0 / np.pi
    e_roll = np.abs(e_ypr[:, 2]) * 180.0 / np.pi

    # scale drift
    motion_gt = np.diff(p_gt, 0)
    motion_es = np.diff(p_es_aligned, 0)
    dist_gt = np.sqrt(np.sum(np.multiply(motion_gt, motion_gt), 1))
    dist_es = np.sqrt(np.sum(np.multiply(motion_es, motion_es), 1))
    e_scale_perc = np.abs((np.divide(dist_es, dist_gt)-1.0) * 100)

    return {
        'trans': e_trans,
        'trans_vec': e_trans_vec,
        'trans_xy': e_trans_xy,
        'trans_z': e_trans_z,
        'rot': e_rot,
        'ypr': e_ypr,
        'yaw': e_yaw,
        'pitch': e_pitch,
        'roll': e_roll,
        'scale_perc': e_scale_perc,
    }
