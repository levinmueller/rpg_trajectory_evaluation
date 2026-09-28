#!/usr/bin/env python3
"""
Estimate an approximate WORLD-MAP transform from the beginning of a trajectory.

The groundtruth (expressed in WORLD) and the VIO estimate (expressed in MAP)
must track the same body frame and be sampled at the same timestamps. The
estimate is aligned to the groundtruth with a 4DOF (position + yaw) alignment
over the first frames that cover --alignment_distance meters of groundtruth
path, which yields T_WORLD_MAP such that:
    p_WORLD = T_WORLD_MAP * p_MAP
As a sanity check, the yaw obtained from the orientations of the first frame
(n=1) is reported next to the position-based yaw (n>1).
"""

import os
import argparse

import numpy as np
import yaml

import add_path
import align_utils as au
import transformations as tf
from trajectory_utils import get_distance_from_start

GT_FN = 'stamped_groundtruth.txt'
EST_FN = 'stamped_traj_estimate.txt'
OUT_FN = 'T_world_map.yaml'
TIMESTAMP_TOL_SEC = 1e-4


def load_synchronized(data_dir):
    """Load groundtruth and estimate (# timestamp tx ty tz qx qy qz qw)."""
    data_gt = np.loadtxt(os.path.join(data_dir, GT_FN), ndmin=2)
    data_es = np.loadtxt(os.path.join(data_dir, EST_FN), ndmin=2)
    if data_gt.shape != data_es.shape:
        raise ValueError(
            'Groundtruth ({0} rows) and estimate ({1} rows) have different '
            'sizes. They must be sampled at the same timestamps.'.format(
                data_gt.shape[0], data_es.shape[0]))
    max_dt = np.max(np.abs(data_gt[:, 0] - data_es[:, 0]))
    if max_dt > TIMESTAMP_TOL_SEC:
        raise ValueError(
            'Groundtruth and estimate timestamps differ by up to {0} s. '
            'They must be sampled at the same timestamps.'.format(max_dt))
    return data_gt, data_es


def n_frames_for_distance(p_gt, alignment_distance):
    """Number of frames needed to cover alignment_distance of GT path."""
    distances = get_distance_from_start(p_gt)
    if distances[-1] < alignment_distance:
        raise ValueError(
            'Groundtruth path ({0:.3f} m) is shorter than the alignment '
            'distance ({1:.3f} m).'.format(distances[-1], alignment_distance))
    idx = int(np.searchsorted(distances, alignment_distance))
    # posyaw with a single frame uses orientations, so use at least 2 frames
    return max(idx + 1, 2)


def yaw_deg(R):
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))


def wrap_deg(angle):
    return float((angle + 180.0) % 360.0 - 180.0)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate an approximate WORLD-MAP transform by aligning '
        'the beginning of the VIO trajectory (MAP) to the groundtruth (WORLD).')
    parser.add_argument(
        'data_dir', type=str,
        help='directory containing {0} and {1}'.format(GT_FN, EST_FN))
    parser.add_argument(
        '--alignment_distance', type=float, required=True,
        help='groundtruth path length in meters used for the alignment')
    parser.add_argument(
        '--skip_frames', type=int, default=0,
        help='number of initial frames to skip before the alignment window')
    args = parser.parse_args()

    assert args.alignment_distance > 0, 'alignment_distance must be positive'
    assert args.skip_frames >= 0, 'skip_frames must be non-negative'

    data_gt, data_es = load_synchronized(args.data_dir)
    data_gt = data_gt[args.skip_frames:]
    data_es = data_es[args.skip_frames:]
    if data_gt.shape[0] < 2:
        raise ValueError('Fewer than 2 frames left after skipping.')

    t_gt = data_gt[:, 0]
    p_gt, q_gt = data_gt[:, 1:4], data_gt[:, 4:8]
    p_es, q_es = data_es[:, 1:4], data_es[:, 4:8]

    n_frames = n_frames_for_distance(p_gt, args.alignment_distance)

    # position-based alignment over the window (n > 1)
    R, t = au.alignPositionYaw(p_es, p_gt, q_es, q_gt, n_frames)
    # orientation-based alignment on the first frame (n = 1), as a check
    R_ori, _ = au.alignPositionYaw(p_es, p_gt, q_es, q_gt, 1)

    T_world_map = np.identity(4)
    T_world_map[0:3, 0:3] = R
    T_world_map[0:3, 3] = t
    roll, pitch, yaw = np.degrees(tf.euler_from_matrix(R, 'sxyz'))

    yaw_pos = yaw_deg(R)
    yaw_ori = yaw_deg(R_ori)

    result = {
        'T_WORLD_MAP': {
            'description': 'p_WORLD = T_WORLD_MAP * p_MAP',
            'x': float(t[0]),
            'y': float(t[1]),
            'z': float(t[2]),
            'roll_deg': float(roll),
            'pitch_deg': float(pitch),
            'yaw_deg': float(yaw),
            'matrix': T_world_map.tolist(),
        },
        'yaw_check': {
            'yaw_deg_position_based': yaw_pos,
            'yaw_deg_orientation_based': yaw_ori,
            'difference_deg': wrap_deg(yaw_pos - yaw_ori),
        },
        'alignment': {
            'alignment_distance_m': float(args.alignment_distance),
            'skip_frames': args.skip_frames,
            'n_frames': n_frames,
            'start_timestamp': float(t_gt[0]),
            'end_timestamp': float(t_gt[n_frames - 1]),
        },
    }

    out_fn = os.path.join(args.data_dir, OUT_FN)
    with open(out_fn, 'w') as f:
        yaml.safe_dump(result, f, sort_keys=False, default_flow_style=False)

    print(yaml.safe_dump(result, sort_keys=False, default_flow_style=False))
    print('Saved to {0}'.format(out_fn))


if __name__ == '__main__':
    main()
