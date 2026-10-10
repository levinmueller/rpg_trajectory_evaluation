#!/usr/bin/env python2
"""
@author: Christian Forster
"""

import os
import numpy as np
import transformations as tf


def get_rigid_body_trafo(quat, trans):
    T = tf.quaternion_matrix(quat)
    T[0:3, 3] = trans
    return T


def get_distance_from_start(gt_translation):
    distances = np.diff(gt_translation[:, 0:3], axis=0)
    distances = np.sqrt(np.sum(np.multiply(distances, distances), 1))
    distances = np.cumsum(distances)
    distances = np.concatenate(([0], distances))
    return distances


def get_strided_start_indices(distances, stride):
    """
    Indices of the samples closest to the path distances 0, stride,
    2*stride, ... along the (monotonic) accumulated distances.
    """
    start_dists = np.arange(0.0, distances[-1] + 0.5 * stride, stride)
    upper = np.clip(np.searchsorted(distances, start_dists), 0,
                    len(distances) - 1)
    lower = np.clip(upper - 1, 0, len(distances) - 1)
    closer_lower = np.abs(distances[lower] - start_dists) <=\
        np.abs(distances[upper] - start_dists)
    start_indices = np.where(closer_lower, lower, upper)
    return np.unique(start_indices).tolist()


def compute_comparison_indices_length(distances, dist, max_dist_diff,
                                      overlap=None):
    """
    Returns (start, end) index pairs of the sub-trajectories: for each start
    sample, the end sample whose accumulated distance is closest to
    start distance + dist, within max_dist_diff and further along the path
    than the start. Starts without such an end sample are skipped.
    overlap=None uses every sample as a start. Otherwise the starts are
    spaced by (1 - overlap) * dist along the path, so that consecutive
    sub-trajectories overlap by the fraction overlap.
    """
    max_idx = len(distances)
    if overlap is None:
        start_indices = range(max_idx)
    else:
        start_indices = get_strided_start_indices(
            distances, (1.0 - overlap) * dist)
    comparisons = []
    for idx in start_indices:
        d = distances[idx]
        best_idx = -1
        error = max_dist_diff
        for i in range(idx, max_idx):
            # distances[i] > d: no zero-length segments if max_dist_diff >= dist
            if distances[i] > d and np.abs(distances[i]-(d+dist)) < error:
                best_idx = i
                error = np.abs(distances[i] - (d+dist))
        if best_idx != -1:
            comparisons.append((idx, best_idx))
    return comparisons


def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    return np.arccos(
        min(1, max(-1, (np.trace(transform[0:3, 0:3]) - 1)/2)))*180.0/np.pi
