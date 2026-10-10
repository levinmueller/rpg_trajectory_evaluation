# Error metrics

This page describes every error metric computed by `analyze_trajectory_single.py`. The code is in `src/rpg_trajectory_evaluation/compute_trajectory_errors.py` and `trajectory.py`.

## Notation and frames

- **W**: fixed frame of the groundtruth (GT).
- **M**: fixed frame of the estimate (VIO). Before alignment, the estimate's poses are expressed in M.
- **B**: body frame. GT and the estimate track the same body frame.
- `T_WB_gt(i)`, `T_MB_es(i)`: GT pose and estimated pose at matched sample `i`.
- `S = (s, R, t)`: alignment computed according to `align_type` / `align_num_frames`. For `se3` and `posyaw`, `s = 1`. It maps the estimate into W:
  `p_es_aligned = s·R·p_es + t`, `R_es_aligned = R·R_es`.

### Statistics

Each metric gives one value per sample. From those values the code computes `rmse`, `mean`, `median`, `std`, `min`, `max` and `num_samples`. Along-track and cross-track errors are stored with their sign, but their statistics and boxplots use the absolute value. Yaw, pitch and roll errors are stored as absolute values from the start.

---

## 1. Absolute trajectory error (ATE)

This metric compares each aligned estimated pose with the GT pose at the same timestamp, in W. The result depends on the alignment. With `posyaw` over the first N frames, it shows how drift accumulates from the start of the trajectory. With `se3` or `sim3` over all frames, it shows how consistent the trajectory is globally.

Output file: `saved_results/traj_est/absolute_err_statistics_<align_type>_<n>.yaml`

| Metric (yaml key) | Computation (per sample `i`) | Unit | Interpretation |
|---|---|---|---|
| `trans` | `‖e‖` with `e = p_es_aligned − p_gt`, in W | m | Full 3D position error. |
| `trans_xy` | `‖e_xy‖` | m | Horizontal position error. |
| `trans_z` | `|e_z|` | m | Vertical position error. Vertical is observable from gravity, so it is usually much smaller than `trans_xy`. |
| `along_track` | `e_xy · d`, where `d` is the unit xy direction of the local GT motion `p_gt(i+1) − p_gt(i)` | m (signed) | Error along the direction of travel. **Sign:** because `e = es − gt`, a positive value means the estimate is ahead of the GT. A systematic bias here often comes from a scale error or a time offset. |
| `cross_track` | `e_xy · d⊥`, where `d⊥ = d` rotated by +90° (to the left) | m (signed) | Lateral error. Positive means the estimate lies to the left of the GT. A cross-track error that keeps growing usually comes from yaw drift. |
| `rot` | Angle of `ΔR = R_es_aligned · R_gtᵀ`, i.e. `‖log(ΔR)‖` | deg | Total orientation error. |
| `yaw`, `pitch`, `roll` | `|·|` of the ZYX Euler angles of `ΔR` | deg | Components of the orientation error. `ΔR` is expressed in W, so these are rotations about the W axes z, y and x, **not about the body axes**. Yaw is unobservable in VIO and drifts. Pitch and roll are observable from gravity and should stay bounded. |
| `scale` | See caveat C2 | % | Intended as a scale error. Its current definition is not meaningful. |

In the samples, a GT sample with no xy motion gets `d = 0`, so its along-track and cross-track errors are 0.

---

## 2. Relative pose error (RPE, "odometry error")

This metric measures the drift that builds up over a sub-trajectory of a given length. It **does not depend on the global alignment**. It is computed from the raw estimate (in M). The only part of the alignment it uses is the scale `s`, which is 1 unless `sim3` is used.

Output file, one per sub-trajectory length `L`: `saved_results/traj_est/relative_error_statistics_<L>.yaml`

### How sub-trajectories are formed and compared

1. For **every** sample `i` used as a start, the end sample `j` is the one whose accumulated GT path length is closest to `dist(i) + L`. It must lie within `max_dist_diff = 0.2·L`. Start samples with no such end sample are dropped. Consecutive sub-trajectories therefore overlap strongly.
2. Relative motions:
   `T_es_rel = T_es(i)⁻¹ · T_es(j)` (translation scaled by `s`), and
   `T_gt_rel = T_gt(i)⁻¹ · T_gt(j)`.
3. Error: `E = T_gt_rel⁻¹ · T_es_rel`.
   This equals **anchoring the estimated sub-trajectory's start pose onto the GT start pose with a full 6-DoF transform and taking the pose error at the endpoint**: `E = T_gt(j)⁻¹ · T_es_anchored(j)`. At this point, `E` is expressed in the GT endpoint body frame.
4. `E` is re-expressed in a gravity-aligned fixed frame: `E_w = R_es(j) · E · R_es(j)ᵀ`. Here `R_es(j)` is the raw estimated orientation, so the fixed frame is M (see caveat C3).
5. The metrics below come from `E_w`, giving one value per start sample. The statistics are taken over all start samples.

So yes: the start of each sub-trajectory is aligned (6-DoF, per sub-trajectory), and the endpoint pose errors make up the statistics.

| Metric (yaml key) | Computation | Unit | Interpretation |
|---|---|---|---|
| `trans` / `trans_perc` | `‖t_E‖`; `/L·100` | m / % | Endpoint position drift over length `L`. `trans_perc` is the standard "drift in % of distance travelled". The percentage uses the nominal `L`, not the actual GT path length, which can differ by up to ±20 %. |
| `trans_xy` / `trans_xy_perc` | `‖t_E,xy‖`; `/L·100` | m / % | Horizontal drift. |
| `trans_z` / `trans_z_perc` | `|t_E,z|`; `/L·100` | m / % | Vertical drift. |
| `along_track` / `along_track_perc` | `e_W,xy · d`, where `e_W = R_gt(j) · t_E` is the endpoint error rotated into W with the **GT** endpoint orientation (not via `E_w`, see C3), and `d` is the unit xy chord of the GT sub-trajectory `p_gt(j) − p_gt(i)`; `/L·100` | m / % (signed) | Drift along the direction of travel. **Sign:** `e_W = p_es_anchored(j) − p_gt(j)`, so a positive value means the estimate overshoots. This matches the ATE sign. It is mainly a symptom of scale error. |
| `cross_track` / `cross_track_perc` | `e_W,xy · d⊥`; `/L·100` | m / % (signed) | Lateral drift. Positive means the estimate ends up to the left. This is mainly a symptom of yaw drift. |
| `rot` / `rot_deg_per_m` | Rotation angle of `E`; `/L` | deg / deg·m⁻¹ | Total orientation drift over `L`. |
| `yaw`, `pitch`, `roll` | `|·|` of the ZYX Euler angles of `E_w` | deg | Yaw drift is the dominant unobservable VIO drift. Pitch and roll are about the axes of the fixed frame, not the body axes. |
| `rot_yaw_per_m`, `rot_pitch_per_m`, `rot_roll_per_m` | The three values above `/L` | deg·m⁻¹ | Length-normalised yaw, pitch and roll drift. |
| `gravity` | `sqrt(pitch² + roll²)` of `E_w` | deg | Approximate tilt error, i.e. how wrong the estimated gravity direction is. It should stay small and not grow with `L`. |

### How to read the RPE

- `trans_perc` and `rot_deg_per_m` should be roughly constant across `L` for random-walk-like drift. Short `L` values are dominated by noise.
- Samples overlap and are correlated, so `num_samples` overstates how many independent samples there are.
- The rotation angle (`rot`), the norms (`trans`, `trans_xy`, `trans_z`) and `yaw` do not change with the choice of fixed frame. The pitch/roll split does (see caveat C3). The along/cross-track split does not, because it uses the GT frame.

---

## Caveats in the current implementation

- **C1 – RPE index shift.** `compute_comparison_indices_length` returns only the end indices that were found. `compute_relative_error` then pairs the k-th end index with start sample `k`. If a start sample in the *middle* of the trajectory has no end sample within ±0.2·L, every later pair is shifted. This happens when GT spacing is large compared with 0.4·L, for example with 2 Hz sampling and short sub-trajectory lengths. Missing start samples at the end of the trajectory are harmless.
- **C2 – `scale` metric.** It uses `np.diff(p, 0)`, which is the identity, so it computes `|‖p_es_aligned‖ / ‖p_gt‖ − 1|·100`. That is a ratio of distances from the origin, not of per-step motions. Do not interpret it as scale drift.
- **C3 – RPE frame.** Step 4 rotates the error with the **raw estimated** orientation, so `E_w` ends up in M's axes. If M and W differ by a yaw `ψ`, the pitch/roll split is rotated by `ψ`. The norms, `rot`, `yaw` and `gravity` are not affected. `along_track` and `cross_track` are **not** taken from `E_w`. They rotate `t_E` with the GT endpoint orientation, so they are expressed in W like `d`, and any M–W yaw has no effect on them.
- **C4 – Along/cross-track reference direction.** ATE uses the direction between consecutive GT samples. That direction is noisy when the GT is noisy compared with the motion per sample (slow motion, high rate), and it is undefined when the platform stands still: those samples get 0 and pull the statistics down. RPE uses the GT chord of the sub-trajectory. On strongly curved sub-trajectories (chord ≪ `L`), "along" means along the chord, not along the path.
