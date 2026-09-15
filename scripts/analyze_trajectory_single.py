#!/usr/bin/env python2

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from colorama import init, Fore

import add_path
from trajectory import Trajectory
import plot_utils as pu
from fn_constants import kNsToEstFnMapping, kNsToMatchFnMapping, kFnExt
from multiple_traj_errors import MulTrajError

init(autoreset=True)
#rc('font', **{'family': 'serif', 'serif': ['Cardo']})  # LEVIN
rc('font', family='serif')
#rc('text', usetex=True) # LEVIN

FORMAT = '.pdf'


def parse_float_values(values):
    """Parse comma- or whitespace-separated float values from the CLI."""
    value_string = ' '.join(values).strip().strip('[]')
    return [float(value) for value in value_string.replace(',', ' ').split()]


def analyze_multiple_trials(results_dir, est_type, n_trials,
                            recalculate_errors=False,
                            preset_boxplot_distances=[],
                            preset_boxplot_percentages=[0.1, 0.2, 0.3, 0.4, 0.5],
                            compute_odometry_error=True):
    traj_list = []
    mt_error = MulTrajError()
    for trial_i in range(n_trials):
        if n_trials == 1:
            suffix = ''
        else:
            suffix = str(trial_i)
        print(Fore.RED+"### Trial {0} ###".format(trial_i))

        match_base_fn = kNsToMatchFnMapping[est_type]+suffix+'.'+kFnExt

        if recalculate_errors:
            Trajectory.remove_cached_error(results_dir,
                                           est_type, suffix)
            Trajectory.remove_files_in_save_dir(results_dir, est_type,
                                                match_base_fn)
        # instantiate Trajectory object (loads data and computes alignment)
        traj = Trajectory(
            results_dir, est_type=est_type, suffix=suffix,
            nm_est=kNsToEstFnMapping[est_type] + suffix + '.'+kFnExt,
            nm_matches=match_base_fn,
            preset_boxplot_distances=preset_boxplot_distances,
            preset_boxplot_percentages=preset_boxplot_percentages)
        if traj.data_loaded:
            traj.compute_absolute_error()
            if compute_odometry_error:
                traj.compute_relative_errors()
        if traj.success:
            # write error stats to yaml and cache the errors
            traj.cache_current_error()
            traj.write_errors_to_yaml()

        if traj.success and not preset_boxplot_distances:
            print("Save the boxplot distances for next trials.")
            preset_boxplot_distances = traj.preset_boxplot_distances

        if traj.success:
            mt_error.addTrajectoryError(traj, trial_i)
            traj_list.append(traj)
        else:
            print("Trials {0} fails, will not count.".format(trial_i))
    mt_error.summary()
    mt_error.updateStatistics()
    return traj_list, mt_error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Analyze trajectory estimate in a folder.''')
    parser.add_argument(
        'result_dir', type=str,
        help="Folder containing the groundtruth and the estimate.")
    parser.add_argument(
        '--plots_dir', type=str,
        help="Folder to output plots",
        default='')
    parser.add_argument(
        '--mul_trials', type=int,
        help='number of trials, None for single run', default=None)
    parser.add_argument(
        '--mul_plot_idx', type=int,
        nargs="*",
        help='index of  trials for plotting', default=[0])
    parser.add_argument(
        '--est_types', nargs="*", type=str,
        default=['traj_est'])
    parser.add_argument('--recalculate_errors',
                        help='Deletes cached errors', action='store_true')
    parser.add_argument('--png',
                        help='Save plots as png instead of pdf',
                        action='store_true')
    parser.add_argument('--plot_scale_traj',
                        help='whether to plot scale colored trajectory (slow)',
                        action='store_true')
    parser.add_argument('--plot', dest='plot',
                        action='store_true')
    parser.add_argument('--no_plot', dest='plot',
                        action='store_false')
    parser.add_argument(
        '--preset_rpe_subtrajectory_lengths_meters', type=str, nargs="+",
        help='Preset subtrajectory lengths for relative error computation in meters',
    )
    parser.add_argument(
        '--preset_rpe_subtrajectory_lengths_percentage', type=str, nargs="+",
        help='Preset subtrajectory lengths as percentages of total trajectory length',
    )
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    assert os.path.exists(args.result_dir)

    for est_type in args.est_types:
        assert est_type in kNsToEstFnMapping
        assert est_type in kNsToMatchFnMapping

    if (args.preset_rpe_subtrajectory_lengths_meters is not None and
            args.preset_rpe_subtrajectory_lengths_percentage is not None):
        raise ValueError(
            "Cannot specify both preset_rpe_subtrajectory_lengths_meters and preset_rpe_subtrajectory_lengths_percentage. Please choose one."
        )

    if args.preset_rpe_subtrajectory_lengths_meters is not None:
        args.preset_rpe_subtrajectory_lengths_meters = parse_float_values(
            args.preset_rpe_subtrajectory_lengths_meters)
    if args.preset_rpe_subtrajectory_lengths_percentage is not None:
        args.preset_rpe_subtrajectory_lengths_percentage = [
            percentage / 100.0
            for percentage in parse_float_values(
                args.preset_rpe_subtrajectory_lengths_percentage)]

    top_plots_dir = args.plots_dir
    if not args.plots_dir:
        top_plots_dir = os.path.join(args.result_dir, 'plots')
    if not os.path.exists(top_plots_dir):
        os.makedirs(top_plots_dir)

    plots_dirs = []
    for est_type in args.est_types:
        plot_dir_i = os.path.join(top_plots_dir, est_type)
        if not os.path.exists(plot_dir_i):
            os.makedirs(plot_dir_i)
        plots_dirs.append(plot_dir_i)
    if args.png:
        FORMAT = '.png'

    print(Fore.YELLOW + "=== Summary ===")
    print(Fore.YELLOW +
          "Going to analyze the results in {0}.".format(args.result_dir))
    print(Fore.YELLOW +
          "Will analyze estimate types: {0}".format(args.est_types))
    print(Fore.YELLOW + 
          "The plots will saved in {0}.".format(plots_dirs))
    n_trials = 1
    if args.mul_trials:
        print(Fore.YELLOW + 
              "We will ananlyze multiple trials #{0}".format(args.mul_trials))
        n_trials = args.mul_trials
        if len(args.mul_plot_idx) == 0:
            args.mul_plot_idx = (np.arange(args.mul_trials)).tolist()
        print(Fore.YELLOW + 
              "We will plot trials {0}.".format(args.mul_plot_idx))
    else:
        args.mul_plot_idx = [0]
    assert len(args.mul_plot_idx) == 1, "Multiple plots not supported yet"

    for est_type_i, plot_dir_i in zip(args.est_types, plots_dirs):
        print(Fore.RED +
              "#### Processing error type {0} ####".format(est_type_i))
        mt_error = MulTrajError()
        traj_list, mt_error = analyze_multiple_trials(
            args.result_dir, 
            est_type_i, 
            n_trials, 
            args.recalculate_errors,
            args.preset_rpe_subtrajectory_lengths_meters or [],
            args.preset_rpe_subtrajectory_lengths_percentage or [],
        )
        if traj_list:
            plot_traj = traj_list[args.mul_plot_idx[0]]
        else:
            print("No success runs, not plotting.")

        if n_trials > 1:
            print(">>> Save results for multiple runs in {0}...".format(
                mt_error.save_results_dir))
            mt_error.saveErrors()
            mt_error.cache_current_error()

        if not args.plot:
            print("#### Skip plotting and go to next error type.")
            continue

        # ----------------------------------------------------------------------
        # CREATE PLOTS
        # ----------------------------------------------------------------------
            # absolute error plots
        print(Fore.MAGENTA +
              ">>> Plotting absolute error for one trajectory...")
                # trajectory top view
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, aspect='equal',
                             xlabel='x [m]', ylabel='y [m]')
        pu.plot_trajectory_top(ax, plot_traj.p_es_aligned, 'b', 'Estimate')
        pu.plot_trajectory_top(ax, plot_traj.p_gt, 'm', 'Groundtruth')
        pu.plot_aligned_top(ax, plot_traj.p_es_aligned, plot_traj.p_gt,
                            plot_traj.align_num_frames)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/trajectory_top' + '_' + plot_traj.align_str +
                    FORMAT, bbox_inches="tight")

            # trajectory side view
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, aspect='equal',
                             xlabel='x [m]', ylabel='z [m]')
        pu.plot_trajectory_side(ax, plot_traj.p_es_aligned, 'b', 'Estimate')
        pu.plot_trajectory_side(ax, plot_traj.p_gt, 'm', 'Groundtruth')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/trajectory_side' + '_' + plot_traj.align_str +
                    FORMAT, bbox_inches="tight")

            # absolute translation error
        fig = plt.figure(figsize=(8, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance [m]', ylabel='Position Drift [mm]',
            xlim=[0, plot_traj.accum_distances[-1]])
        pu.plot_error_n_dim(ax, plot_traj.accum_distances,
                            plot_traj.abs_errors['abs_e_trans_vec']*1000,
                            plot_dir_i)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/translation_error' + '_' + plot_traj.align_str
                    + FORMAT, bbox_inches="tight")

            # absolute rotation error
        fig = plt.figure(figsize=(8, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance [m]', ylabel='Orient. err. [deg]',
            xlim=[0, plot_traj.accum_distances[-1]])
        pu.plot_error_n_dim(
            ax, plot_traj.accum_distances,
            plot_traj.abs_errors['abs_e_ypr']*180.0/np.pi, plot_dir_i,
            labels=['yaw', 'pitch', 'roll'])
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rotation_error'+'_'+plot_traj.align_str +
                    FORMAT, bbox_inches='tight')

            # absolute scale error
        fig = plt.figure(figsize=(8, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance [m]', ylabel='Scale Drift [\%]',
            xlim=[0, plot_traj.accum_distances[-1]])
        pu.plot_error_n_dim(
            ax, plot_traj.accum_distances,
            np.reshape(plot_traj.abs_errors['abs_e_scale_perc'], (-1, 1)),
            plot_dir_i, colors=['b'], labels=['scale'])
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/scale_error'+'_'+plot_traj.align_str+FORMAT,
                    bbox_inches='tight')

        if args.plot_scale_traj:
            fig = plt.figure(figsize=(6, 12))
            ax_top = fig.add_subplot(211, aspect='equal',
                                     xlabel='x [m]', ylabel='y [m]',
                                     title='Top')
            ax_top.grid(ls='--', color='0.7')
            ax_side = fig.add_subplot(212, aspect='equal',
                                      xlabel='x [m]', ylabel='z [m]',
                                      title='Side')
            ax_side.grid(ls='--', color='0.7')
            abs_scale_e = np.abs(
                np.reshape(plot_traj.abs_errors['abs_e_scale_perc'], (-1, 1)),)
            color_idx =\
                (abs_scale_e-np.min(abs_scale_e))/(
                    np.max(abs_scale_e)-np.min(abs_scale_e))
            for idx, val in enumerate(color_idx[:-1]):
                c = matplotlib.cm.jet(val).flatten()
                ax_top.plot(plot_traj.p_gt[idx:idx+2, 0],
                            plot_traj.p_gt[idx:idx+2, 1], color=c)
                ax_side.plot(plot_traj.p_gt[idx:idx+2, 0],
                             plot_traj.p_gt[idx:idx+2, 2], color=c)
            fig.tight_layout()
            fig.savefig(plot_dir_i+'/scale_error_traj' + '_' +
                        plot_traj.align_str + FORMAT, bbox_inches="tight")

        print(Fore.MAGENTA+">>> Plotting relative (odometry) error...")
        suffix = ''
        if n_trials > 1:
            suffix = '_mt'

            # relative error plots
        plot_types = [
            'rel_trans', 'rel_trans_perc', 
            'rel_trans_xy', 'rel_trans_xy_perc', 
            'rel_trans_z', 'rel_trans_z_perc', 
            'rel_rot', 'rel_rot_deg_per_m',
            'rel_yaw', 'rel_yaw_deg_per_m',
            'rel_pitch', 'rel_pitch_deg_per_m',
            'rel_roll', 'rel_roll_deg_per_m',
            'rel_gravity'
        ]
        rel_errors, distances = mt_error.get_relative_errors_and_distances(
            error_types=plot_types)

        labels = ['Estimate']
        colors = ['b']

                # 3D translation error [m]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='3D Translation error [m]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_trans'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_3D_translation_error' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # 3D translation error [%]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='3D Translation error [%]')
        pu.boxplot_compare(
            ax, distances, rel_errors['rel_trans_perc'], labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_3D_translation_error_perc'+suffix+FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # 2D translation error [m]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='x-y Translation error [m]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_trans_xy'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_2D_translation_error' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # 2D translation error [%]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='x-y Translation error [%]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_trans_xy_perc'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_2D_translation_error_perc' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # vertical translation error [m]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='z Translation error [m]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_trans_z'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_vertical_translation_error' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # vertical translation error [%]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='z Translation error [%]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_trans_z_perc'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_vertical_translation_error_perc' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # rotation error [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Rotation error [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_rot'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_rot_' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # yaw error [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Yaw error [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_yaw'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_yaw_error' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # pitch error [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Pitch error [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_pitch'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_pitch_error' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # roll error [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Roll error [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_roll'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_roll_error' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # gravity (pitch-roll) error [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Gravity error [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_gravity'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_gravity_error' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # rotation error per meter traveled [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Rotation error per meter [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_rot_deg_per_m'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_rot_deg_per_m' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # yaw error per meter traveled [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Yaw error per meter [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_yaw_deg_per_m'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_yaw_deg_per_m' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # pitch error per meter traveled [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Pitch error per meter [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_pitch_deg_per_m'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_pitch_deg_per_m' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)

                # roll error per meter traveled [deg]
        fig = plt.figure(figsize=(6, 2.5))
        ax = fig.add_subplot(
            111, xlabel='Distance traveled [m]',
            ylabel='Roll error per meter [deg]')
        pu.boxplot_compare(ax, distances, rel_errors['rel_roll_deg_per_m'],
                           labels, colors)
        fig.tight_layout()
        fig.savefig(plot_dir_i+'/rel_roll_deg_per_m' + suffix + FORMAT,
                    bbox_inches="tight")
        plt.close(fig)


        print(Fore.GREEN +
              "#### Done processing error type {0} ####".format(est_type_i))
    import subprocess as s
    s.call(['notify-send', 'rpg_trajectory_evaluation finished',
            'results in: {0}'.format(os.path.abspath(args.result_dir))])

