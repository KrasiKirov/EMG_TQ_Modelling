# evaluation package — shared metrics and plotting for model assessment
from .metrics import snap_to_operating_points, compute_metrics, compute_per_position_metrics
from .plots import (plot_full_trials, plot_per_position_metrics,
                    plot_r2_summary, plot_r2_overlay,
                    plot_pred_vs_true, plot_training_curves,
                    plot_capacity_comparison,
                    plot_passive_coverage,
                    plot_passive_interpolation_range,
                    plot_passive_variability,
                    plot_passive_torque_curve,
                    plot_passive_residual_by_position,
                    plot_extreme_vs_middle_summary,
                    plot_nrmse_and_residual_boxplots,
                    plot_emg_torque_scatter_by_position,
                    plot_emg_torque_linearity_by_position,
                    plot_plantarflexion_residual_detail)
