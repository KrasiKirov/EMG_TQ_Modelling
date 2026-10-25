# evaluation package — shared metrics and plotting for model assessment
from .metrics import snap_to_operating_points, compute_metrics, compute_per_position_metrics
from .plots import (plot_full_trials, plot_per_position_metrics,
                    plot_r2_summary, plot_r2_overlay)
