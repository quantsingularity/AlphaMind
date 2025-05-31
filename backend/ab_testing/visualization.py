"""
Visualization tools for A/B testing results.

This module provides classes for visualizing A/B test results,
including plots for comparing metrics, showing statistical significance,
and tracking experiment progress over time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import datetime
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class ExperimentVisualizer:
    """
    Class for visualizing A/B test results.
    
    This class provides methods for creating various visualizations
    of A/B test results, including metric comparisons, statistical
    significance, and time series plots.
    """
    
    def __init__(
        self,
        style: str = "whitegrid",
        context: str = "notebook",
        palette: str = "deep",
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        style : str, default="whitegrid"
            Seaborn style.
        context : str, default="notebook"
            Seaborn context.
        palette : str, default="deep"
            Color palette.
        figsize : tuple, default=(10, 6)
            Default figure size.
        """
        self.style = style
        self.context = context
        self.palette = palette
        self.figsize = figsize
        
        # Set up plotting style
        sns.set_style(style)
        sns.set_context(context)
        sns.set_palette(palette)
    
    def plot_metric_comparison(
        self,
        results: pd.DataFrame,
        metric: str,
        variant_col: str = "variant",
        value_col: str = "value",
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        show_stats: bool = True,
        plot_type: str = "box"
    ) -> Figure:
        """
        Plot comparison of a metric across variants.
        
        Parameters
        ----------
        results : DataFrame
            DataFrame containing results.
        metric : str
            Name of the metric to plot.
        variant_col : str, default="variant"
            Name of the column containing variant names.
        value_col : str, default="value"
            Name of the column containing metric values.
        figsize : tuple, optional
            Figure size. If None, uses default.
        title : str, optional
            Plot title. If None, uses default.
        show_stats : bool, default=True
            Whether to show summary statistics.
        plot_type : str, default="box"
            Type of plot. Options: "box", "violin", "bar", "strip".
            
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Filter results for the specified metric
        if "metric" in results.columns:
            df = results[results["metric"] == metric].copy()
        else:
            df = results.copy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Create plot based on type
        if plot_type == "box":
            sns.boxplot(x=variant_col, y=value_col, data=df, ax=ax)
            if show_stats:
                sns.stripplot(x=variant_col, y=value_col, data=df, ax=ax, 
                             color="black", alpha=0.5, size=3, jitter=True)
        elif plot_type == "violin":
            sns.violinplot(x=variant_col, y=value_col, data=df, ax=ax, inner="point")
            if show_stats:
                sns.stripplot(x=variant_col, y=value_col, data=df, ax=ax, 
                             color="black", alpha=0.5, size=3, jitter=True)
        elif plot_type == "bar":
            # Calculate mean and confidence interval
            sns.barplot(x=variant_col, y=value_col, data=df, ax=ax, capsize=0.1)
        elif plot_type == "strip":
            sns.stripplot(x=variant_col, y=value_col, data=df, ax=ax, jitter=True)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        # Add summary statistics if requested
        if show_stats:
            stats = df.groupby(variant_col)[value_col].agg(['mean', 'std', 'count'])
            stats_text = "Summary Statistics:\n"
            for variant, row in stats.iterrows():
                stats_text += f"{variant}: Mean={row['mean']:.4f}, Std={row['std']:.4f}, n={row['count']}\n"
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
        
        # Add labels and title
        ax.set_xlabel("Variant")
        ax.set_ylabel(metric)
        ax.set_title(title or f"Comparison of {metric} across Variants")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_metric_over_time(
        self,
        results: pd.DataFrame,
        metric: str,
        variant_col: str = "variant",
        value_col: str = "value",
        time_col: str = "timestamp",
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        rolling_window: Optional[int] = None,
        confidence_interval: Optional[float] = None
    ) -> Figure:
        """
        Plot metric values over time.
        
        Parameters
        ----------
        results : DataFrame
            DataFrame containing results.
        metric : str
            Name of the metric to plot.
        variant_col : str, default="variant"
            Name of the column containing variant names.
        value_col : str, default="value"
            Name of the column containing metric values.
        time_col : str, default="timestamp"
            Name of the column containing timestamps.
        figsize : tuple, optional
            Figure size. If None, uses default.
        title : str, optional
            Plot title. If None, uses default.
        rolling_window : int, optional
            Window size for rolling average. If None, no rolling average is applied.
        confidence_interval : float, optional
            Confidence interval for the rolling average. If None, no confidence interval is shown.
            
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Filter results for the specified metric
        if "metric" in results.columns:
            df = results[results["metric"] == metric].copy()
        else:
            df = results.copy()
        
        # Ensure timestamp column is datetime
        if pd.api.types.is_string_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Plot raw data as scatter
        for variant, group in df.groupby(variant_col):
            ax.scatter(group[time_col], group[value_col], label=f"{variant} (raw)", 
                      alpha=0.3, s=10)
        
        # Add rolling average if requested
        if rolling_window is not None:
            for variant, group in df.groupby(variant_col):
                # Sort by time
                group = group.sort_values(time_col)
                
                # Calculate rolling average
                rolling_avg = group[value_col].rolling(window=rolling_window, min_periods=1).mean()
                
                # Plot rolling average
                ax.plot(group[time_col], rolling_avg, label=f"{variant} (rolling avg)", linewidth=2)
                
                # Add confidence interval if requested
                if confidence_interval is not None:
                    rolling_std = group[value_col].rolling(window=rolling_window, min_periods=1).std()
                    z_score = stats.norm.ppf(1 - (1 - confidence_interval) / 2)
                    margin = z_score * rolling_std / np.sqrt(rolling_window)
                    
                    ax.fill_between(
                        group[time_col],
                        rolling_avg - margin,
                        rolling_avg + margin,
                        alpha=0.2,
                        label=f"{variant} ({confidence_interval:.0%} CI)"
                    )
        
        # Add labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel(metric)
        ax.set_title(title or f"{metric} over Time")
        
        # Add legend
        ax.legend()
        
        # Format x-axis as dates
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_cumulative_metric(
        self,
        results: pd.DataFrame,
        metric: str,
        variant_col: str = "variant",
        value_col: str = "value",
        time_col: str = "timestamp",
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        confidence_interval: Optional[float] = None
    ) -> Figure:
        """
        Plot cumulative metric values over time.
        
        Parameters
        ----------
        results : DataFrame
            DataFrame containing results.
        metric : str
            Name of the metric to plot.
        variant_col : str, default="variant"
            Name of the column containing variant names.
        value_col : str, default="value"
            Name of the column containing metric values.
        time_col : str, default="timestamp"
            Name of the column containing timestamps.
        figsize : tuple, optional
            Figure size. If None, uses default.
        title : str, optional
            Plot title. If None, uses default.
        confidence_interval : float, optional
            Confidence interval for the cumulative average. If None, no confidence interval is shown.
            
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Filter results for the specified metric
        if "metric" in results.columns:
            df = results[results["metric"] == metric].copy()
        else:
            df = results.copy()
        
        # Ensure timestamp column is datetime
        if pd.api.types.is_string_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Plot cumulative average for each variant
        for variant, group in df.groupby(variant_col):
            # Sort by time
            group = group.sort_values(time_col)
            
            # Calculate cumulative statistics
            group["cumulative_mean"] = group[value_col].expanding().mean()
            group["cumulative_std"] = group[value_col].expanding().std()
            group["cumulative_count"] = np.arange(1, len(group) + 1)
            
            # Plot cumulative mean
            ax.plot(group[time_col], group["cumulative_mean"], label=variant, linewidth=2)
            
            # Add confidence interval if requested
            if confidence_interval is not None:
                z_score = stats.norm.ppf(1 - (1 - confidence_interval) / 2)
                margin = z_score * group["cumulative_std"] / np.sqrt(group["cumulative_count"])
                
                ax.fill_between(
                    group[time_col],
                    group["cumulative_mean"] - margin,
                    group["cumulative_mean"] + margin,
                    alpha=0.2
                )
        
        # Add labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Cumulative Average {metric}")
        ax.set_title(title or f"Cumulative Average {metric} over Time")
        
        # Add legend
        ax.legend()
        
        # Format x-axis as dates
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_significance_test(
        self,
        test_results: Dict,
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot results of a statistical significance test.
        
        Parameters
        ----------
        test_results : dict
            Dictionary containing test results.
        figsize : tuple, optional
            Figure size. If None, uses default.
        title : str, optional
            Plot title. If None, uses default.
            
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Extract test type
        test_type = test_results.get("test", "Unknown Test")
        
        # Plot based on test type
        if test_type == "t-test":
            self._plot_t_test_results(test_results, ax)
        elif test_type == "mann-whitney-u":
            self._plot_mann_whitney_results(test_results, ax)
        elif test_type == "bayesian-ab-test":
            self._plot_bayesian_results(test_results, ax)
        else:
            ax.text(0.5, 0.5, f"Visualization not implemented for {test_type}",
                   ha="center", va="center", transform=ax.transAxes)
        
        # Add title
        ax.set_title(title or f"Results of {test_type}")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def _plot_t_test_results(
        self,
        test_results: Dict,
        ax: Axes
    ) -> None:
        """
        Plot t-test results.
        
        Parameters
        ----------
        test_results : dict
            Dictionary containing t-test results.
        ax : Axes
            Matplotlib axes to plot on.
        """
        # Extract relevant values
        control_mean = test_results["control_mean"]
        treatment_mean = test_results["treatment_mean"]
        control_std = test_results["control_std"]
        treatment_std = test_results["treatment_std"]
        control_n = test_results["control_n"]
        treatment_n = test_results["treatment_n"]
        p_value = test_results["p_value"]
        ci_lower = test_results["ci_lower"]
        ci_upper = test_results["ci_upper"]
        
        # Create bar plot
        variants = ["Control", "Treatment"]
        means = [control_mean, treatment_mean]
        stds = [control_std / np.sqrt(control_n), treatment_std / np.sqrt(treatment_n)]
        
        ax.bar(variants, means, yerr=stds, capsize=10, alpha=0.7)
        
        # Add p-value and significance
        is_significant = p_value < 0.05
        sig_text = "Significant" if is_significant else "Not Significant"
        ax.text(0.5, 0.95, f"p-value: {p_value:.4f} ({sig_text})",
               ha="center", va="top", transform=ax.transAxes,
               bbox=dict(boxstyle="round", alpha=0.1, color="green" if is_significant else "red"))
        
        # Add confidence interval
        ax.text(0.5, 0.85, f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]",
               ha="center", va="top", transform=ax.transAxes,
               bbox=dict(boxstyle="round", alpha=0.1))
        
        # Add sample sizes
        ax.text(0, 0.05, f"n = {control_n}", ha="center", va="bottom", transform=ax.transAxes)
        ax.text(1, 0.05, f"n = {treatment_n}", ha="center", va="bottom", transform=ax.transAxes)
        
        # Add labels
        ax.set_ylabel("Mean Value")
        ax.set_xlabel("Variant")
    
    def _plot_mann_whitney_results(
        self,
        test_results: Dict,
        ax: Axes
    ) -> None:
        """
        Plot Mann-Whitney U test results.
        
        Parameters
        ----------
        test_results : dict
            Dictionary containing Mann-Whitney U test results.
        ax : Axes
            Matplotlib axes to plot on.
        """
        # Extract relevant values
        control_median = test_results["control_median"]
        treatment_median = test_results["treatment_median"]
        control_n = test_results["control_n"]
        treatment_n = test_results["treatment_n"]
        p_value = test_results["p_value"]
        effect_size = test_results.get("effect_size_r", 0)
        
        # Create bar plot
        variants = ["Control", "Treatment"]
        medians = [control_median, treatment_median]
        
        ax.bar(variants, medians, alpha=0.7)
        
        # Add p-value and significance
        is_significant = p_value < 0.05
        sig_text = "Significant" if is_significant else "Not Significant"
        ax.text(0.5, 0.95, f"p-value: {p_value:.4f} ({sig_text})",
               ha="center", va="top", transform=ax.transAxes,
               bbox=dict(boxstyle="round", alpha=0.1, color="green" if is_significant else "red"))
        
        # Add effect size
        effect_size_text = "Small" if abs(effect_size) < 0.3 else "Medium" if abs(effect_size) < 0.5 else "Large"
        ax.text(0.5, 0.85, f"Effect Size: {effect_size:.4f} ({effect_size_text})",
               ha="center", va="top", transform=ax.transAxes,
               bbox=dict(boxstyle="round", alpha=0.1))
        
        # Add sample sizes
        ax.text(0, 0.05, f"n = {control_n}", ha="center", va="bottom", transform=ax.transAxes)
        ax.text(1, 0.05, f"n = {treatment_n}", ha="center", va="bottom", transform=ax.transAxes)
        
        # Add labels
        ax.set_ylabel("Median Value")
        ax.set_xlabel("Variant")
    
    def _plot_bayesian_results(
        self,
        test_results: Dict,
        ax: Axes
    ) -> None:
        """
        Plot Bayesian A/B test results.
        
        Parameters
        ----------
        test_results : dict
            Dictionary containing Bayesian A/B test results.
        ax : Axes
            Matplotlib axes to plot on.
        """
        # Extract relevant values
        prob_improvement = test_results["prob_improvement"]
        expected_improvement = test_results["expected_improvement"]
        ci_lower = test_results["ci_lower"]
        ci_upper = test_results["ci_upper"]
        
        # Create horizontal bar for probability
        ax.barh(["Probability of Improvement"], [prob_improvement], color="skyblue", alpha=0.7)
        ax.axvline(0.95, color="red", linestyle="--", label="95% Threshold")
        
        # Add expected improvement
        ax.text(0.5, 0.8, f"Expected Improvement: {expected_improvement:.4f}",
               ha="center", va="top", transform=ax.transAxes,
               bbox=dict(boxstyle="round", alpha=0.1))
        
        # Add credible interval
        ax.text(0.5, 0.7, f"95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]",
               ha="center", va="top", transform=ax.transAxes,
               bbox=dict(boxstyle="round", alpha=0.1))
        
        # Add significance
        is_significant = prob_improvement > 0.95
        sig_text = "Significant" if is_significant else "Not Significant"
        ax.text(0.5, 0.9, f"Probability: {prob_improvement:.4f} ({sig_text})",
               ha="center", va="top", transform=ax.transAxes,
               bbox=dict(boxstyle="round", alpha=0.1, color="green" if is_significant else "red"))
        
        # Set limits and labels
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_ylabel("")
        
        # Add legend
        ax.legend()
    
    def plot_multiple_metrics(
        self,
        results: pd.DataFrame,
        metrics: List[str],
        variant_col: str = "variant",
        value_col: str = "value",
        metric_col: str = "metric",
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None,
        plot_type: str = "bar"
    ) -> Figure:
        """
        Plot multiple metrics across variants.
        
        Parameters
        ----------
        results : DataFrame
            DataFrame containing results.
        metrics : list
            List of metrics to plot.
        variant_col : str, default="variant"
            Name of the column containing variant names.
        value_col : str, default="value"
            Name of the column containing metric values.
        metric_col : str, default="metric"
            Name of the column containing metric names.
        figsize : tuple, optional
            Figure size. If None, uses default.
        title : str, optional
            Plot title. If None, uses default.
        plot_type : str, default="bar"
            Type of plot. Options: "bar", "box", "violin".
            
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Filter results for the specified metrics
        df = results[results[metric_col].isin(metrics)].copy()
        
        if df.empty:
            raise ValueError(f"No data found for metrics: {metrics}")
        
        # Determine grid layout
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        # Create figure
        figsize = figsize or (5 * n_cols, 4 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # Handle single subplot case
        if n_metrics == 1:
            axes = np.array([axes])
        
        # Flatten axes for easy indexing
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                metric_df = df[df[metric_col] == metric]
                
                if plot_type == "bar":
                    # Calculate mean and confidence interval
                    sns.barplot(x=variant_col, y=value_col, data=metric_df, ax=ax, capsize=0.1)
                elif plot_type == "box":
                    sns.boxplot(x=variant_col, y=value_col, data=metric_df, ax=ax)
                elif plot_type == "violin":
                    sns.violinplot(x=variant_col, y=value_col, data=metric_df, ax=ax, inner="point")
                else:
                    raise ValueError(f"Unsupported plot type: {plot_type}")
                
                # Add labels
                ax.set_title(metric)
                ax.set_xlabel("Variant")
                ax.set_ylabel("Value")
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_conversion_funnel(
        self,
        results: pd.DataFrame,
        funnel_stages: List[str],
        variant_col: str = "variant",
        metric_col: str = "metric",
        value_col: str = "value",
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot conversion funnel for different variants.
        
        Parameters
        ----------
        results : DataFrame
            DataFrame containing results.
        funnel_stages : list
            List of funnel stages (metrics) in order.
        variant_col : str, default="variant"
            Name of the column containing variant names.
        metric_col : str, default="metric"
            Name of the column containing metric names.
        value_col : str, default="value"
            Name of the column containing metric values.
        figsize : tuple, optional
            Figure size. If None, uses default.
        title : str, optional
            Plot title. If None, uses default.
            
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Filter results for the specified metrics
        df = results[results[metric_col].isin(funnel_stages)].copy()
        
        if df.empty:
            raise ValueError(f"No data found for funnel stages: {funnel_stages}")
        
        # Calculate mean values for each variant and stage
        funnel_data = df.groupby([variant_col, metric_col])[value_col].mean().reset_index()
        
        # Pivot data for plotting
        pivot_data = funnel_data.pivot(index=metric_col, columns=variant_col, values=value_col)
        
        # Reorder rows based on funnel stages
        pivot_data = pivot_data.reindex(funnel_stages)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        
        # Plot funnel
        pivot_data.plot(kind="bar", ax=ax)
        
        # Add labels and title
        ax.set_xlabel("Funnel Stage")
        ax.set_ylabel("Conversion Rate")
        ax.set_title(title or "Conversion Funnel by Variant")
        
        # Add legend
        ax.legend(title="Variant")
        
        # Add value labels
        for i, variant in enumerate(pivot_data.columns):
            for j, stage in enumerate(pivot_data.index):
                value = pivot_data.loc[stage, variant]
                ax.text(j + (i / len(pivot_data.columns)) - 0.4 + (0.8 / len(pivot_data.columns)),
                       value + 0.01,
                       f"{value:.2f}",
                       ha="center", va="bottom", rotation=90)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_experiment_calendar(
        self,
        experiments: List[Dict],
        figsize: Optional[Tuple[int, int]] = None,
        title: Optional[str] = None
    ) -> Figure:
        """
        Plot calendar view of experiments.
        
        Parameters
        ----------
        experiments : list
            List of experiment dictionaries with start_date and end_date.
        figsize : tuple, optional
            Figure size. If None, uses default.
        title : str, optional
            Plot title. If None, uses default.
            
        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize or (12, 8))
        
        # Sort experiments by start date
        sorted_experiments = sorted(experiments, key=lambda x: x.get("start_date", datetime.datetime.now()))
        
        # Plot each experiment as a horizontal bar
        for i, exp in enumerate(sorted_experiments):
            name = exp.get("name", f"Experiment {i+1}")
            start_date = exp.get("start_date")
            end_date = exp.get("end_date", datetime.datetime.now())
            status = exp.get("status", "unknown")
            
            if start_date:
                # Determine color based on status
                if status == "completed":
                    color = "green"
                elif status == "running":
                    color = "blue"
                elif status == "paused":
                    color = "orange"
                elif status == "failed":
                    color = "red"
                else:
                    color = "gray"
                
                # Plot bar
                ax.barh(i, (end_date - start_date).total_seconds() / (24 * 3600),
                       left=start_date, height=0.5, color=color, alpha=0.7)
                
                # Add experiment name
                ax.text(start_date, i, f" {name} ", va="center", ha="left",
                       bbox=dict(boxstyle="round", alpha=0.1))
        
        # Set y-ticks
        ax.set_yticks(range(len(sorted_experiments)))
        ax.set_yticklabels([])
        
        # Format x-axis as dates
        fig.autofmt_xdate()
        
        # Add labels and title
        ax.set_xlabel("Date")
        ax.set_ylabel("Experiments")
        ax.set_title(title or "Experiment Calendar")
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="green", alpha=0.7, label="Completed"),
            Patch(facecolor="blue", alpha=0.7, label="Running"),
            Patch(facecolor="orange", alpha=0.7, label="Paused"),
            Patch(facecolor="red", alpha=0.7, label="Failed"),
            Patch(facecolor="gray", alpha=0.7, label="Unknown")
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
