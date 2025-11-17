"""
Model validation reporting for financial machine learning models.

This module provides tools for generating comprehensive validation reports
for machine learning models in financial applications, including performance
metrics, cross-validation results, and model diagnostics.
"""

from datetime import datetime
from io import StringIO
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .comparison import ModelComparison
from .metrics import calculate_metrics


class ValidationReport:
    """
    Comprehensive validation report for machine learning models.

    This class provides methods for generating detailed validation reports
    for machine learning models, including performance metrics, cross-validation
    results, and model diagnostics.

    Parameters
    ----------
    model_name : str
        Name of the model.
    model : object
        Trained machine learning model.
    task_type : str, default="regression"
        Type of machine learning task.
    feature_names : list, optional
        List of feature names. If None, uses X0, X1, etc.
    target_name : str, optional
        Name of the target variable.
    """

    def __init__(
        self,
        model_name: str,
        model: Any,
        task_type: str = "regression",
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model = model
        self.task_type = task_type
        self.feature_names = feature_names
        self.target_name = target_name or "Target"
        self.report_id = str(uuid.uuid4())[:8]
        self.creation_time = datetime.now()

        # Initialize report sections
        self.sections = {
            "model_info": {},
            "performance_metrics": {},
            "cross_validation": {},
            "feature_importance": {},
            "diagnostics": {},
            "warnings": [],
        }

        # Store model info
        self._store_model_info()

    def _store_model_info(self) -> None:
        """Store basic model information in the report."""
        model_info = {
            "model_name": self.model_name,
            "model_type": type(self.model).__name__,
            "task_type": self.task_type,
            "report_id": self.report_id,
            "creation_time": self.creation_time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Try to extract model parameters
        try:
            if hasattr(self.model, "get_params"):
                model_info["parameters"] = self.model.get_params()
        except:
            model_info["parameters"] = "Could not extract model parameters"

        self.sections["model_info"] = model_info

    def add_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        dataset_name: str = "test",
        custom_metrics: Optional[Dict[str, callable]] = None,
    ) -> None:
        """
        Add performance metrics to the report.

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values.
        sample_weights : array-like, optional
            Sample weights.
        dataset_name : str, default="test"
            Name of the dataset (e.g., "train", "test", "validation").
        custom_metrics : dict, optional
            Dictionary of custom metric functions with names as keys.
        """
        from .metrics import calculate_metrics

        # Calculate metrics
        metrics = calculate_metrics(
            y_true,
            y_pred,
            sample_weights=sample_weights,
            task_type=self.task_type,
            custom_metrics=custom_metrics,
        )

        # Store metrics
        self.sections["performance_metrics"][dataset_name] = metrics

    def add_cross_validation_results(
        self, cv_results: Dict[str, List[float]], cv_method: str = "k-fold"
    ) -> None:
        """
        Add cross-validation results to the report.

        Parameters
        ----------
        cv_results : dict
            Dictionary of cross-validation results, with metric names as keys
            and lists of values for each fold as values.
        cv_method : str, default="k-fold"
            Cross-validation method used.
        """
        # Calculate summary statistics
        cv_summary = {}
        for metric, values in cv_results.items():
            cv_summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "values": values,
            }

        # Store cross-validation results
        self.sections["cross_validation"] = {
            "method": cv_method,
            "n_folds": len(next(iter(cv_results.values()))),
            "metrics": cv_summary,
        }

    def add_feature_importance(self, importance_values: pd.DataFrame) -> None:
        """
        Add feature importance to the report.

        Parameters
        ----------
        importance_values : DataFrame
            DataFrame containing feature importance values.
        """
        # Store feature importance
        self.sections["feature_importance"] = importance_values.to_dict(
            orient="records"
        )

    def add_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Add residual analysis to the report (for regression models).

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values.
        """
        if self.task_type != "regression":
            self.sections["warnings"].append(
                "Residual analysis is only applicable for regression models"
            )
            return

        # Calculate residuals
        residuals = y_true - y_pred

        # Calculate residual statistics
        residual_stats = {
            "mean": np.mean(residuals),
            "std": np.std(residuals),
            "min": np.min(residuals),
            "max": np.max(residuals),
            "skewness": float(pd.Series(residuals).skew()),
            "kurtosis": float(pd.Series(residuals).kurtosis()),
        }

        # Check for normality
        from scipy import stats

        _, p_value = stats.shapiro(residuals)
        residual_stats["shapiro_p_value"] = p_value
        residual_stats["is_normal"] = p_value > 0.05

        # Store residual analysis
        self.sections["diagnostics"]["residual_analysis"] = residual_stats

    def add_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True
    ) -> None:
        """
        Add confusion matrix to the report (for classification models).

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values.
        normalize : bool, default=True
            Whether to normalize the confusion matrix.
        """
        if self.task_type != "classification":
            self.sections["warnings"].append(
                "Confusion matrix is only applicable for classification models"
            )
            return

        from sklearn.metrics import confusion_matrix

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Normalize if requested
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        # Store confusion matrix
        self.sections["diagnostics"]["confusion_matrix"] = {
            "matrix": cm.tolist(),
            "normalized": normalize,
        }

    def add_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        """
        Add ROC curve to the report (for binary classification models).

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_score : array-like
            Target scores (probability estimates for the positive class).
        """
        if self.task_type != "classification":
            self.sections["warnings"].append(
                "ROC curve is only applicable for classification models"
            )
            return

        from sklearn.metrics import auc, roc_curve

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Store ROC curve
        self.sections["diagnostics"]["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": roc_auc,
        }

    def add_precision_recall_curve(
        self, y_true: np.ndarray, y_score: np.ndarray
    ) -> None:
        """
        Add precision-recall curve to the report (for binary classification models).

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_score : array-like
            Target scores (probability estimates for the positive class).
        """
        if self.task_type != "classification":
            self.sections["warnings"].append(
                "Precision-recall curve is only applicable for classification models"
            )
            return

        from sklearn.metrics import average_precision_score, precision_recall_curve

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        # Store precision-recall curve
        self.sections["diagnostics"]["precision_recall_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": ap,
        }

    def add_learning_curve(
        self, train_sizes: np.ndarray, train_scores: np.ndarray, test_scores: np.ndarray
    ) -> None:
        """
        Add learning curve to the report.

        Parameters
        ----------
        train_sizes : array-like
            Training set sizes.
        train_scores : array-like
            Scores on training sets.
        test_scores : array-like
            Scores on test sets.
        """
        # Store learning curve
        self.sections["diagnostics"]["learning_curve"] = {
            "train_sizes": train_sizes.tolist(),
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "train_scores_std": np.std(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
            "test_scores_std": np.std(test_scores, axis=1).tolist(),
        }

    def add_calibration_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> None:
        """
        Add calibration curve to the report (for classification models).

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_prob : array-like
            Probability estimates.
        n_bins : int, default=10
            Number of bins to discretize the [0, 1] interval.
        """
        if self.task_type != "classification":
            self.sections["warnings"].append(
                "Calibration curve is only applicable for classification models"
            )
            return

        from sklearn.calibration import calibration_curve

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Store calibration curve
        self.sections["diagnostics"]["calibration_curve"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "n_bins": n_bins,
        }

    def add_custom_section(self, section_name: str, content: Any) -> None:
        """
        Add custom section to the report.

        Parameters
        ----------
        section_name : str
            Name of the section.
        content : any
            Content of the section.
        """
        # Convert content to JSON-serializable format if possible
        try:
            json.dumps(content)
            self.sections[section_name] = content
        except:
            self.sections[section_name] = str(content)

    def generate_report(
        self,
        format: str = "markdown",
        output_file: Optional[str] = None,
        include_plots: bool = True,
    ) -> Union[str, None]:
        """
        Generate validation report.

        Parameters
        ----------
        format : str, default="markdown"
            Report format. Options: "markdown", "html", "json".
        output_file : str, optional
            Path to save the report. If None, returns the report as a string.
        include_plots : bool, default=True
            Whether to include plots in the report.

        Returns
        -------
        report : str or None
            Report as a string if output_file is None, otherwise None.
        """
        if format == "json":
            return self._generate_json_report(output_file)
        elif format == "markdown":
            return self._generate_markdown_report(output_file, include_plots)
        elif format == "html":
            return self._generate_html_report(output_file, include_plots)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_json_report(
        self, output_file: Optional[str] = None
    ) -> Union[str, None]:
        """Generate JSON report."""
        # Convert to JSON
        report_json = json.dumps(self.sections, indent=2)

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_json)
            return None

        return report_json

    def _generate_markdown_report(
        self, output_file: Optional[str] = None, include_plots: bool = True
    ) -> Union[str, None]:
        """Generate Markdown report."""
        # Initialize report
        report = StringIO()

        # Add title
        report.write(f"# Model Validation Report: {self.model_name}\n\n")

        # Add model info
        report.write("## Model Information\n\n")
        model_info = self.sections["model_info"]
        report.write(f"- **Model Name:** {model_info['model_name']}\n")
        report.write(f"- **Model Type:** {model_info['model_type']}\n")
        report.write(f"- **Task Type:** {model_info['task_type']}\n")
        report.write(f"- **Report ID:** {model_info['report_id']}\n")
        report.write(f"- **Creation Time:** {model_info['creation_time']}\n\n")

        # Add parameters if available
        if "parameters" in model_info and isinstance(model_info["parameters"], dict):
            report.write("### Model Parameters\n\n")
            report.write("| Parameter | Value |\n")
            report.write("|-----------|-------|\n")
            for param, value in model_info["parameters"].items():
                report.write(f"| {param} | {value} |\n")
            report.write("\n")

        # Add performance metrics
        if self.sections["performance_metrics"]:
            report.write("## Performance Metrics\n\n")

            for dataset, metrics in self.sections["performance_metrics"].items():
                report.write(f"### {dataset.capitalize()} Set\n\n")
                report.write("| Metric | Value |\n")
                report.write("|--------|------:|\n")

                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        report.write(f"| {metric} | {value:.4f} |\n")
                    else:
                        report.write(f"| {metric} | {value} |\n")

                report.write("\n")

        # Add cross-validation results
        if self.sections["cross_validation"]:
            cv = self.sections["cross_validation"]
            report.write("## Cross-Validation Results\n\n")
            report.write(f"- **Method:** {cv['method']}\n")
            report.write(f"- **Number of Folds:** {cv['n_folds']}\n\n")

            report.write("| Metric | Mean | Std | Min | Max |\n")
            report.write("|--------|-----:|----:|----:|----:|\n")

            for metric, stats in cv["metrics"].items():
                report.write(
                    f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                    f"{stats['min']:.4f} | {stats['max']:.4f} |\n"
                )

            report.write("\n")

        # Add feature importance
        if self.sections["feature_importance"]:
            report.write("## Feature Importance\n\n")
            report.write("| Feature | Importance |\n")
            report.write("|---------|----------:|\n")

            for item in self.sections["feature_importance"]:
                feature = item["Feature"]
                importance = item["Importance"]
                report.write(f"| {feature} | {importance:.4f} |\n")

            report.write("\n")

        # Add diagnostics
        if self.sections["diagnostics"]:
            report.write("## Model Diagnostics\n\n")

            # Add residual analysis
            if "residual_analysis" in self.sections["diagnostics"]:
                report.write("### Residual Analysis\n\n")
                ra = self.sections["diagnostics"]["residual_analysis"]

                report.write("| Statistic | Value |\n")
                report.write("|-----------|------:|\n")
                report.write(f"| Mean | {ra['mean']:.4f} |\n")
                report.write(f"| Std | {ra['std']:.4f} |\n")
                report.write(f"| Min | {ra['min']:.4f} |\n")
                report.write(f"| Max | {ra['max']:.4f} |\n")
                report.write(f"| Skewness | {ra['skewness']:.4f} |\n")
                report.write(f"| Kurtosis | {ra['kurtosis']:.4f} |\n")
                report.write(f"| Shapiro p-value | {ra['shapiro_p_value']:.4f} |\n")
                report.write(f"| Is Normal | {ra['is_normal']} |\n\n")

            # Add confusion matrix
            if "confusion_matrix" in self.sections["diagnostics"]:
                report.write("### Confusion Matrix\n\n")
                cm = self.sections["diagnostics"]["confusion_matrix"]

                report.write("```\n")
                for row in cm["matrix"]:
                    report.write(
                        " ".join(
                            f"{x:.4f}" if cm["normalized"] else f"{x}" for x in row
                        )
                        + "\n"
                    )
                report.write("```\n\n")

            # Add ROC curve
            if "roc_curve" in self.sections["diagnostics"]:
                report.write("### ROC Curve\n\n")
                roc = self.sections["diagnostics"]["roc_curve"]
                report.write(f"- **AUC:** {roc['auc']:.4f}\n\n")

            # Add precision-recall curve
            if "precision_recall_curve" in self.sections["diagnostics"]:
                report.write("### Precision-Recall Curve\n\n")
                pr = self.sections["diagnostics"]["precision_recall_curve"]
                report.write(
                    f"- **Average Precision:** {pr['average_precision']:.4f}\n\n"
                )

            # Add learning curve
            if "learning_curve" in self.sections["diagnostics"]:
                report.write("### Learning Curve\n\n")
                lc = self.sections["diagnostics"]["learning_curve"]

                report.write(
                    "| Train Size | Train Score | Train Std | Test Score | Test Std |\n"
                )
                report.write(
                    "|----------:|-----------:|---------:|----------:|--------:|\n"
                )

                for i, size in enumerate(lc["train_sizes"]):
                    report.write(
                        f"| {size} | {lc['train_scores_mean'][i]:.4f} | {lc['train_scores_std'][i]:.4f} | "
                        f"{lc['test_scores_mean'][i]:.4f} | {lc['test_scores_std'][i]:.4f} |\n"
                    )

                report.write("\n")

            # Add calibration curve
            if "calibration_curve" in self.sections["diagnostics"]:
                report.write("### Calibration Curve\n\n")
                cc = self.sections["diagnostics"]["calibration_curve"]

                report.write("| Predicted Probability | True Probability |\n")
                report.write("|----------------------:|----------------:|\n")

                for i in range(len(cc["prob_pred"])):
                    report.write(
                        f"| {cc['prob_pred'][i]:.4f} | {cc['prob_true'][i]:.4f} |\n"
                    )

                report.write("\n")

        # Add warnings
        if self.sections["warnings"]:
            report.write("## Warnings\n\n")

            for warning in self.sections["warnings"]:
                report.write(f"- {warning}\n")

            report.write("\n")

        # Add custom sections
        for section_name, content in self.sections.items():
            if section_name not in [
                "model_info",
                "performance_metrics",
                "cross_validation",
                "feature_importance",
                "diagnostics",
                "warnings",
            ]:
                report.write(f"## {section_name}\n\n")

                if isinstance(content, dict):
                    for key, value in content.items():
                        report.write(f"- **{key}:** {value}\n")
                elif isinstance(content, list):
                    for item in content:
                        report.write(f"- {item}\n")
                else:
                    report.write(f"{content}\n")

                report.write("\n")

        # Get report as string
        report_str = report.getvalue()

        # Save to file if specified
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_str)
            return None

        return report_str

    def _generate_html_report(
        self, output_file: Optional[str] = None, include_plots: bool = True
    ) -> Union[str, None]:
        """Generate HTML report."""
        # First generate markdown report
        md_report = self._generate_markdown_report(None, include_plots)

        # Convert markdown to HTML
        try:
            import markdown

            html_report = markdown.markdown(md_report, extensions=["tables"])

            # Add basic styling
            html_report = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Model Validation Report: {self.model_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .warning {{ color: #e74c3c; }}
                </style>
            </head>
            <body>
                {html_report}
            </body>
            </html>
            """

            # Save to file if specified
            if output_file:
                with open(output_file, "w") as f:
                    f.write(html_report)
                return None

            return html_report

        except ImportError:
            # If markdown package is not available, fall back to markdown
            if output_file:
                self._generate_markdown_report(output_file, include_plots)
                return None

            return md_report

    def plot_performance_comparison(
        self,
        datasets: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot performance comparison across datasets.

        Parameters
        ----------
        datasets : list, optional
            List of dataset names to include. If None, includes all datasets.
        metrics : list, optional
            List of metrics to include. If None, includes all metrics.
        figsize : tuple, default=(12, 8)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if not self.sections["performance_metrics"]:
            raise ValueError("No performance metrics available")

        # Use all datasets if not specified
        if datasets is None:
            datasets = list(self.sections["performance_metrics"].keys())

        # Filter datasets that exist
        datasets = [d for d in datasets if d in self.sections["performance_metrics"]]

        if not datasets:
            raise ValueError("No valid datasets specified")

        # Get all metrics from first dataset
        all_metrics = list(self.sections["performance_metrics"][datasets[0]].keys())

        # Use all metrics if not specified
        if metrics is None:
            metrics = all_metrics

        # Filter metrics that exist
        metrics = [m for m in metrics if m in all_metrics]

        if not metrics:
            raise ValueError("No valid metrics specified")

        # Create data for plotting
        data = []
        for dataset in datasets:
            for metric in metrics:
                value = self.sections["performance_metrics"][dataset].get(metric)
                if value is not None and isinstance(value, (int, float)):
                    data.append({"Dataset": dataset, "Metric": metric, "Value": value})

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot grouped bar chart
        sns.barplot(x="Metric", y="Value", hue="Dataset", data=df, ax=ax)

        # Add labels and title
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title(title or f"Performance Metrics Comparison: {self.model_name}")

        # Rotate x-tick labels if there are many metrics
        if len(metrics) > 4:
            plt.xticks(rotation=45, ha="right")

        # Add legend
        ax.legend(title="Dataset")

        # Adjust layout
        plt.tight_layout()

        return fig

    def plot_cross_validation_results(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot cross-validation results.

        Parameters
        ----------
        metrics : list, optional
            List of metrics to include. If None, includes all metrics.
        figsize : tuple, default=(12, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if not self.sections["cross_validation"]:
            raise ValueError("No cross-validation results available")

        cv = self.sections["cross_validation"]

        # Get all metrics
        all_metrics = list(cv["metrics"].keys())

        # Use all metrics if not specified
        if metrics is None:
            metrics = all_metrics

        # Filter metrics that exist
        metrics = [m for m in metrics if m in all_metrics]

        if not metrics:
            raise ValueError("No valid metrics specified")

        # Create figure
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

        # Handle single metric case
        if len(metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Get metric data
            values = cv["metrics"][metric]["values"]
            mean = cv["metrics"][metric]["mean"]
            std = cv["metrics"][metric]["std"]

            # Plot values
            ax.boxplot(values)

            # Add mean line
            ax.axhline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}")

            # Add labels
            ax.set_title(f"{metric}")
            ax.set_ylabel("Value")
            ax.set_xlabel("Fold")

            # Add legend
            ax.legend()

        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Cross-Validation Results: {self.model_name}", fontsize=16)

        # Adjust layout
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)

        return fig

    def plot_feature_importance(
        self,
        top_n: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        color: str = "skyblue",
        show_values: bool = True,
    ) -> plt.Figure:
        """
        Plot feature importance.

        Parameters
        ----------
        top_n : int, optional
            Number of top features to plot. If None, plots all features.
        figsize : tuple, default=(10, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.
        color : str, default="skyblue"
            Bar color.
        show_values : bool, default=True
            Whether to show importance values on bars.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if not self.sections["feature_importance"]:
            raise ValueError("No feature importance available")

        # Convert to DataFrame
        df = pd.DataFrame(self.sections["feature_importance"])

        # Select top N features if specified
        if top_n is not None:
            df = df.head(top_n)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot bars
        bars = ax.barh(df["Feature"], df["Importance"], color=color)

        # Add error bars if available
        if "Std" in df.columns:
            ax.errorbar(
                df["Importance"],
                df["Feature"],
                xerr=df["Std"],
                fmt="none",
                ecolor="black",
                capsize=3,
            )

        # Add values to bars if requested
        if show_values:
            for bar in bars:
                ax.text(
                    bar.get_width() + (0.01 * max(df["Importance"])),
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}",
                    va="center",
                )

        # Add labels and title
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(title or f"Feature Importance: {self.model_name}")

        # Adjust layout
        plt.tight_layout()

        return fig

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: Tuple[int, int] = (12, 10),
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot residual analysis (for regression models).

        Parameters
        ----------
        y_true : array-like
            Ground truth (correct) target values.
        y_pred : array-like
            Estimated target values.
        figsize : tuple, default=(12, 10)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.task_type != "regression":
            raise ValueError(
                "Residual analysis is only applicable for regression models"
            )

        # Calculate residuals
        residuals = y_true - y_pred

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Plot residuals vs predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color="r", linestyle="-")
        axes[0, 0].set_xlabel("Predicted Values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Predicted")

        # Plot residuals histogram
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color="skyblue")
        axes[0, 1].axvline(x=0, color="r", linestyle="-")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Residuals Distribution")

        # Plot Q-Q plot
        from scipy import stats

        stats.probplot(residuals, plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")

        # Plot actual vs predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--")

        axes[1, 1].set_xlabel("Actual Values")
        axes[1, 1].set_ylabel("Predicted Values")
        axes[1, 1].set_title("Actual vs Predicted")

        # Add overall title
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Residual Analysis: {self.model_name}", fontsize=16)

        # Adjust layout
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)

        return fig

    def plot_confusion_matrix(
        self,
        figsize: Tuple[int, int] = (8, 6),
        title: Optional[str] = None,
        cmap: str = "Blues",
    ) -> plt.Figure:
        """
        Plot confusion matrix (for classification models).

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.
        cmap : str, default="Blues"
            Colormap for heatmap.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.task_type != "classification":
            raise ValueError(
                "Confusion matrix is only applicable for classification models"
            )

        if "confusion_matrix" not in self.sections["diagnostics"]:
            raise ValueError("No confusion matrix available")

        cm = self.sections["diagnostics"]["confusion_matrix"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(cm["matrix"], interpolation="nearest", cmap=cmap)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(
            "Count" if not cm["normalized"] else "Proportion", rotation=-90, va="bottom"
        )

        # Add labels
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title or f"Confusion Matrix: {self.model_name}")

        # Add text to cells
        thresh = (
            (np.array(cm["matrix"]) > 0.5).astype(int)
            if cm["normalized"]
            else np.max(cm["matrix"]) / 2
        )
        for i in range(len(cm["matrix"])):
            for j in range(len(cm["matrix"][i])):
                ax.text(
                    j,
                    i,
                    (
                        f"{cm['matrix'][i][j]:.2f}"
                        if cm["normalized"]
                        else f"{cm['matrix'][i][j]}"
                    ),
                    ha="center",
                    va="center",
                    color="white" if cm["matrix"][i][j] > thresh else "black",
                )

        # Adjust layout
        plt.tight_layout()

        return fig

    def plot_roc_curve(
        self, figsize: Tuple[int, int] = (8, 6), title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve (for classification models).

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.task_type != "classification":
            raise ValueError("ROC curve is only applicable for classification models")

        if "roc_curve" not in self.sections["diagnostics"]:
            raise ValueError("No ROC curve available")

        roc = self.sections["diagnostics"]["roc_curve"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot ROC curve
        ax.plot(
            roc["fpr"], roc["tpr"], lw=2, label=f'ROC curve (AUC = {roc["auc"]:.4f})'
        )

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], "k--", lw=2)

        # Add labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or f"ROC Curve: {self.model_name}")

        # Add legend
        ax.legend(loc="lower right")

        # Adjust layout
        plt.tight_layout()

        return fig

    def plot_precision_recall_curve(
        self, figsize: Tuple[int, int] = (8, 6), title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curve (for classification models).

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.task_type != "classification":
            raise ValueError(
                "Precision-recall curve is only applicable for classification models"
            )

        if "precision_recall_curve" not in self.sections["diagnostics"]:
            raise ValueError("No precision-recall curve available")

        pr = self.sections["diagnostics"]["precision_recall_curve"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot precision-recall curve
        ax.plot(
            pr["recall"],
            pr["precision"],
            lw=2,
            label=f'Precision-Recall curve (AP = {pr["average_precision"]:.4f})',
        )

        # Add labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or f"Precision-Recall Curve: {self.model_name}")

        # Add legend
        ax.legend(loc="lower left")

        # Adjust layout
        plt.tight_layout()

        return fig

    def plot_learning_curve(
        self, figsize: Tuple[int, int] = (8, 6), title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot learning curve.

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if "learning_curve" not in self.sections["diagnostics"]:
            raise ValueError("No learning curve available")

        lc = self.sections["diagnostics"]["learning_curve"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot learning curve
        train_sizes = lc["train_sizes"]
        train_scores_mean = lc["train_scores_mean"]
        train_scores_std = lc["train_scores_std"]
        test_scores_mean = lc["test_scores_mean"]
        test_scores_std = lc["test_scores_std"]

        # Plot training scores
        ax.fill_between(
            train_sizes,
            [m - s for m, s in zip(train_scores_mean, train_scores_std)],
            [m + s for m, s in zip(train_scores_mean, train_scores_std)],
            alpha=0.1,
            color="r",
        )
        ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")

        # Plot test scores
        ax.fill_between(
            train_sizes,
            [m - s for m, s in zip(test_scores_mean, test_scores_std)],
            [m + s for m, s in zip(test_scores_mean, test_scores_std)],
            alpha=0.1,
            color="g",
        )
        ax.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )

        # Add labels and title
        ax.set_xlabel("Training examples")
        ax.set_ylabel("Score")
        ax.set_title(title or f"Learning Curve: {self.model_name}")

        # Add legend
        ax.legend(loc="best")

        # Add grid
        ax.grid(True)

        # Adjust layout
        plt.tight_layout()

        return fig

    def plot_calibration_curve(
        self, figsize: Tuple[int, int] = (8, 6), title: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot calibration curve (for classification models).

        Parameters
        ----------
        figsize : tuple, default=(8, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if self.task_type != "classification":
            raise ValueError(
                "Calibration curve is only applicable for classification models"
            )

        if "calibration_curve" not in self.sections["diagnostics"]:
            raise ValueError("No calibration curve available")

        cc = self.sections["diagnostics"]["calibration_curve"]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot calibration curve
        ax.plot(cc["prob_pred"], cc["prob_true"], "s-", label=f"{self.model_name}")

        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

        # Add labels and title
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(title or f"Calibration Curve: {self.model_name}")

        # Add legend
        ax.legend(loc="best")

        # Adjust layout
        plt.tight_layout()

        return fig
