"""
Model validation reporting for financial machine learning models.

This module provides tools for generating comprehensive validation reports
for machine learning models in financial applications, including performance
metrics, cross-validation results, and model diagnostics.
"""

from datetime import datetime
from io import StringIO
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn import metrics as sk_metrics
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


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
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.task_type = task_type
        self.feature_names = feature_names
        self.target_name = target_name or "Target"
        self.report_id = str(uuid.uuid4())[:8]
        self.creation_time = datetime.now()
        self.sections = {
            "model_info": {},
            "performance_metrics": {},
            "cross_validation": {},
            "feature_importance": {},
            "diagnostics": {},
            "warnings": [],
        }
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
        try:
            if hasattr(self.model, "get_params"):
                model_info["parameters"] = self.model.get_params()
        except Exception:
            model_info["parameters"] = "Could not extract model parameters"
        self.sections["model_info"] = model_info

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Internal helper to calculate standard metrics based on task type."""
        metrics = {}
        if task_type == "regression":
            metrics["MSE"] = sk_metrics.mean_squared_error(
                y_true, y_pred, sample_weight=sample_weights
            )
            metrics["RMSE"] = np.sqrt(metrics["MSE"])
            metrics["MAE"] = sk_metrics.mean_absolute_error(
                y_true, y_pred, sample_weight=sample_weights
            )
            metrics["R2"] = sk_metrics.r2_score(
                y_true, y_pred, sample_weight=sample_weights
            )
        elif task_type == "classification":
            metrics["Accuracy"] = sk_metrics.accuracy_score(
                y_true, y_pred, sample_weight=sample_weights
            )
            # Assuming binary classification for simplicity in defaults,
            # or macro average for multiclass
            avg_method = "binary" if len(np.unique(y_true)) == 2 else "macro"
            metrics["Precision"] = sk_metrics.precision_score(
                y_true,
                y_pred,
                average=avg_method,
                sample_weight=sample_weights,
                zero_division=0,
            )
            metrics["Recall"] = sk_metrics.recall_score(
                y_true,
                y_pred,
                average=avg_method,
                sample_weight=sample_weights,
                zero_division=0,
            )
            metrics["F1"] = sk_metrics.f1_score(
                y_true,
                y_pred,
                average=avg_method,
                sample_weight=sample_weights,
                zero_division=0,
            )
        return metrics

    def add_performance_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        dataset_name: str = "test",
        custom_metrics: Optional[Dict[str, Callable]] = None,
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
        # Calculate standard metrics internally to avoid external dependency issues
        metrics = self._calculate_metrics(
            y_true, y_pred, self.task_type, sample_weights
        )

        if custom_metrics:
            for name, func in custom_metrics.items():
                try:
                    metrics[name] = func(y_true, y_pred)
                except Exception as e:
                    self.sections["warnings"].append(
                        f"Failed to calculate custom metric {name}: {str(e)}"
                    )

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
        cv_summary = {}
        for metric, values in cv_results.items():
            cv_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": values,
            }
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
            DataFrame containing feature importance values. Must have columns 'Feature' and 'Importance'.
        """
        if (
            "Feature" not in importance_values.columns
            or "Importance" not in importance_values.columns
        ):
            self.sections["warnings"].append(
                "Feature importance DataFrame missing required columns 'Feature' or 'Importance'."
            )
            return

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

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        residuals = y_true - y_pred

        residual_stats = {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "skewness": float(pd.Series(residuals).skew()),
            "kurtosis": float(pd.Series(residuals).kurtosis()),
        }

        # Handle small sample sizes for Shapiro test if necessary,
        # though scipy handles up to N=5000 typically.
        if len(residuals) > 3:
            _, p_value = stats.shapiro(residuals)
            residual_stats["shapiro_p_value"] = float(p_value)
            residual_stats["is_normal"] = p_value > 0.05

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

        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            # Avoid division by zero
            row_sums = cm.sum(axis=1)[:, np.newaxis]
            row_sums[row_sums == 0] = 1
            cm = cm.astype("float") / row_sums

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

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        self.sections["diagnostics"]["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc),
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

        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        self.sections["diagnostics"]["precision_recall_curve"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": float(ap),
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

        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
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
        try:
            # Ensure content is JSON serializable if possible, otherwise convert to str
            json.dumps(content)
            self.sections[section_name] = content
        except (TypeError, OverflowError):
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
        report_json = json.dumps(self.sections, indent=2)
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_json)
            return None
        return report_json

    def _generate_markdown_report(
        self, output_file: Optional[str] = None, include_plots: bool = True
    ) -> Union[str, None]:
        """Generate Markdown report."""
        report = StringIO()
        report.write(f"# Model Validation Report: {self.model_name}\n\n")
        report.write("## Model Information\n\n")
        model_info = self.sections["model_info"]
        report.write(f"- **Model Name:** {model_info.get('model_name', 'N/A')}\n")
        report.write(f"- **Model Type:** {model_info.get('model_type', 'N/A')}\n")
        report.write(f"- **Task Type:** {model_info.get('task_type', 'N/A')}\n")
        report.write(f"- **Report ID:** {model_info.get('report_id', 'N/A')}\n")
        report.write(
            f"- **Creation Time:** {model_info.get('creation_time', 'N/A')}\n\n"
        )

        if "parameters" in model_info and isinstance(model_info["parameters"], dict):
            report.write("### Model Parameters\n\n")
            report.write("| Parameter | Value |\n")
            report.write("|-----------|-------|\n")
            for param, value in model_info["parameters"].items():
                report.write(f"| {param} | {value} |\n")
            report.write("\n")

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

        if self.sections["cross_validation"]:
            cv = self.sections["cross_validation"]
            report.write("## Cross-Validation Results\n\n")
            report.write(f"- **Method:** {cv.get('method', 'N/A')}\n")
            report.write(f"- **Number of Folds:** {cv.get('n_folds', 'N/A')}\n\n")
            report.write("| Metric | Mean | Std | Min | Max |\n")
            report.write("|--------|-----:|----:|----:|----:|\n")
            for metric, stats in cv["metrics"].items():
                report.write(
                    f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n"
                )
            report.write("\n")

        if self.sections["feature_importance"]:
            report.write("## Feature Importance\n\n")
            report.write("| Feature | Importance |\n")
            report.write("|---------|----------:|\n")
            # Sort by importance desc for display
            fi_list = sorted(
                self.sections["feature_importance"],
                key=lambda x: x.get("Importance", 0),
                reverse=True,
            )
            for item in fi_list[:20]:  # Show top 20 in table to save space
                feature = item.get("Feature", "Unknown")
                importance = item.get("Importance", 0)
                report.write(f"| {feature} | {importance:.4f} |\n")
            if len(fi_list) > 20:
                report.write(f"| ... | ... |\n")
            report.write("\n")

        if self.sections["diagnostics"]:
            report.write("## Model Diagnostics\n\n")
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
                if "shapiro_p_value" in ra:
                    report.write(f"| Shapiro p-value | {ra['shapiro_p_value']:.4f} |\n")
                    report.write(f"| Is Normal | {ra['is_normal']} |\n")
                report.write("\n")

            if "confusion_matrix" in self.sections["diagnostics"]:
                report.write("### Confusion Matrix\n\n")
                cm = self.sections["diagnostics"]["confusion_matrix"]
                report.write("```\n")
                for row in cm["matrix"]:
                    report.write(
                        " ".join(
                            (f"{x:.4f}" if cm["normalized"] else f"{x}" for x in row)
                        )
                        + "\n"
                    )
                report.write("```\n\n")

            if "roc_curve" in self.sections["diagnostics"]:
                report.write("### ROC Curve\n\n")
                roc = self.sections["diagnostics"]["roc_curve"]
                report.write(f"- **AUC:** {roc['auc']:.4f}\n\n")

            if "precision_recall_curve" in self.sections["diagnostics"]:
                report.write("### Precision-Recall Curve\n\n")
                pr = self.sections["diagnostics"]["precision_recall_curve"]
                report.write(
                    f"- **Average Precision:** {pr['average_precision']:.4f}\n\n"
                )

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
                        f"| {size} | {lc['train_scores_mean'][i]:.4f} | {lc['train_scores_std'][i]:.4f} | {lc['test_scores_mean'][i]:.4f} | {lc['test_scores_std'][i]:.4f} |\n"
                    )
                report.write("\n")

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

        if self.sections["warnings"]:
            report.write("## Warnings\n\n")
            for warning in self.sections["warnings"]:
                report.write(f"- {warning}\n")
            report.write("\n")

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

        report_str = report.getvalue()
        if output_file:
            with open(output_file, "w") as f:
                f.write(report_str)
            return None
        return report_str

    def _generate_html_report(
        self, output_file: Optional[str] = None, include_plots: bool = True
    ) -> Union[str, None]:
        """Generate HTML report."""
        md_report = self._generate_markdown_report(None, include_plots)
        try:
            import markdown

            html_content = markdown.markdown(md_report, extensions=["tables"])
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
                {html_content}
            </body>
            </html>
            """
            if output_file:
                with open(output_file, "w") as f:
                    f.write(html_report)
                return None
            return html_report
        except ImportError:
            if output_file:
                return self._generate_markdown_report(output_file, include_plots)
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

        if datasets is None:
            datasets = list(self.sections["performance_metrics"].keys())
        datasets = [d for d in datasets if d in self.sections["performance_metrics"]]

        if not datasets:
            raise ValueError("No valid datasets specified")

        all_metrics = list(self.sections["performance_metrics"][datasets[0]].keys())
        if metrics is None:
            metrics = all_metrics
        metrics = [m for m in metrics if m in all_metrics]

        if not metrics:
            raise ValueError("No valid metrics specified")

        data = []
        for dataset in datasets:
            for metric in metrics:
                value = self.sections["performance_metrics"][dataset].get(metric)
                if value is not None and isinstance(value, (int, float)):
                    data.append({"Dataset": dataset, "Metric": metric, "Value": value})

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x="Metric", y="Value", hue="Dataset", data=df, ax=ax)
        ax.set_xlabel("Metric")
        ax.set_ylabel("Value")
        ax.set_title(title or f"Performance Metrics Comparison: {self.model_name}")

        if len(metrics) > 4:
            plt.xticks(rotation=45, ha="right")
        ax.legend(title="Dataset")
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
        all_metrics = list(cv["metrics"].keys())
        if metrics is None:
            metrics = all_metrics
        metrics = [m for m in metrics if m in all_metrics]

        if not metrics:
            raise ValueError("No valid metrics specified")

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = cv["metrics"][metric]["values"]
            mean = cv["metrics"][metric]["mean"]

            ax.boxplot(values)
            ax.axhline(mean, color="red", linestyle="--", label=f"Mean: {mean:.4f}")
            ax.set_title(f"{metric}")
            ax.set_ylabel("Value")
            ax.set_xlabel("Fold")
            ax.legend()

        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f"Cross-Validation Results: {self.model_name}", fontsize=16)

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
            raise ValueError("No feature importance values available")

        # Convert list of dicts back to DataFrame for plotting
        df = pd.DataFrame(self.sections["feature_importance"])

        # Sort by importance
        df = df.sort_values("Importance", ascending=False)

        if top_n:
            df = df.head(top_n)

        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x="Importance", y="Feature", data=df, color=color, ax=ax)

        if show_values:
            # Add text labels to the end of each bar
            for i, v in enumerate(df["Importance"]):
                ax.text(v, i, f" {v:.4f}", va="center")

        ax.set_title(title or f"Feature Importance: {self.model_name}")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        return fig
