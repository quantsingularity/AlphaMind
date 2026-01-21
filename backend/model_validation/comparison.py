"""
Model comparison utilities for financial machine learning models.

This module provides tools for comparing multiple models across different
metrics, statistical tests for model performance differences, and
visualization utilities for model comparison.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_validation.metrics import calculate_metrics
from scipy import stats


class ModelComparison:
    """
    Compare multiple models across different metrics and datasets.

    This class provides utilities for comprehensive model comparison,
    including performance metrics, statistical tests, and visualizations.

    Parameters
    ----------
    models : dict
        Dictionary of models to compare, with model names as keys.
    metrics : list, optional
        List of metric names to calculate. If None, uses default metrics.
    """

    def __init__(
        self, models: Dict[str, Any], metrics: Optional[List[str]] = None
    ) -> None:
        self.models = models
        self.metrics = metrics or ["mse", "rmse", "mae", "r2"]
        self.results = {}
        self.statistical_tests = StatisticalTests()

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        task_type: str = "regression",
        cv: Optional[Any] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Evaluate all models on the given dataset.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        task_type : str, default="regression"
            Type of machine learning task.
        cv : cross-validation generator, optional
            Cross-validation strategy. If None, uses 5-fold CV.
        sample_weights : array-like, optional
            Sample weights.

        Returns
        -------
        results : DataFrame
            DataFrame containing evaluation results.
        """
        from sklearn.model_selection import KFold

        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        self.results = {model_name: [] for model_name in self.models}
        for model_name, model in self.models.items():
            fold_results: List[Any] = []
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = (X[train_idx], X[test_idx])
                y_train, y_test = (y[train_idx], y[test_idx])
                if sample_weights is not None:
                    train_weights = sample_weights[train_idx]
                    test_weights = sample_weights[test_idx]
                    model.fit(X_train, y_train, sample_weight=train_weights)
                else:
                    model.fit(X_train, y_train)
                if task_type == "classification" and hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_test)
                    if y_pred.shape[1] == 2:
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = model.predict(X_test)
                fold_metrics = calculate_metrics(
                    y_test,
                    y_pred,
                    sample_weights=test_weights if sample_weights is not None else None,
                    task_type=task_type,
                )
                fold_results.append(fold_metrics)
            avg_metrics: Dict[str, Any] = {}
            for metric in fold_results[0].keys():
                avg_metrics[metric] = np.mean([fold[metric] for fold in fold_results])
                avg_metrics[f"{metric}_std"] = np.std(
                    [fold[metric] for fold in fold_results]
                )
            self.results[model_name] = avg_metrics
        results_df = pd.DataFrame(self.results).T
        return results_df

    def statistical_comparison(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        baseline_model: str,
        alpha: float = 0.05,
        n_permutations: int = 1000,
    ) -> pd.DataFrame:
        """
        Perform statistical tests to compare models against a baseline.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        baseline_model : str
            Name of the baseline model to compare against.
        alpha : float, default=0.05
            Significance level.
        n_permutations : int, default=1000
            Number of permutations for permutation tests.

        Returns
        -------
        comparison : DataFrame
            DataFrame containing statistical test results.
        """
        if baseline_model not in self.models:
            raise ValueError(f"Baseline model '{baseline_model}' not found in models")
        baseline = self.models[baseline_model]
        comparison_results: Dict[str, Any] = {}
        for model_name, model in self.models.items():
            if model_name == baseline_model:
                continue
            t_test_result = self.statistical_tests.paired_t_test(baseline, model, X, y)
            perm_test_result = self.statistical_tests.permutation_test(
                baseline, model, X, y, n_permutations=n_permutations
            )
            comparison_results[model_name] = {
                "t_statistic": t_test_result[0],
                "t_p_value": t_test_result[1],
                "t_significant": t_test_result[1] < alpha,
                "perm_p_value": perm_test_result,
                "perm_significant": perm_test_result < alpha,
            }
        comparison_df = pd.DataFrame(comparison_results).T
        return comparison_df

    def plot_comparison(
        self,
        metric: str = "rmse",
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        sort_values: bool = True,
    ) -> plt.Figure:
        """
        Plot comparison of models based on a specific metric.

        Parameters
        ----------
        metric : str, default="rmse"
            Metric to plot.
        figsize : tuple, default=(10, 6)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.
        sort_values : bool, default=True
            Whether to sort models by metric value.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        if metric not in next(iter(self.results.values())):
            raise ValueError(f"Metric '{metric}' not found in results")
        metric_values = {
            model: results[metric] for model, results in self.results.items()
        }
        metric_std = {
            model: results.get(f"{metric}_std", 0)
            for model, results in self.results.items()
        }
        df = pd.DataFrame({"value": metric_values, "std": metric_std})
        if sort_values:
            df = df.sort_values("value")
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df.index, df["value"], xerr=df["std"], capsize=5)
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Model")
        ax.set_title(title or f"Model Comparison - {metric.upper()}")
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + df["std"].iloc[i] + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{df['value'].iloc[i]:.4f}",
                va="center",
            )
        plt.tight_layout()
        return fig

    def plot_metric_heatmap(
        self,
        figsize: Tuple[int, int] = (12, 8),
        title: Optional[str] = None,
        cmap: str = "viridis",
    ) -> plt.Figure:
        """
        Plot heatmap of all metrics for all models.

        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size.
        title : str, optional
            Plot title. If None, uses default title.
        cmap : str, default="viridis"
            Colormap for heatmap.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        results_df = pd.DataFrame(self.results).T
        metric_cols = [col for col in results_df.columns if not col.endswith("_std")]
        results_df = results_df[metric_cols]
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(results_df.values, cmap=cmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(metric_cols)))
        ax.set_yticks(np.arange(len(results_df.index)))
        ax.set_xticklabels(metric_cols)
        ax.set_yticklabels(results_df.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(len(results_df.index)):
            for j in range(len(metric_cols)):
                ax.text(
                    j,
                    i,
                    f"{results_df.iloc[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if results_df.iloc[i, j] > results_df[metric_cols[j]].mean()
                        else "black"
                    ),
                )
        ax.set_title(title or "Model Performance Metrics Heatmap")
        plt.tight_layout()
        return fig


class StatisticalTests:
    """
    Statistical tests for comparing model performance.

    This class provides various statistical tests for comparing
    the performance of machine learning models.
    """

    def paired_t_test(
        self,
        model1: Any,
        model2: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        metric_fn: Optional[callable] = None,
    ) -> Tuple[float, float]:
        """
        Perform paired t-test on model predictions.

        Parameters
        ----------
        model1 : object
            First model.
        model2 : object
            Second model.
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        metric_fn : callable, optional
            Function to calculate metric. If None, uses squared error.

        Returns
        -------
        t_statistic : float
            T-statistic.
        p_value : float
            P-value.
        """
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        if metric_fn is None:
            errors1 = (y - y_pred1) ** 2
            errors2 = (y - y_pred2) ** 2
        else:
            errors1 = np.array([metric_fn(y[i], y_pred1[i]) for i in range(len(y))])
            errors2 = np.array([metric_fn(y[i], y_pred2[i]) for i in range(len(y))])
        t_statistic, p_value = stats.ttest_rel(errors1, errors2)
        return (t_statistic, p_value)

    def permutation_test(
        self,
        model1: Any,
        model2: Any,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        metric_fn: Optional[callable] = None,
        n_permutations: int = 1000,
        random_state: Optional[int] = None,
    ) -> float:
        """
        Perform permutation test on model predictions.

        Parameters
        ----------
        model1 : object
            First model.
        model2 : object
            Second model.
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        metric_fn : callable, optional
            Function to calculate metric. If None, uses mean squared error.
        n_permutations : int, default=1000
            Number of permutations.
        random_state : int, optional
            Random state for reproducibility.

        Returns
        -------
        p_value : float
            P-value from permutation test.
        """
        from sklearn.metrics import mean_squared_error

        rng = np.random.RandomState(random_state)
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)
        if metric_fn is None:
            metric1 = mean_squared_error(y, y_pred1)
            metric2 = mean_squared_error(y, y_pred2)
        else:
            metric1 = metric_fn(y, y_pred1)
            metric2 = metric_fn(y, y_pred2)
        observed_diff = abs(metric1 - metric2)
        count = 0
        for _ in range(n_permutations):
            permutation = rng.permutation(len(y))
            y_perm1 = y_pred1[permutation]
            y_perm2 = y_pred2[permutation]
            if metric_fn is None:
                perm_metric1 = mean_squared_error(y, y_perm1)
                perm_metric2 = mean_squared_error(y, y_perm2)
            else:
                perm_metric1 = metric_fn(y, y_perm1)
                perm_metric2 = metric_fn(y, y_perm2)
            perm_diff = abs(perm_metric1 - perm_metric2)
            if perm_diff >= observed_diff:
                count += 1
        p_value = count / n_permutations
        return p_value

    def diebold_mariano_test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        loss_fn: Optional[callable] = None,
        h: int = 1,
    ) -> Tuple[float, float]:
        """
        Perform Diebold-Mariano test for comparing forecast accuracy.

        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred1 : array-like
            Predictions from first model.
        y_pred2 : array-like
            Predictions from second model.
        loss_fn : callable, optional
            Loss function. If None, uses squared error.
        h : int, default=1
            Forecast horizon.

        Returns
        -------
        dm_statistic : float
            Diebold-Mariano statistic.
        p_value : float
            P-value.
        """
        e1 = y_true - y_pred1
        e2 = y_true - y_pred2
        if loss_fn is None:
            d = e1**2 - e2**2
        else:
            d = np.array(
                [
                    loss_fn(y_true[i], y_pred1[i]) - loss_fn(y_true[i], y_pred2[i])
                    for i in range(len(y_true))
                ]
            )
        d_bar = np.mean(d)
        n = len(d)
        gamma_0 = np.sum((d - d_bar) ** 2) / n
        gamma_sum = 0
        for i in range(1, h):
            gamma_i = np.sum((d[i:] - d_bar) * (d[:-i] - d_bar)) / n
            gamma_sum += gamma_i
        var_d = gamma_0 + 2 * gamma_sum
        dm_statistic = d_bar / np.sqrt(var_d / n)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))
        return (dm_statistic, p_value)

    def model_confidence_set(
        self,
        models: Dict[str, Any],
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        loss_fn: Optional[callable] = None,
        alpha: float = 0.05,
        B: int = 1000,
        random_state: Optional[int] = None,
    ) -> List[str]:
        """
        Perform Model Confidence Set procedure to select best models.

        Parameters
        ----------
        models : dict
            Dictionary of models to compare.
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        loss_fn : callable, optional
            Loss function. If None, uses squared error.
        alpha : float, default=0.05
            Significance level.
        B : int, default=1000
            Number of bootstrap samples.
        random_state : int, optional
            Random state for reproducibility.

        Returns
        -------
        selected_models : list
            List of selected model names.
        """
        rng = np.random.RandomState(random_state)
        predictions: Dict[str, Any] = {}
        for name, model in models.items():
            predictions[name] = model.predict(X)
        losses: Dict[str, Any] = {}
        for name, pred in predictions.items():
            if loss_fn is None:
                losses[name] = (y - pred) ** 2
            else:
                losses[name] = np.array([loss_fn(y[i], pred[i]) for i in range(len(y))])
        model_set = list(models.keys())
        while len(model_set) > 1:
            loss_diffs: Dict[str, Any] = {}
            for i, model_i in enumerate(model_set):
                for j, model_j in enumerate(model_set):
                    if i < j:
                        loss_diffs[model_i, model_j] = losses[model_i] - losses[model_j]
            t_stats: Dict[str, Any] = {}
            for (model_i, model_j), diff in loss_diffs.items():
                t_stats[model_i, model_j] = np.mean(diff) / (
                    np.std(diff) / np.sqrt(len(diff))
                )
            max_t = max((abs(t) for t in t_stats.values()))
            max_t_boot: List[Any] = []
            for _ in range(B):
                boot_idx = rng.choice(len(y), size=len(y), replace=True)
                boot_diffs: Dict[str, Any] = {}
                for (model_i, model_j), diff in loss_diffs.items():
                    boot_diffs[model_i, model_j] = diff[boot_idx]
                boot_t_stats: Dict[str, Any] = {}
                for (model_i, model_j), diff in boot_diffs.items():
                    boot_t_stats[model_i, model_j] = np.mean(diff) / (
                        np.std(diff) / np.sqrt(len(diff))
                    )
                max_t_boot.append(max((abs(t) for t in boot_t_stats.values())))
            p_value = np.mean(np.array(max_t_boot) > max_t)
            if p_value < alpha:
                model_losses = {model: np.mean(losses[model]) for model in model_set}
                worst_model = max(model_losses, key=model_losses.get)
                model_set.remove(worst_model)
            else:
                break
        return model_set
