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
from scipy import stats

from .metrics import calculate_metrics


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

    def __init__(self, models: Dict[str, Any], metrics: Optional[List[str]] = None):
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

        # Default to 5-fold CV if not provided
        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize results dictionary
        self.results = {model_name: [] for model_name in self.models}

        # Evaluate each model using cross-validation
        for model_name, model in self.models.items():
            fold_results = []

            for train_idx, test_idx in cv.split(X, y):
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Apply sample weights if provided
                if sample_weights is not None:
                    train_weights = sample_weights[train_idx]
                    test_weights = sample_weights[test_idx]

                    # Fit model with weights
                    model.fit(X_train, y_train, sample_weight=train_weights)
                else:
                    # Fit model without weights
                    model.fit(X_train, y_train)

                # Make predictions
                if task_type == "classification" and hasattr(model, "predict_proba"):
                    y_pred = model.predict_proba(X_test)
                    if y_pred.shape[1] == 2:  # Binary classification
                        y_pred = y_pred[:, 1]
                else:
                    y_pred = model.predict(X_test)

                # Calculate metrics
                fold_metrics = calculate_metrics(
                    y_test,
                    y_pred,
                    sample_weights=test_weights if sample_weights is not None else None,
                    task_type=task_type,
                )

                fold_results.append(fold_metrics)

            # Average metrics across folds
            avg_metrics = {}
            for metric in fold_results[0].keys():
                avg_metrics[metric] = np.mean([fold[metric] for fold in fold_results])
                avg_metrics[f"{metric}_std"] = np.std(
                    [fold[metric] for fold in fold_results]
                )

            self.results[model_name] = avg_metrics

        # Convert results to DataFrame
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

        # Get baseline model
        baseline = self.models[baseline_model]

        # Initialize results
        comparison_results = {}

        # Compare each model to baseline
        for model_name, model in self.models.items():
            if model_name == baseline_model:
                continue

            # Perform t-test on predictions
            t_test_result = self.statistical_tests.paired_t_test(baseline, model, X, y)

            # Perform permutation test
            perm_test_result = self.statistical_tests.permutation_test(
                baseline, model, X, y, n_permutations=n_permutations
            )

            # Store results
            comparison_results[model_name] = {
                "t_statistic": t_test_result[0],
                "t_p_value": t_test_result[1],
                "t_significant": t_test_result[1] < alpha,
                "perm_p_value": perm_test_result,
                "perm_significant": perm_test_result < alpha,
            }

        # Convert to DataFrame
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

        # Extract metric values
        metric_values = {
            model: results[metric] for model, results in self.results.items()
        }
        metric_std = {
            model: results.get(f"{metric}_std", 0)
            for model, results in self.results.items()
        }

        # Convert to DataFrame
        df = pd.DataFrame({"value": metric_values, "std": metric_std})

        # Sort if requested
        if sort_values:
            df = df.sort_values("value")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot bars
        bars = ax.barh(df.index, df["value"], xerr=df["std"], capsize=5)

        # Add labels and title
        ax.set_xlabel(metric.upper())
        ax.set_ylabel("Model")
        ax.set_title(title or f"Model Comparison - {metric.upper()}")

        # Add values to bars
        for i, bar in enumerate(bars):
            ax.text(
                bar.get_width() + df["std"].iloc[i] + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{df['value'].iloc[i]:.4f}",
                va="center",
            )

        # Adjust layout
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

        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results).T

        # Filter out std columns
        metric_cols = [col for col in results_df.columns if not col.endswith("_std")]
        results_df = results_df[metric_cols]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        im = ax.imshow(results_df.values, cmap=cmap)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(metric_cols)))
        ax.set_yticks(np.arange(len(results_df.index)))
        ax.set_xticklabels(metric_cols)
        ax.set_yticklabels(results_df.index)

        # Rotate x tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add values to cells
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

        # Add title
        ax.set_title(title or "Model Performance Metrics Heatmap")

        # Adjust layout
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
        # Get predictions
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)

        # Calculate errors
        if metric_fn is None:
            # Default to squared error
            errors1 = (y - y_pred1) ** 2
            errors2 = (y - y_pred2) ** 2
        else:
            # Use provided metric function
            errors1 = np.array([metric_fn(y[i], y_pred1[i]) for i in range(len(y))])
            errors2 = np.array([metric_fn(y[i], y_pred2[i]) for i in range(len(y))])

        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(errors1, errors2)

        return t_statistic, p_value

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

        # Set random state
        rng = np.random.RandomState(random_state)

        # Get predictions
        y_pred1 = model1.predict(X)
        y_pred2 = model2.predict(X)

        # Calculate metric
        if metric_fn is None:
            # Default to mean squared error
            metric1 = mean_squared_error(y, y_pred1)
            metric2 = mean_squared_error(y, y_pred2)
        else:
            # Use provided metric function
            metric1 = metric_fn(y, y_pred1)
            metric2 = metric_fn(y, y_pred2)

        # Calculate observed difference
        observed_diff = abs(metric1 - metric2)

        # Perform permutation test
        count = 0
        for _ in range(n_permutations):
            # Permute predictions
            permutation = rng.permutation(len(y))
            y_perm1 = y_pred1[permutation]
            y_perm2 = y_pred2[permutation]

            # Calculate metric on permuted predictions
            if metric_fn is None:
                perm_metric1 = mean_squared_error(y, y_perm1)
                perm_metric2 = mean_squared_error(y, y_perm2)
            else:
                perm_metric1 = metric_fn(y, y_perm1)
                perm_metric2 = metric_fn(y, y_perm2)

            # Calculate permuted difference
            perm_diff = abs(perm_metric1 - perm_metric2)

            # Count if permuted difference is greater than observed
            if perm_diff >= observed_diff:
                count += 1

        # Calculate p-value
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
        # Calculate errors
        e1 = y_true - y_pred1
        e2 = y_true - y_pred2

        # Calculate loss differential
        if loss_fn is None:
            # Default to squared error
            d = e1**2 - e2**2
        else:
            # Use provided loss function
            d = np.array(
                [
                    loss_fn(y_true[i], y_pred1[i]) - loss_fn(y_true[i], y_pred2[i])
                    for i in range(len(y_true))
                ]
            )

        # Calculate mean of loss differential
        d_bar = np.mean(d)

        # Calculate autocovariance of loss differential
        n = len(d)
        gamma_0 = np.sum((d - d_bar) ** 2) / n

        # Calculate sum of autocovariances
        gamma_sum = 0
        for i in range(1, h):
            gamma_i = np.sum((d[i:] - d_bar) * (d[:-i] - d_bar)) / n
            gamma_sum += gamma_i

        # Calculate variance of loss differential
        var_d = gamma_0 + 2 * gamma_sum

        # Calculate Diebold-Mariano statistic
        dm_statistic = d_bar / np.sqrt(var_d / n)

        # Calculate p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_statistic)))

        return dm_statistic, p_value

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
        from sklearn.metrics import mean_squared_error

        # Set random state
        rng = np.random.RandomState(random_state)

        # Get predictions for all models
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X)

        # Calculate losses
        losses = {}
        for name, pred in predictions.items():
            if loss_fn is None:
                # Default to squared error
                losses[name] = (y - pred) ** 2
            else:
                # Use provided loss function
                losses[name] = np.array([loss_fn(y[i], pred[i]) for i in range(len(y))])

        # Initialize model set
        model_set = list(models.keys())

        # Perform Model Confidence Set procedure
        while len(model_set) > 1:
            # Calculate loss differences
            loss_diffs = {}
            for i, model_i in enumerate(model_set):
                for j, model_j in enumerate(model_set):
                    if i < j:
                        loss_diffs[(model_i, model_j)] = (
                            losses[model_i] - losses[model_j]
                        )

            # Calculate test statistics
            t_stats = {}
            for (model_i, model_j), diff in loss_diffs.items():
                t_stats[(model_i, model_j)] = np.mean(diff) / (
                    np.std(diff) / np.sqrt(len(diff))
                )

            # Find maximum absolute t-statistic
            max_t = max(abs(t) for t in t_stats.values())

            # Bootstrap distribution of maximum t-statistic
            max_t_boot = []
            for _ in range(B):
                # Generate bootstrap sample
                boot_idx = rng.choice(len(y), size=len(y), replace=True)

                # Calculate bootstrap loss differences
                boot_diffs = {}
                for (model_i, model_j), diff in loss_diffs.items():
                    boot_diffs[(model_i, model_j)] = diff[boot_idx]

                # Calculate bootstrap t-statistics
                boot_t_stats = {}
                for (model_i, model_j), diff in boot_diffs.items():
                    boot_t_stats[(model_i, model_j)] = np.mean(diff) / (
                        np.std(diff) / np.sqrt(len(diff))
                    )

                # Find maximum absolute bootstrap t-statistic
                max_t_boot.append(max(abs(t) for t in boot_t_stats.values()))

            # Calculate p-value
            p_value = np.mean(np.array(max_t_boot) > max_t)

            # Check if we should eliminate a model
            if p_value < alpha:
                # Find model with worst performance
                model_losses = {model: np.mean(losses[model]) for model in model_set}
                worst_model = max(model_losses, key=model_losses.get)

                # Remove worst model
                model_set.remove(worst_model)
            else:
                # No more eliminations
                break

        return model_set
