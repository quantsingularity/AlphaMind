"""
Model explainability tools for financial machine learning models.

This module provides tools for explaining and interpreting machine learning models
in financial applications, including feature importance, partial dependence plots,
SHAP values, and permutation importance.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance as sk_permutation_importance


class FeatureImportance:
    """
    Feature importance analysis for machine learning models.

    This class provides methods for extracting and visualizing feature
    importance from various types of machine learning models.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    feature_names : list, optional
        List of feature names. If None, uses X0, X1, etc.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names
        self.importance_values = None

    def extract_importance(self, method: str = "auto") -> pd.DataFrame:
        """
        Extract feature importance from the model.

        Parameters
        ----------
        method : str, default="auto"
            Method to extract feature importance:
            - "auto": Automatically detect based on model type
            - "native": Use model's built-in feature_importances_
            - "coefficients": Use model coefficients (linear models)
            - "permutation": Use permutation importance (requires separate class)
            - "shap": Use SHAP values (requires separate class)

        Returns
        -------
        importance_df : DataFrame
            DataFrame containing feature importance values.
        """
        if method == "auto":
            if hasattr(self.model, "feature_importances_"):
                method = "native"
            elif hasattr(self.model, "coef_"):
                method = "coefficients"
            else:
                raise ValueError(
                    "Could not automatically detect feature importance method. Please specify method explicitly or use PermutationImportance/ShapExplainer."
                )
        if method == "native":
            if not hasattr(self.model, "feature_importances_"):
                raise ValueError("Model does not have feature_importances_ attribute")
            importance = self.model.feature_importances_
        elif method == "coefficients":
            if not hasattr(self.model, "coef_"):
                raise ValueError("Model does not have coef_ attribute")
            if len(self.model.coef_.shape) == 1:
                importance = np.abs(self.model.coef_)
            else:
                importance = np.mean(np.abs(self.model.coef_), axis=0)
        elif method in ["permutation", "shap"]:
            raise ValueError(
                f"Method '{method}' requires separate class: use PermutationImportance or ShapExplainer."
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        if self.feature_names is None:
            self.feature_names = [f"X{i}" for i in range(len(importance))]
        if len(self.feature_names) != len(importance):
            raise ValueError(
                f"Length of feature_names ({len(self.feature_names)}) does not match length of importance values ({len(importance)})"
            )
        importance_df = pd.DataFrame(
            {"Feature": self.feature_names, "Importance": importance}
        )
        importance_df = importance_df.sort_values("Importance", ascending=False)
        self.importance_values = importance_df
        return importance_df

    def plot(
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
        if self.importance_values is None:
            raise ValueError(
                "No importance values available. Run extract_importance() or use the PermutationImportance class."
            )
        if top_n is not None:
            df = self.importance_values.head(top_n)
        else:
            df = self.importance_values
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df["Feature"], df["Importance"], color=color)
        if "Std" in df.columns:
            df_reordered = df.sort_values("Importance", ascending=False)
            ax.errorbar(
                df_reordered["Importance"],
                df_reordered["Feature"],
                xerr=df_reordered["Std"],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
        if show_values:
            if not df.empty and df["Importance"].abs().max() > 0:
                max_importance = df["Importance"].abs().max()
            else:
                max_importance = 1.0
            for bar in bars:
                ax.text(
                    bar.get_width() + 0.01 * max_importance,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}",
                    va="center",
                )
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title(title or "Feature Importance")
        plt.tight_layout()
        return fig


class PartialDependence:
    """
    Partial dependence analysis for machine learning models.

    This class provides methods for calculating and visualizing
    partial dependence plots for machine learning models.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    feature_names : list, optional
        List of feature names. If None, uses X0, X1, etc.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names
        self.pdp_results = {}

    def compute(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        features: Union[List[int], List[str], List[Tuple[str, str]]],
        grid_resolution: int = 50,
        percentiles: Tuple[float, float] = (0.05, 0.95),
        method: str = "auto",
    ) -> Dict:
        """
        Compute partial dependence for specified features.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        features : list
            List of feature indices or names (for 1D PDP) or a list of tuples
            of feature names (for 2D PDP/interaction).
        grid_resolution : int, default=50
            Number of points in the grid.
        percentiles : tuple, default=(0.05, 0.95)
            Lower and upper percentiles to consider for the grid.
        method : str, default="auto"
            Method to compute partial dependence:
            - "auto": Automatically select method
            - "brute": Use brute force method
            - "recursion": Use recursion method (for tree-based models)

        Returns
        -------
        pdp_results : dict
            Dictionary containing partial dependence results. Keys are feature names or
            "feature1_feature2" for interactions.
        """
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_np = X.values
        else:
            X_np = X
            if self.feature_names is None:
                self.feature_names = [f"X{i}" for i in range(X.shape[1])]

        def map_features_to_indices(f):
            if isinstance(f, str):
                return self.feature_names.index(f)
            elif isinstance(f, int):
                return f
            elif isinstance(f, tuple) and len(f) == 2:
                return (self.feature_names.index(f[0]), self.feature_names.index(f[1]))
            else:
                raise ValueError(f"Invalid feature specification: {f}")

        feature_indices_for_compute: List[Any] = []
        for f in features:
            if isinstance(f, tuple):
                feature_indices_for_compute.append(
                    (map_features_to_indices(f[0]), map_features_to_indices(f[1]))
                )
            else:
                feature_indices_for_compute.append(map_features_to_indices(f))
        for feature, idxs in zip(features, feature_indices_for_compute):
            if isinstance(feature, str):
                result_key = feature
            elif isinstance(feature, int):
                result_key = self.feature_names[feature]
            elif isinstance(feature, tuple):
                result_key = f"{feature[0]}_{feature[1]}"
            pdp_result = partial_dependence(
                self.model,
                X_np,
                [idxs],
                grid_resolution=grid_resolution,
                percentiles=percentiles,
                method=method,
            )
            if isinstance(idxs, int):
                self.pdp_results[result_key] = {
                    "values": pdp_result["values"][0],
                    "predictions": pdp_result["average"][0],
                }
            elif isinstance(idxs, tuple):
                self.pdp_results[result_key] = {
                    "values": [pdp_result["values"][0], pdp_result["values"][1]],
                    "predictions": pdp_result["average"][0],
                }
        return self.pdp_results

    def plot(
        self,
        features: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        ncols: int = 2,
        title: Optional[str] = None,
        line_color: str = "blue",
        fill_color: str = "lightblue",
    ) -> plt.Figure:
        """
                Plot 1D partial dependence for specified features.

                This generates plots similar to:
                #

        [Image of a Partial Dependence Plot]


                Parameters
                ----------
                features : list, optional
                    List of 1D feature names to plot. If None, plots all computed 1D features.
                figsize : tuple, default=(12, 8)
                    Figure size.
                ncols : int, default=2
                    Number of columns in the plot grid.
                title : str, optional
                    Plot title. If None, uses default title.
                line_color : str, default="blue"
                    Line color.
                fill_color : str, default="lightblue"
                    Fill color for confidence interval.

                Returns
                -------
                fig : Figure
                    Matplotlib figure.
        """
        if not self.pdp_results:
            raise ValueError(
                "No partial dependence results available. Run compute() first."
            )
        one_d_pdp_results = {
            k: v
            for k, v in self.pdp_results.items()
            if isinstance(v["values"], np.ndarray) and v["values"].ndim == 1
        }
        if features is None:
            features = list(one_d_pdp_results.keys())
        features = [f for f in features if f in one_d_pdp_results]
        if not features:
            raise ValueError("No valid 1D features specified for plotting.")
        n_features = len(features)
        nrows = (n_features + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, feature in enumerate(features):
            ax = axes[i]
            results = one_d_pdp_results[feature]
            values = results["values"]
            predictions = results["predictions"]
            ax.plot(values, predictions, color=line_color, linewidth=2)
            ax.set_xlabel(feature)
            ax.set_ylabel("Partial Dependence")
            ax.set_title(f"Partial Dependence of {feature}")
            ax.grid(True, linestyle="--", alpha=0.7)
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        if title:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=0.9)
        return fig

    def plot_interaction(
        self,
        feature_pair: Tuple[str, str],
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "viridis",
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 2D partial dependence for a pair of features.

        The corresponding interaction must have been computed by the `compute` method.

        Parameters
        ----------
        feature_pair : tuple
            Tuple of two feature names (e.g., ("featureA", "featureB")).
        figsize : tuple, default=(10, 8)
            Figure size.
        cmap : str, default="viridis"
            Colormap for contour plot.
        title : str, optional
            Plot title. If None, uses default title.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        if not self.pdp_results:
            raise ValueError(
                "No partial dependence results available. Run compute() first."
            )
        feature1, feature2 = feature_pair
        interaction_key_fwd = f"{feature1}_{feature2}"
        interaction_key_rev = f"{feature2}_{feature1}"
        if interaction_key_fwd in self.pdp_results:
            interaction_key = interaction_key_fwd
        elif interaction_key_rev in self.pdp_results:
            interaction_key = interaction_key_rev
            feature1, feature2 = (feature2, feature1)
        else:
            raise ValueError(
                f"Interaction for {feature_pair} (and reverse) not computed. Ensure you called compute() with the tuple."
            )
        results = self.pdp_results[interaction_key]
        if not (
            isinstance(results["values"], list)
            and len(results["values"]) == 2
            and (results["predictions"].ndim == 2)
        ):
            raise ValueError(
                f"Stored result for key {interaction_key} is not a 2D partial dependence result."
            )
        values1 = results["values"][0]
        values2 = results["values"][1]
        predictions = results["predictions"]
        X1, X2 = np.meshgrid(values1, values2)
        fig, ax = plt.subplots(figsize=figsize)
        contour = ax.contourf(X1, X2, predictions.T, cmap=cmap)
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label("Partial Dependence")
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(
            title or f"Partial Dependence Interaction: {feature1} vs {feature2}"
        )
        plt.tight_layout()
        return fig


class ShapExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretability.

    This class provides methods for calculating and visualizing SHAP values
    for machine learning models.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    feature_names : list, optional
        List of feature names. If None, uses X0, X1, etc.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.data = None

    def compute_shap(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        algorithm: str = "auto",
        n_background: int = 100,
        random_state: Optional[int] = None,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compute SHAP values for the given data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        algorithm : str, default="auto"
            SHAP algorithm to use:
            - "auto": Automatically select algorithm
            - "tree": Use TreeExplainer (for tree-based models)
            - "linear": Use LinearExplainer (for linear models)
            - "kernel": Use KernelExplainer (model-agnostic)
            - "deep": Use DeepExplainer (for deep learning models)
            - "gradient": Use GradientExplainer (for deep learning models)
        n_background : int, default=100
            Number of background samples for KernelExplainer.
        random_state : int, optional
            Random state for reproducibility.

        Returns
        -------
        shap_values : ndarray or list of ndarray
            SHAP values. List of arrays for multiclass output.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP package is required for SHAP explanations. Install it with: pip install shap"
            )
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X_data = X
            X_np = X.values
        else:
            if self.feature_names is None:
                self.feature_names = [f"X{i}" for i in range(X.shape[1])]
            X_data = pd.DataFrame(X, columns=self.feature_names)
            X_np = X
        self.data = X_data
        if random_state is not None:
            np.random.seed(random_state)
        if algorithm == "auto":
            is_tree = hasattr(self.model, "estimators_") or hasattr(self.model, "tree_")
            is_linear = hasattr(self.model, "coef_") and (not is_tree)
            is_deep = False
            if is_tree:
                algorithm = "tree"
            elif is_linear:
                algorithm = "linear"
            elif is_deep:
                algorithm = "gradient"
            else:
                algorithm = "kernel"
        if algorithm == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif algorithm == "linear":
            self.explainer = shap.LinearExplainer(self.model, X_np)
        elif algorithm == "kernel":
            if X_np.shape[0] > 0:
                background_samples = shap.sample(
                    X_np, n_background, random_state=random_state
                )
            else:
                raise ValueError("X must not be empty for KernelExplainer.")
            self.explainer = shap.KernelExplainer(
                self.model.predict, background_samples
            )
        elif algorithm in ["deep", "gradient"]:
            background = X_np
            if algorithm == "deep":
                self.explainer = shap.DeepExplainer(self.model, background)
            else:
                self.explainer = shap.GradientExplainer(self.model, background)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        if algorithm in ["kernel", "deep", "gradient"]:
            self.shap_values = self.explainer.shap_values(X_np)
        else:
            self.shap_values = self.explainer.shap_values(X_np)
        return self.shap_values

    def plot_summary(
        self,
        max_display: int = 20,
        plot_type: str = "bar",
        class_index: Optional[int] = None,
    ) -> None:
        """
        Plot SHAP summary.

        Parameters
        ----------
        max_display : int, default=20
            Maximum number of features to display.
        plot_type : str, default="bar"
            Type of summary plot:
            - "bar": Bar plot of mean absolute SHAP values
            - "dot": Dot/Beeswarm plot of SHAP values
            - "violin": Violin plot of SHAP values
        class_index : int, optional
            Class index for multiclass classification. If None, uses first class.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP package is required for SHAP explanations. Install it with: pip install shap"
            )
        if self.shap_values is None or self.data is None:
            raise ValueError("No SHAP values available. Run compute_shap() first.")
        if isinstance(self.shap_values, list):
            if class_index is None:
                class_index = 0
            if class_index >= len(self.shap_values):
                raise ValueError(f"Class index {class_index} out of range")
            shap_values = self.shap_values[class_index]
        else:
            shap_values = self.shap_values
        shap_data = self.data
        if plot_type == "bar":
            shap.summary_plot(
                shap_values,
                shap_data,
                feature_names=self.feature_names,
                max_display=max_display,
                plot_type="bar",
                show=False,
            )
        elif plot_type in ["dot", "violin", "beeswarm"]:
            shap.summary_plot(
                shap_values,
                shap_data,
                feature_names=self.feature_names,
                max_display=max_display,
                plot_type="dot" if plot_type in ["dot", "beeswarm"] else "violin",
                show=False,
            )
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        plt.tight_layout()
        plt.show()

    def plot_dependence(
        self,
        feature: Union[int, str],
        interaction_index: Optional[Union[int, str]] = "auto",
    ) -> None:
        """
                Plot SHAP dependence plot.

                This generates plots similar to:
                #

        [Image of a SHAP Dependence Plot]


                Parameters
                ----------
                feature : int or str
                    Feature index or name to plot.
                interaction_index : int, str, or "auto", optional
                    Feature to use for coloring. If "auto", uses feature with highest interaction.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP package is required for SHAP explanations. Install it with: pip install shap"
            )
        if self.shap_values is None or self.data is None:
            raise ValueError("No SHAP values available. Run compute_shap() first.")
        if isinstance(self.shap_values, list):
            shap_values = self.shap_values[0]
        else:
            shap_values = self.shap_values
        shap.dependence_plot(
            feature,
            shap_values,
            self.data,
            feature_names=self.feature_names,
            interaction_index=interaction_index,
            show=False,
        )
        plt.tight_layout()
        plt.show()

    def plot_force(
        self, sample_index: int = 0, class_index: Optional[int] = None
    ) -> None:
        """
                Plot SHAP force plot for a single prediction.

                This generates plots similar to:
                #

        [Image of a SHAP Force Plot]


                Parameters
                ----------
                sample_index : int, default=0
                    Index of the sample to explain.
                class_index : int, optional
                    Class index for multiclass classification. If None, uses first class.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP package is required for SHAP explanations. Install it with: pip install shap"
            )
        if self.shap_values is None or self.explainer is None or self.data is None:
            raise ValueError(
                "No SHAP values or explainer available. Run compute_shap() first."
            )
        if isinstance(self.shap_values, list):
            if class_index is None:
                class_index = 0
            if class_index >= len(self.shap_values):
                raise ValueError(f"Class index {class_index} out of range")
            shap_values = self.shap_values[class_index]
            if (
                isinstance(self.explainer.expected_value, list)
                or getattr(self.explainer.expected_value, "ndim", 0) > 0
            ):
                expected_value = self.explainer.expected_value[class_index]
            else:
                expected_value = self.explainer.expected_value
        else:
            shap_values = self.shap_values
            expected_value = self.explainer.expected_value
        if isinstance(self.data, pd.DataFrame):
            sample_data = self.data.iloc[sample_index]
        else:
            sample_data = self.data[sample_index]
        shap.force_plot(
            expected_value,
            shap_values[sample_index],
            sample_data,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.show()


class PermutationImportance:
    """
    Permutation importance for model interpretability.

    This class provides methods for calculating and visualizing
    permutation importance for machine learning models.

    Parameters
    ----------
    model : object
        Trained machine learning model.
    feature_names : list, optional
        List of feature names. If None, uses X0, X1, etc.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        self.model = model
        self.feature_names = feature_names
        self.importance_values = None

    def compute(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: Optional[Union[str, Callable]] = None,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate permutation importance.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        scoring : str or callable, optional
            Scoring metric. If None, uses R^2 for regression and accuracy for classification.
        n_repeats : int, default=10
            Number of times to permute each feature.
        random_state : int, optional
            Random state for reproducibility.
        n_jobs : int, optional
            Number of jobs to run in parallel.

        Returns
        -------
        importance_df : DataFrame
            DataFrame containing permutation importance values.
        """
        perm_importance = sk_permutation_importance(
            self.model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"X{i}" for i in range(X.shape[1])]
        importance_df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance": perm_importance.importances_mean,
                "Std": perm_importance.importances_std,
            }
        )
        importance_df = importance_df.sort_values("Importance", ascending=False)
        self.importance_values = importance_df
        return importance_df

    def plot(
        self,
        top_n: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        color: str = "skyblue",
        show_values: bool = True,
    ) -> plt.Figure:
        """
                Plot permutation importance.

                This generates plots similar to:
                #

        [Image of a Permutation Importance Plot]


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
        if self.importance_values is None:
            raise ValueError("No importance values available. Run compute() first.")
        if top_n is not None:
            df = self.importance_values.head(top_n)
        else:
            df = self.importance_values
        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(df["Feature"], df["Importance"], color=color)
        ax.errorbar(
            df["Importance"],
            df["Feature"],
            xerr=df["Std"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
        if show_values:
            if not df.empty and df["Importance"].abs().max() > 0:
                max_importance = df["Importance"].abs().max()
            else:
                max_importance = 1.0
            for bar in bars:
                ax.text(
                    bar.get_width() + 0.01 * max_importance,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.4f}",
                    va="center",
                )
        ax.set_xlabel("Importance (Mean Decrease in Score)")
        ax.set_ylabel("Feature")
        ax.set_title(title or "Permutation Feature Importance")
        plt.tight_layout()
        return fig
