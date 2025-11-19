#""""""
## Model explainability tools for financial machine learning models.
#
## This module provides tools for explaining and interpreting machine learning models
## in financial applications, including feature importance, partial dependence plots,
## SHAP values, and permutation importance.
#""""""

# from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.inspection import partial_dependence
# from sklearn.inspection import permutation_importance as sk_permutation_importance


# class FeatureImportance:
#    """"""
##     Feature importance analysis for machine learning models.
#
##     This class provides methods for extracting and visualizing feature
##     importance from various types of machine learning models.
#
##     Parameters
#    ----------
##     model : object
##         Trained machine learning model.
##     feature_names : list, optional
##         List of feature names. If None, uses X0, X1, etc.
#    """"""

#     def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
#         self.model = model
#         self.feature_names = feature_names
#         self.importance_values = None

#     def extract_importance(self, method: str = "auto") -> pd.DataFrame:
#        """"""
##         Extract feature importance from the model.
#
##         Parameters
#        ----------
##         method : str, default="auto"
##             Method to extract feature importance:
##             - "auto": Automatically detect based on model type
##             - "native": Use model's built-in feature_importances_
##             - "coefficients": Use model coefficients (linear models)
##             - "permutation": Use permutation importance (requires X and y)
##             - "shap": Use SHAP values (requires additional call to compute_shap)
#
##         Returns
#        -------
##         importance_df : DataFrame
##             DataFrame containing feature importance values.
#        """"""
#         if method == "auto":
            # Try to automatically detect the appropriate method
#             if hasattr(self.model, "feature_importances_"):
#                 method = "native"
#             elif hasattr(self.model, "coef_"):
#                 method = "coefficients"
#             else:
#                 raise ValueError(
                    "Could not automatically detect feature importance method. "
                    "Please specify method explicitly."
                )

        # Extract importance based on method
#         if method == "native":
#             if not hasattr(self.model, "feature_importances_"):
#                 raise ValueError("Model does not have feature_importances_ attribute")

#             importance = self.model.feature_importances_

#         elif method == "coefficients":
#             if not hasattr(self.model, "coef_"):
#                 raise ValueError("Model does not have coef_ attribute")

            # Handle different coefficient shapes
#             if len(self.model.coef_.shape) == 1:
                # Binary classification or regression
#                 importance = np.abs(self.model.coef_)
#             else:
                # Multiclass classification
#                 importance = np.mean(np.abs(self.model.coef_), axis=0)

#         else:
#             raise ValueError(f"Unsupported method: {method}")

        # Create feature names if not provided
#         if self.feature_names is None:
#             self.feature_names = [f"X{i}" for i in range(len(importance))]

        # Ensure feature names match importance length
#         if len(self.feature_names) != len(importance):
#             raise ValueError(
#                 f"Length of feature_names ({len(self.feature_names)}) does not match "
#                 f"length of importance values ({len(importance)})"
            )

        # Create DataFrame
#         importance_df = pd.DataFrame(
#             {"Feature": self.feature_names, "Importance": importance}
        )

        # Sort by importance
#         importance_df = importance_df.sort_values("Importance", ascending=False)

        # Store importance values
#         self.importance_values = importance_df

#         return importance_df

#     def permutation_importance(
#         self,
#         X: Union[np.ndarray, pd.DataFrame],
#         y: Union[np.ndarray, pd.Series],
#         scoring: Optional[Union[str, Callable]] = None,
#         n_repeats: int = 10,
#         random_state: Optional[int] = None,
#         n_jobs: Optional[int] = None,
#     ) -> pd.DataFrame:
#        """"""
##         Calculate permutation importance.
#
##         Parameters
#        ----------
##         X : array-like
##             Feature matrix.
##         y : array-like
##             Target vector.
##         scoring : str or callable, optional
##             Scoring metric. If None, uses R^2 for regression and accuracy for classification.
##         n_repeats : int, default=10
##             Number of times to permute each feature.
##         random_state : int, optional
##             Random state for reproducibility.
##         n_jobs : int, optional
##             Number of jobs to run in parallel.
#
##         Returns
#        -------
##         importance_df : DataFrame
##             DataFrame containing permutation importance values.
#        """"""
        # Calculate permutation importance
#         perm_importance = sk_permutation_importance(
#             self.model,
            X,
            y,
#             scoring=scoring,
#             n_repeats=n_repeats,
#             random_state=random_state,
#             n_jobs=n_jobs,
        )

        # Create feature names if not provided
#         if self.feature_names is None:
#             self.feature_names = [f"X{i}" for i in range(X.shape[1])]

        # Create DataFrame
#         importance_df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance": perm_importance.importances_mean,
                "Std": perm_importance.importances_std,
            }
        )

        # Sort by importance
#         importance_df = importance_df.sort_values("Importance", ascending=False)

        # Store importance values
#         self.importance_values = importance_df

#         return importance_df

#     def plot(
#         self,
#         top_n: Optional[int] = None,
#         figsize: Tuple[int, int] = (10, 6),
#         title: Optional[str] = None,
#         color: str = "skyblue",
#         show_values: bool = True,
#     ) -> plt.Figure:
#        """"""
##         Plot feature importance.
#
##         Parameters
#        ----------
##         top_n : int, optional
##             Number of top features to plot. If None, plots all features.
##         figsize : tuple, default=(10, 6)
##             Figure size.
##         title : str, optional
##             Plot title. If None, uses default title.
##         color : str, default="skyblue"
##             Bar color.
##         show_values : bool, default=True
##             Whether to show importance values on bars.
#
##         Returns
#        -------
##         fig : Figure
##             Matplotlib figure.
#        """"""
#         if self.importance_values is None:
#             raise ValueError(
                "No importance values available. Run extract_importance() first."
            )

        # Select top N features if specified
#         if top_n is not None:
#             df = self.importance_values.head(top_n)
#         else:
#             df = self.importance_values

        # Create figure
#         fig, ax = plt.subplots(figsize=figsize)

        # Plot bars
#         bars = ax.barh(df["Feature"], df["Importance"], color=color)

        # Add error bars if available
#         if "Std" in df.columns:
#             ax.errorbar(
#                 df["Importance"],
#                 df["Feature"],
#                 xerr=df["Std"],
#                 fmt="none",
#                 ecolor="black",
#                 capsize=3,
            )

        # Add values to bars if requested
#         if show_values:
#             for bar in bars:
#                 ax.text(
#                     bar.get_width() + (0.01 * max(df["Importance"])),
#                     bar.get_y() + bar.get_height() / 2,
#                     f"{bar.get_width():.4f}",
#                     va="center",
                )

        # Add labels and title
#         ax.set_xlabel("Importance")
#         ax.set_ylabel("Feature")
#         ax.set_title(title or "Feature Importance")

        # Adjust layout
#         plt.tight_layout()

#         return fig


# class PartialDependence:
#    """"""
##     Partial dependence analysis for machine learning models.
#
##     This class provides methods for calculating and visualizing
##     partial dependence plots for machine learning models.
#
##     Parameters
#    ----------
##     model : object
##         Trained machine learning model.
##     feature_names : list, optional
##         List of feature names. If None, uses X0, X1, etc.
#    """"""

#     def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
#         self.model = model
#         self.feature_names = feature_names
#         self.pdp_results = {}

#     def compute(
#         self,
#         X: Union[np.ndarray, pd.DataFrame],
#         features: Union[List[int], List[str]],
#         grid_resolution: int = 50,
#         percentiles: Tuple[float, float] = (0.05, 0.95),
#         method: str = "auto",
#     ) -> Dict:
#        """"""
##         Compute partial dependence for specified features.
#
##         Parameters
#        ----------
##         X : array-like
##             Feature matrix.
##         features : list
##             List of feature indices or names to compute PDP for.
##         grid_resolution : int, default=50
##             Number of points in the grid.
##         percentiles : tuple, default=(0.05, 0.95)
##             Lower and upper percentiles to consider for the grid.
##         method : str, default="auto"
##             Method to compute partial dependence:
##             - "auto": Automatically select method
##             - "brute": Use brute force method
##             - "recursion": Use recursion method (for tree-based models)
#
##         Returns
#        -------
##         pdp_results : dict
##             Dictionary containing partial dependence results.
#        """"""
        # Convert feature names to indices if necessary
#         if self.feature_names is not None and isinstance(features[0], str):
#             feature_indices = [self.feature_names.index(f) for f in features]
#         else:
#             feature_indices = features

            # Create feature names if not provided
#             if self.feature_names is None:
#                 self.feature_names = [f"X{i}" for i in range(X.shape[1])]

        # Compute partial dependence for each feature
#         for idx in feature_indices:
            # Get feature name
#             if isinstance(idx, int):
#                 feature_name = self.feature_names[idx]
#             else:
#                 feature_name = idx
#                 idx = self.feature_names.index(feature_name)

            # Compute partial dependence
#             pdp_result = partial_dependence(
#                 self.model,
                X,
#                 [idx],
#                 grid_resolution=grid_resolution,
#                 percentiles=percentiles,
#                 method=method,
            )

            # Store results
#             self.pdp_results[feature_name] = {
                "values": pdp_result["values"][0],
                "predictions": pdp_result["average"][0],
            }

#         return self.pdp_results

#     def plot(
#         self,
#         features: Optional[List[str]] = None,
#         figsize: Tuple[int, int] = (12, 8),
#         ncols: int = 2,
#         title: Optional[str] = None,
#         line_color: str = "blue",
#         fill_color: str = "lightblue",
#     ) -> plt.Figure:
#        """"""
##         Plot partial dependence for specified features.
#
##         Parameters
#        ----------
##         features : list, optional
##             List of feature names to plot. If None, plots all computed features.
##         figsize : tuple, default=(12, 8)
##             Figure size.
##         ncols : int, default=2
##             Number of columns in the plot grid.
##         title : str, optional
##             Plot title. If None, uses default title.
##         line_color : str, default="blue"
##             Line color.
##         fill_color : str, default="lightblue"
##             Fill color for confidence interval.
#
##         Returns
#        -------
##         fig : Figure
##             Matplotlib figure.
#        """"""
#         if not self.pdp_results:
#             raise ValueError(
                "No partial dependence results available. Run compute() first."
            )

        # Use all features if not specified
#         if features is None:
#             features = list(self.pdp_results.keys())

        # Filter features that have been computed
#         features = [f for f in features if f in self.pdp_results]

#         if not features:
#             raise ValueError("No valid features specified")

        # Calculate grid layout
#         n_features = len(features)
#         nrows = (n_features + ncols - 1) // ncols

        # Create figure
#         fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Flatten axes for easy indexing
#         if nrows == 1 and ncols == 1:
#             axes = np.array([axes])
#         axes = axes.flatten()

        # Plot each feature
#         for i, feature in enumerate(features):
#             ax = axes[i]

            # Get PDP data
#             values = self.pdp_results[feature]["values"]
#             predictions = self.pdp_results[feature]["predictions"]

            # Plot PDP
#             ax.plot(values, predictions, color=line_color, linewidth=2)

            # Fill area if confidence interval is available
#             if (
                "lower" in self.pdp_results[feature]
#                 and "upper" in self.pdp_results[feature]
            ):
#                 lower = self.pdp_results[feature]["lower"]
#                 upper = self.pdp_results[feature]["upper"]
#                 ax.fill_between(values, lower, upper, color=fill_color, alpha=0.3)

            # Add labels
#             ax.set_xlabel(feature)
#             ax.set_ylabel("Partial Dependence")
#             ax.set_title(f"Partial Dependence of {feature}")

            # Add grid
#             ax.grid(True, linestyle="--", alpha=0.7)

        # Hide unused subplots
#         for i in range(n_features, len(axes)):
#             axes[i].set_visible(False)

        # Add overall title
#         if title:
#             fig.suptitle(title, fontsize=16)

        # Adjust layout
#         plt.tight_layout()
#         if title:
#             plt.subplots_adjust(top=0.9)

#         return fig

#     def plot_interaction(
#         self,
#         feature_pair: Tuple[str, str],
#         figsize: Tuple[int, int] = (10, 8),
#         cmap: str = "viridis",
#         title: Optional[str] = None,
#     ) -> plt.Figure:
#        """"""
##         Plot 2D partial dependence for a pair of features.
#
##         Parameters
#        ----------
##         feature_pair : tuple
##             Tuple of two feature names.
##         figsize : tuple, default=(10, 8)
##             Figure size.
##         cmap : str, default="viridis"
##             Colormap for contour plot.
##         title : str, optional
##             Plot title. If None, uses default title.
#
##         Returns
#        -------
##         fig : Figure
##             Matplotlib figure.
#        """"""
#         if not self.pdp_results:
#             raise ValueError(
                "No partial dependence results available. Run compute() first."
            )

        # Check if both features are available
#         feature1, feature2 = feature_pair
#         interaction_key = f"{feature1}_{feature2}"

#         if interaction_key not in self.pdp_results:
#             raise ValueError(f"Interaction for {feature1} and {feature2} not computed")

        # Get PDP data
#         values1 = self.pdp_results[interaction_key]["values"][0]
#         values2 = self.pdp_results[interaction_key]["values"][1]
#         predictions = self.pdp_results[interaction_key]["predictions"]

        # Create meshgrid
#         X1, X2 = np.meshgrid(values1, values2)

        # Create figure
#         fig, ax = plt.subplots(figsize=figsize)

        # Plot contour
#         contour = ax.contourf(X1, X2, predictions, cmap=cmap)

        # Add colorbar
#         cbar = fig.colorbar(contour, ax=ax)
#         cbar.set_label("Partial Dependence")

        # Add labels
#         ax.set_xlabel(feature1)
#         ax.set_ylabel(feature2)
#         ax.set_title(
#             title or f"Partial Dependence Interaction: {feature1} vs {feature2}"
        )

        # Adjust layout
#         plt.tight_layout()

#         return fig


# class ShapExplainer:
#    """"""
##     SHAP (SHapley Additive exPlanations) for model interpretability.
#
##     This class provides methods for calculating and visualizing SHAP values
##     for machine learning models.
#
##     Parameters
#    ----------
##     model : object
##         Trained machine learning model.
##     feature_names : list, optional
##         List of feature names. If None, uses X0, X1, etc.
#    """"""

#     def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
#         self.model = model
#         self.feature_names = feature_names
#         self.explainer = None
#         self.shap_values = None
#         self.data = None

#     def compute_shap(
#         self,
#         X: Union[np.ndarray, pd.DataFrame],
#         algorithm: str = "auto",
#         n_background: int = 100,
#         random_state: Optional[int] = None,
#     ) -> np.ndarray:
#        """"""
##         Compute SHAP values for the given data.
#
##         Parameters
#        ----------
##         X : array-like
##             Feature matrix.
##         algorithm : str, default="auto"
##             SHAP algorithm to use:
##             - "auto": Automatically select algorithm
##             - "tree": Use TreeExplainer (for tree-based models)
##             - "linear": Use LinearExplainer (for linear models)
##             - "kernel": Use KernelExplainer (model-agnostic)
##             - "deep": Use DeepExplainer (for deep learning models)
##             - "gradient": Use GradientExplainer (for deep learning models)
##         n_background : int, default=100
##             Number of background samples for KernelExplainer.
##         random_state : int, optional
##             Random state for reproducibility.
#
##         Returns
#        -------
##         shap_values : ndarray
##             SHAP values.
#        """"""
#         try:
#             import shap
#         except ImportError:
#             raise ImportError(
                "SHAP package is required for SHAP explanations. "
                "Install it with: pip install shap"
            )

        # Convert to numpy array if DataFrame
#         if isinstance(X, pd.DataFrame):
            # Store feature names if not provided
#             if self.feature_names is None:
#                 self.feature_names = X.columns.tolist()
#             X_np = X.values
#         else:
#             X_np = X

            # Create feature names if not provided
#             if self.feature_names is None:
#                 self.feature_names = [f"X{i}" for i in range(X.shape[1])]

        # Store data for later use
#         self.data = X

        # Set random state
#         if random_state is not None:
#             np.random.seed(random_state)

        # Select algorithm if auto
#         if algorithm == "auto":
#             if hasattr(self.model, "predict_proba") and hasattr(self.model, "classes_"):
                # Classification model
#                 if hasattr(self.model, "estimators_") or hasattr(self.model, "tree_"):
#                     algorithm = "tree"
#                 elif hasattr(self.model, "coef_"):
#                     algorithm = "linear"
#                 else:
#                     algorithm = "kernel"
#             else:
                # Regression model
#                 if hasattr(self.model, "estimators_") or hasattr(self.model, "tree_"):
#                     algorithm = "tree"
#                 elif hasattr(self.model, "coef_"):
#                     algorithm = "linear"
#                 else:
#                     algorithm = "kernel"

        # Create explainer based on algorithm
#         if algorithm == "tree":
#             self.explainer = shap.TreeExplainer(self.model)
#         elif algorithm == "linear":
#             self.explainer = shap.LinearExplainer(self.model, X_np)
#         elif algorithm == "kernel":
            # Select background samples
#             if n_background < X_np.shape[0]:
#                 background = shap.sample(X_np, n_background)
#             else:
#                 background = X_np

            # Create explainer
#             self.explainer = shap.KernelExplainer(self.model.predict, background)
#         elif algorithm == "deep":
#             self.explainer = shap.DeepExplainer(self.model, X_np)
#         elif algorithm == "gradient":
#             self.explainer = shap.GradientExplainer(self.model, X_np)
#         else:
#             raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Compute SHAP values
#         self.shap_values = self.explainer.shap_values(X_np)

#         return self.shap_values

#     def plot_summary(
#         self,
#         max_display: int = 20,
#         plot_type: str = "bar",
#         class_index: Optional[int] = None,
#     ) -> None:
#        """"""
##         Plot SHAP summary.
#
##         Parameters
#        ----------
##         max_display : int, default=20
##             Maximum number of features to display.
##         plot_type : str, default="bar"
##             Type of summary plot:
##             - "bar": Bar plot of mean absolute SHAP values
##             - "dot": Dot plot of SHAP values
##             - "violin": Violin plot of SHAP values
##         class_index : int, optional
##             Class index for multiclass classification. If None, uses first class.
#        """"""
#         try:
#             import shap
#         except ImportError:
#             raise ImportError(
                "SHAP package is required for SHAP explanations. "
                "Install it with: pip install shap"
            )

#         if self.shap_values is None:
#             raise ValueError("No SHAP values available. Run compute_shap() first.")

        # Handle multiclass classification
#         if isinstance(self.shap_values, list):
#             if class_index is None:
#                 class_index = 0

#             if class_index >= len(self.shap_values):
#                 raise ValueError(f"Class index {class_index} out of range")

#             shap_values = self.shap_values[class_index]
#         else:
#             shap_values = self.shap_values

        # Create feature names if not provided
#         if self.feature_names is None:
#             self.feature_names = [f"X{i}" for i in range(shap_values.shape[1])]

        # Plot summary based on type
#         if plot_type == "bar":
#             shap.summary_plot(
#                 shap_values,
#                 self.data,
#                 feature_names=self.feature_names,
#                 max_display=max_display,
#                 plot_type="bar",
            )
#         elif plot_type in ["dot", "violin"]:
#             shap.summary_plot(
#                 shap_values,
#                 self.data,
#                 feature_names=self.feature_names,
#                 max_display=max_display,
            )
#         else:
#             raise ValueError(f"Unsupported plot type: {plot_type}")

#     def plot_dependence(
#         self,
#         feature: Union[int, str],
#         interaction_index: Optional[Union[int, str]] = "auto",
#     ) -> None:
#        """"""
##         Plot SHAP dependence plot.
#
##         Parameters
#        ----------
##         feature : int or str
##             Feature index or name to plot.
##         interaction_index : int, str, or "auto", optional
##             Feature to use for coloring. If "auto", uses feature with highest interaction.
#        """"""
#         try:
#             import shap
#         except ImportError:
#             raise ImportError(
                "SHAP package is required for SHAP explanations. "
                "Install it with: pip install shap"
            )

#         if self.shap_values is None:
#             raise ValueError("No SHAP values available. Run compute_shap() first.")

        # Convert feature name to index if necessary
#         if isinstance(feature, str) and self.feature_names is not None:
#             feature_idx = self.feature_names.index(feature)
#         else:
#             feature_idx = feature

        # Convert interaction feature name to index if necessary
#         if (
#             isinstance(interaction_index, str)
#             and interaction_index != "auto"
#             and self.feature_names is not None
        ):
#             interaction_idx = self.feature_names.index(interaction_index)
#         else:
#             interaction_idx = interaction_index

        # Handle multiclass classification
#         if isinstance(self.shap_values, list):
#             shap_values = self.shap_values[0]
#         else:
#             shap_values = self.shap_values

        # Plot dependence
#         shap.dependence_plot(
#             feature_idx,
#             shap_values,
#             self.data,
#             feature_names=self.feature_names,
#             interaction_index=interaction_idx,
        )

#     def plot_force(
#         self, sample_index: int = 0, class_index: Optional[int] = None
#     ) -> None:
#        """"""
##         Plot SHAP force plot for a single prediction.
#
##         Parameters
#        ----------
##         sample_index : int, default=0
##             Index of the sample to explain.
##         class_index : int, optional
##             Class index for multiclass classification. If None, uses first class.
#        """"""
#         try:
#             import shap
#         except ImportError:
#             raise ImportError(
                "SHAP package is required for SHAP explanations. "
                "Install it with: pip install shap"
            )

#         if self.shap_values is None:
#             raise ValueError("No SHAP values available. Run compute_shap() first.")

        # Handle multiclass classification
#         if isinstance(self.shap_values, list):
#             if class_index is None:
#                 class_index = 0

#             if class_index >= len(self.shap_values):
#                 raise ValueError(f"Class index {class_index} out of range")

#             shap_values = self.shap_values[class_index]
#             expected_value = self.explainer.expected_value[class_index]
#         else:
#             shap_values = self.shap_values
#             expected_value = self.explainer.expected_value

        # Get sample data
#         if isinstance(self.data, pd.DataFrame):
#             sample_data = self.data.iloc[sample_index]
#         else:
#             sample_data = self.data[sample_index]

        # Plot force plot
#         shap.force_plot(
#             expected_value,
#             shap_values[sample_index],
#             sample_data,
#             feature_names=self.feature_names,
        )


# class PermutationImportance:
#    """"""
##     Permutation importance for model interpretability.
#
##     This class provides methods for calculating and visualizing
##     permutation importance for machine learning models.
#
##     Parameters
#    ----------
##     model : object
##         Trained machine learning model.
##     feature_names : list, optional
##         List of feature names. If None, uses X0, X1, etc.
#    """"""

#     def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
#         self.model = model
#         self.feature_names = feature_names
#         self.importance_values = None

#     def compute(
#         self,
#         X: Union[np.ndarray, pd.DataFrame],
#         y: Union[np.ndarray, pd.Series],
#         scoring: Optional[Union[str, Callable]] = None,
#         n_repeats: int = 10,
#         random_state: Optional[int] = None,
#         n_jobs: Optional[int] = None,
#     ) -> pd.DataFrame:
#        """"""
##         Compute permutation importance.
#
##         Parameters
#        ----------
##         X : array-like
##             Feature matrix.
##         y : array-like
##             Target vector.
##         scoring : str or callable, optional
##             Scoring metric. If None, uses R^2 for regression and accuracy for classification.
##         n_repeats : int, default=10
##             Number of times to permute each feature.
##         random_state : int, optional
##             Random state for reproducibility.
##         n_jobs : int, optional
##             Number of jobs to run in parallel.
#
##         Returns
#        -------
##         importance_df : DataFrame
##             DataFrame containing permutation importance values.
#        """"""
        # Calculate permutation importance
#         perm_importance = sk_permutation_importance(
#             self.model,
            X,
            y,
#             scoring=scoring,
#             n_repeats=n_repeats,
#             random_state=random_state,
#             n_jobs=n_jobs,
        )

        # Create feature names if not provided
#         if self.feature_names is None:
#             if isinstance(X, pd.DataFrame):
#                 self.feature_names = X.columns.tolist()
#             else:
#                 self.feature_names = [f"X{i}" for i in range(X.shape[1])]

        # Create DataFrame
#         importance_df = pd.DataFrame(
            {
                "Feature": self.feature_names,
                "Importance": perm_importance.importances_mean,
                "Std": perm_importance.importances_std,
            }
        )

        # Sort by importance
#         importance_df = importance_df.sort_values("Importance", ascending=False)

        # Store importance values
#         self.importance_values = importance_df

#         return importance_df

#     def plot(
#         self,
#         top_n: Optional[int] = None,
#         figsize: Tuple[int, int] = (10, 6),
#         title: Optional[str] = None,
#         color: str = "skyblue",
#         show_values: bool = True,
#     ) -> plt.Figure:
#        """"""
##         Plot permutation importance.
#
##         Parameters
#        ----------
##         top_n : int, optional
##             Number of top features to plot. If None, plots all features.
##         figsize : tuple, default=(10, 6)
##             Figure size.
##         title : str, optional
##             Plot title. If None, uses default title.
##         color : str, default="skyblue"
##             Bar color.
##         show_values : bool, default=True
##             Whether to show importance values on bars.
#
##         Returns
#        -------
##         fig : Figure
##             Matplotlib figure.
#        """"""
#         if self.importance_values is None:
#             raise ValueError("No importance values available. Run compute() first.")

        # Select top N features if specified
#         if top_n is not None:
#             df = self.importance_values.head(top_n)
#         else:
#             df = self.importance_values

        # Create figure
#         fig, ax = plt.subplots(figsize=figsize)

        # Plot bars
#         bars = ax.barh(df["Feature"], df["Importance"], color=color)

        # Add error bars
#         ax.errorbar(
#             df["Importance"],
#             df["Feature"],
#             xerr=df["Std"],
#             fmt="none",
#             ecolor="black",
#             capsize=3,
        )

        # Add values to bars if requested
#         if show_values:
#             for bar in bars:
#                 ax.text(
#                     bar.get_width() + (0.01 * max(df["Importance"])),
#                     bar.get_y() + bar.get_height() / 2,
#                     f"{bar.get_width():.4f}",
#                     va="center",
                )

        # Add labels and title
#         ax.set_xlabel("Importance")
#         ax.set_ylabel("Feature")
#         ax.set_title(title or "Permutation Importance")

        # Adjust layout
#         plt.tight_layout()

#         return fig
