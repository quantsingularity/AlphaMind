"""
Experiment definition and management for A/B testing.

This module provides classes for defining and managing experiments
in the A/B testing framework, including individual experiments and
experiment groups.
"""

import datetime
from enum import Enum
import json
import os
from typing import Any, Dict, List, Optional, Union
import uuid

import pandas as pd


class ExperimentStatus(Enum):
    """Status of an experiment."""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class Experiment:
    """
    Class representing a single A/B test experiment.

    This class provides methods for defining, running, and analyzing
    A/B test experiments in financial applications.

    Parameters
    ----------
    name : str
        Name of the experiment.
    description : str, optional
        Description of the experiment.
    variants : dict, optional
        Dictionary of experiment variants with variant names as keys.
    metrics : list, optional
        List of metrics to track.
    start_date : datetime, optional
        Start date of the experiment.
    end_date : datetime, optional
        End date of the experiment.
    """

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        variants: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[str]] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description or ""
        self.variants = variants or {"A": None, "B": None}
        self.metrics = metrics or []
        self.start_date = start_date
        self.end_date = end_date
        self.status = ExperimentStatus.CREATED
        self.results = {}
        self.metadata = {}
        self.creation_date = datetime.datetime.now()

    def add_variant(self, name: str, variant: Any) -> None:
        """
        Add a variant to the experiment.

        Parameters
        ----------
        name : str
            Name of the variant.
        variant : any
            Variant object or configuration.
        """
        self.variants[name] = variant

    def add_metric(self, metric: str) -> None:
        """
        Add a metric to track in the experiment.

        Parameters
        ----------
        metric : str
            Name of the metric.
        """
        if metric not in self.metrics:
            self.metrics.append(metric)

    def start(self, start_date: Optional[datetime.datetime] = None) -> None:
        """
        Start the experiment.

        Parameters
        ----------
        start_date : datetime, optional
            Start date of the experiment. If None, uses current time.
        """
        self.start_date = start_date or datetime.datetime.now()
        self.status = ExperimentStatus.RUNNING

    def pause(self) -> None:
        """Pause the experiment."""
        if self.status == ExperimentStatus.RUNNING:
            self.status = ExperimentStatus.PAUSED

    def resume(self) -> None:
        """Resume the experiment."""
        if self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.RUNNING

    def complete(self, end_date: Optional[datetime.datetime] = None) -> None:
        """
        Complete the experiment.

        Parameters
        ----------
        end_date : datetime, optional
            End date of the experiment. If None, uses current time.
        """
        self.end_date = end_date or datetime.datetime.now()
        self.status = ExperimentStatus.COMPLETED

    def fail(self, reason: str) -> None:
        """
        Mark the experiment as failed.

        Parameters
        ----------
        reason : str
            Reason for failure.
        """
        self.status = ExperimentStatus.FAILED
        self.metadata["failure_reason"] = reason

    def add_result(
        self,
        variant: str,
        metric: str,
        value: float,
        timestamp: Optional[datetime.datetime] = None,
    ) -> None:
        """
        Add a result for a variant and metric.

        Parameters
        ----------
        variant : str
            Name of the variant.
        metric : str
            Name of the metric.
        value : float
            Value of the metric.
        timestamp : datetime, optional
            Timestamp of the result. If None, uses current time.
        """
        if variant not in self.variants:
            raise ValueError(f"Variant '{variant}' not found in experiment")

        if metric not in self.metrics:
            raise ValueError(f"Metric '{metric}' not found in experiment")

        timestamp = timestamp or datetime.datetime.now()

        if metric not in self.results:
            self.results[metric] = {}

        if variant not in self.results[metric]:
            self.results[metric][variant] = []

        self.results[metric][variant].append({"value": value, "timestamp": timestamp})

    def add_results_batch(
        self,
        results: pd.DataFrame,
        variant_col: str = "variant",
        metric_col: str = "metric",
        value_col: str = "value",
        timestamp_col: Optional[str] = None,
    ) -> None:
        """
        Add a batch of results from a DataFrame.

        Parameters
        ----------
        results : DataFrame
            DataFrame containing results.
        variant_col : str, default="variant"
            Name of the column containing variant names.
        metric_col : str, default="metric"
            Name of the column containing metric names.
        value_col : str, default="value"
            Name of the column containing metric values.
        timestamp_col : str, optional
            Name of the column containing timestamps. If None, uses current time.
        """
        for _, row in results.iterrows():
            variant = row[variant_col]
            metric = row[metric_col]
            value = row[value_col]

            timestamp = None
            if timestamp_col and timestamp_col in row:
                timestamp = row[timestamp_col]

            self.add_result(variant, metric, value, timestamp)

    def get_results(
        self,
        metric: Optional[str] = None,
        variant: Optional[str] = None,
        as_dataframe: bool = True,
    ) -> Union[Dict, pd.DataFrame]:
        """
        Get results for the experiment.

        Parameters
        ----------
        metric : str, optional
            Name of the metric to get results for.
            If None, returns results for all metrics.
        variant : str, optional
            Name of the variant to get results for.
            If None, returns results for all variants.
        as_dataframe : bool, default=True
            Whether to return results as a DataFrame.

        Returns
        -------
        results : dict or DataFrame
            Results for the experiment.
        """
        if metric and metric not in self.results:
            raise ValueError(f"No results found for metric '{metric}'")

        if variant and not any(
            variant in self.results.get(m, {}) for m in self.results
        ):
            raise ValueError(f"No results found for variant '{variant}'")

        # Filter results by metric and variant
        filtered_results = {}

        if metric:
            metrics = [metric]
        else:
            metrics = list(self.results.keys())

        for m in metrics:
            if m not in self.results:
                continue

            filtered_results[m] = {}

            if variant:
                variants = [variant] if variant in self.results[m] else []
            else:
                variants = list(self.results[m].keys())

            for v in variants:
                filtered_results[m][v] = self.results[m][v]

        # Convert to DataFrame if requested
        if as_dataframe:
            data = []

            for m, variants in filtered_results.items():
                for v, values in variants.items():
                    for result in values:
                        data.append(
                            {
                                "metric": m,
                                "variant": v,
                                "value": result["value"],
                                "timestamp": result["timestamp"],
                            }
                        )

            return pd.DataFrame(data)

        return filtered_results

    def get_summary_statistics(
        self, metric: Optional[str] = None, variant: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get summary statistics for the experiment.

        Parameters
        ----------
        metric : str, optional
            Name of the metric to get statistics for.
            If None, returns statistics for all metrics.
        variant : str, optional
            Name of the variant to get statistics for.
            If None, returns statistics for all variants.

        Returns
        -------
        stats : DataFrame
            Summary statistics for the experiment.
        """
        # Get results as DataFrame
        results_df = self.get_results(metric, variant, as_dataframe=True)

        if results_df.empty:
            return pd.DataFrame()

        # Calculate statistics
        stats = []

        for (m, v), group in results_df.groupby(["metric", "variant"]):
            values = group["value"]

            stats.append(
                {
                    "metric": m,
                    "variant": v,
                    "count": len(values),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "25%": values.quantile(0.25),
                    "median": values.median(),
                    "75%": values.quantile(0.75),
                    "max": values.max(),
                }
            )

        return pd.DataFrame(stats)

    def to_dict(self) -> Dict:
        """
        Convert experiment to dictionary.

        Returns
        -------
        experiment_dict : dict
            Dictionary representation of the experiment.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": {k: str(v) for k, v in self.variants.items()},
            "metrics": self.metrics,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "status": self.status.value,
            "creation_date": self.creation_date.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self, include_results: bool = False) -> str:
        """
        Convert experiment to JSON.

        Parameters
        ----------
        include_results : bool, default=False
            Whether to include results in the JSON.

        Returns
        -------
        experiment_json : str
            JSON representation of the experiment.
        """
        data = self.to_dict()

        if include_results:
            # Convert results to serializable format
            results_dict = {}

            for metric, variants in self.results.items():
                results_dict[metric] = {}

                for variant, values in variants.items():
                    results_dict[metric][variant] = [
                        {"value": v["value"], "timestamp": v["timestamp"].isoformat()}
                        for v in values
                    ]

            data["results"] = results_dict

        return json.dumps(data, indent=2)

    def save(self, directory: str, include_results: bool = True) -> str:
        """
        Save experiment to file.

        Parameters
        ----------
        directory : str
            Directory to save the experiment to.
        include_results : bool, default=True
            Whether to include results in the saved file.

        Returns
        -------
        filepath : str
            Path to the saved file.
        """
        os.makedirs(directory, exist_ok=True)

        # Create filename
        filename = f"{self.id}_{self.name.replace(' ', '_')}.json"
        filepath = os.path.join(directory, filename)

        # Save to file
        with open(filepath, "w") as f:
            f.write(self.to_json(include_results=include_results))

        return filepath

    @classmethod
    def load(cls, filepath: str) -> "Experiment":
        """
        Load experiment from file.

        Parameters
        ----------
        filepath : str
            Path to the experiment file.

        Returns
        -------
        experiment : Experiment
            Loaded experiment.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Create experiment
        experiment = cls(
            name=data["name"], description=data["description"], metrics=data["metrics"]
        )

        # Set attributes
        experiment.id = data["id"]
        experiment.variants = data["variants"]

        if data["start_date"]:
            experiment.start_date = datetime.datetime.fromisoformat(data["start_date"])

        if data["end_date"]:
            experiment.end_date = datetime.datetime.fromisoformat(data["end_date"])

        experiment.status = ExperimentStatus(data["status"])
        experiment.creation_date = datetime.datetime.fromisoformat(
            data["creation_date"]
        )
        experiment.metadata = data["metadata"]

        # Load results if available
        if "results" in data:
            for metric, variants in data["results"].items():
                experiment.results[metric] = {}

                for variant, values in variants.items():
                    experiment.results[metric][variant] = [
                        {
                            "value": v["value"],
                            "timestamp": datetime.datetime.fromisoformat(
                                v["timestamp"]
                            ),
                        }
                        for v in values
                    ]

        return experiment


class ExperimentGroup:
    """
    Class representing a group of related experiments.

    This class provides methods for managing and analyzing
    multiple related experiments.

    Parameters
    ----------
    name : str
        Name of the experiment group.
    description : str, optional
        Description of the experiment group.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description or ""
        self.experiments = {}
        self.creation_date = datetime.datetime.now()

    def add_experiment(self, experiment: Experiment) -> None:
        """
        Add an experiment to the group.

        Parameters
        ----------
        experiment : Experiment
            Experiment to add.
        """
        self.experiments[experiment.id] = experiment

    def remove_experiment(self, experiment_id: str) -> None:
        """
        Remove an experiment from the group.

        Parameters
        ----------
        experiment_id : str
            ID of the experiment to remove.
        """
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]

    def get_experiment(self, experiment_id: str) -> Experiment:
        """
        Get an experiment from the group.

        Parameters
        ----------
        experiment_id : str
            ID of the experiment to get.

        Returns
        -------
        experiment : Experiment
            Experiment with the specified ID.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment with ID '{experiment_id}' not found in group")

        return self.experiments[experiment_id]

    def get_experiments(
        self, status: Optional[ExperimentStatus] = None
    ) -> List[Experiment]:
        """
        Get experiments from the group.

        Parameters
        ----------
        status : ExperimentStatus, optional
            Status to filter experiments by.
            If None, returns all experiments.

        Returns
        -------
        experiments : list
            List of experiments matching the specified status.
        """
        if status is None:
            return list(self.experiments.values())

        return [e for e in self.experiments.values() if e.status == status]

    def start_all(self, start_date: Optional[datetime.datetime] = None) -> None:
        """
        Start all experiments in the group.

        Parameters
        ----------
        start_date : datetime, optional
            Start date for the experiments. If None, uses current time.
        """
        for experiment in self.experiments.values():
            if experiment.status == ExperimentStatus.CREATED:
                experiment.start(start_date)

    def pause_all(self) -> None:
        """Pause all running experiments in the group."""
        for experiment in self.experiments.values():
            if experiment.status == ExperimentStatus.RUNNING:
                experiment.pause()

    def resume_all(self) -> None:
        """Resume all paused experiments in the group."""
        for experiment in self.experiments.values():
            if experiment.status == ExperimentStatus.PAUSED:
                experiment.resume()

    def complete_all(self, end_date: Optional[datetime.datetime] = None) -> None:
        """
        Complete all running or paused experiments in the group.

        Parameters
        ----------
        end_date : datetime, optional
            End date for the experiments. If None, uses current time.
        """
        for experiment in self.experiments.values():
            if experiment.status in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
                experiment.complete(end_date)

    def get_combined_results(
        self, metric: Optional[str] = None, as_dataframe: bool = True
    ) -> Union[Dict, pd.DataFrame]:
        """
        Get combined results for all experiments in the group.

        Parameters
        ----------
        metric : str, optional
            Name of the metric to get results for.
            If None, returns results for all metrics.
        as_dataframe : bool, default=True
            Whether to return results as a DataFrame.

        Returns
        -------
        results : dict or DataFrame
            Combined results for the experiments.
        """
        if as_dataframe:
            dfs = []

            for experiment in self.experiments.values():
                df = experiment.get_results(metric=metric, as_dataframe=True)

                if not df.empty:
                    df["experiment_id"] = experiment.id
                    df["experiment_name"] = experiment.name
                    dfs.append(df)

            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()

        else:
            combined_results = {}

            for experiment_id, experiment in self.experiments.items():
                combined_results[experiment_id] = experiment.get_results(
                    metric=metric, as_dataframe=False
                )

            return combined_results

    def get_summary_statistics(self, metric: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary statistics for all experiments in the group.

        Parameters
        ----------
        metric : str, optional
            Name of the metric to get statistics for.
            If None, returns statistics for all metrics.

        Returns
        -------
        stats : DataFrame
            Summary statistics for all experiments.
        """
        dfs = []

        for experiment in self.experiments.values():
            df = experiment.get_summary_statistics(metric=metric)

            if not df.empty:
                df["experiment_id"] = experiment.id
                df["experiment_name"] = experiment.name
                dfs.append(df)

        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def to_dict(self) -> Dict:
        """
        Convert experiment group to dictionary.

        Returns
        -------
        group_dict : dict
            Dictionary representation of the experiment group.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "experiment_ids": list(self.experiments.keys()),
            "creation_date": self.creation_date.isoformat(),
        }

    def to_json(self, include_experiments: bool = False) -> str:
        """
        Convert experiment group to JSON.

        Parameters
        ----------
        include_experiments : bool, default=False
            Whether to include experiments in the JSON.

        Returns
        -------
        group_json : str
            JSON representation of the experiment group.
        """
        data = self.to_dict()

        if include_experiments:
            data["experiments"] = {
                exp_id: exp.to_dict() for exp_id, exp in self.experiments.items()
            }

        return json.dumps(data, indent=2)

    def save(
        self,
        directory: str,
        include_experiments: bool = True,
        save_experiments: bool = True,
    ) -> str:
        """
        Save experiment group to file.

        Parameters
        ----------
        directory : str
            Directory to save the experiment group to.
        include_experiments : bool, default=True
            Whether to include experiments in the saved file.
        save_experiments : bool, default=True
            Whether to save individual experiment files.

        Returns
        -------
        filepath : str
            Path to the saved file.
        """
        os.makedirs(directory, exist_ok=True)

        # Create filename
        filename = f"{self.id}_{self.name.replace(' ', '_')}.json"
        filepath = os.path.join(directory, filename)

        # Save individual experiments if requested
        if save_experiments:
            experiments_dir = os.path.join(directory, f"{self.id}_experiments")
            os.makedirs(experiments_dir, exist_ok=True)

            for experiment in self.experiments.values():
                experiment.save(experiments_dir)

        # Save group to file
        with open(filepath, "w") as f:
            f.write(self.to_json(include_experiments=include_experiments))

        return filepath

    @classmethod
    def load(cls, filepath: str, load_experiments: bool = True) -> "ExperimentGroup":
        """
        Load experiment group from file.

        Parameters
        ----------
        filepath : str
            Path to the experiment group file.
        load_experiments : bool, default=True
            Whether to load individual experiment files.

        Returns
        -------
        group : ExperimentGroup
            Loaded experiment group.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Create group
        group = cls(name=data["name"], description=data["description"])

        # Set attributes
        group.id = data["id"]
        group.creation_date = datetime.datetime.fromisoformat(data["creation_date"])

        # Load experiments if available
        if "experiments" in data and load_experiments:
            for exp_id, exp_data in data["experiments"].items():
                experiment = Experiment(
                    name=exp_data["name"],
                    description=exp_data["description"],
                    metrics=exp_data["metrics"],
                )
                # Set additional attributes for the experiment
                experiment.id = exp_id
                experiment.creation_date = datetime.datetime.fromisoformat(
                    exp_data["creation_date"]
                )
                group.experiments[exp_id] = experiment

        return group
