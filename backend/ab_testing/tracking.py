"""
Experiment tracking for A/B testing.

This module provides classes for tracking experiments and their results
in the A/B testing framework, including persistent storage and retrieval.
"""

import datetime
import os
import pickle
import sqlite3
from typing import Dict, List, Optional, Union, Any
import uuid
import pandas as pd
from ab_testing.experiment import Experiment, ExperimentStatus


class ExperimentResult:
    """
    Class representing the result of an experiment.

    This class provides methods for storing and analyzing
    the results of A/B test experiments.

    Parameters
    ----------
    experiment_id : str
        ID of the experiment.
    variant : str
        Name of the variant.
    metric : str
        Name of the metric.
    value : float
        Value of the metric.
    timestamp : datetime, optional
        Timestamp of the result. If None, uses current time.
    metadata : dict, optional
        Additional metadata for the result.
    """

    def __init__(
        self,
        experiment_id: str,
        variant: str,
        metric: str,
        value: float,
        timestamp: Optional[datetime.datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.id = str(uuid.uuid4())
        self.experiment_id = experiment_id
        self.variant = variant
        self.metric = metric
        self.value = value
        self.timestamp = timestamp or datetime.datetime.now()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert result to dictionary.

        Returns
        -------
        result_dict : dict
            Dictionary representation of the result.
        """
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "variant": self.variant,
            "metric": self.metric,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls: Any, data: Dict) -> "ExperimentResult":
        """Create result from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation of the result.

        Returns
        -------
        result : ExperimentResult
            Result created from the dictionary.
        """
        result = cls(
            experiment_id=data["experiment_id"],
            variant=data["variant"],
            metric=data["metric"],
            value=data["value"],
            metadata=data.get("metadata", {}),
        )
        result.id = data["id"]
        result.timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        return result


class ExperimentTracker:
    """
    Class for tracking experiments and their results.

    This class provides methods for storing, retrieving, and
    analyzing experiments and their results.

    Parameters
    ----------
    storage_dir : str, optional
        Directory for storing experiment data.
        If None, uses in-memory storage.
    db_path : str, optional
        Path to SQLite database for storing results.
        If None, uses in-memory database.
    """

    def __init__(
        self, storage_dir: Optional[str] = None, db_path: Optional[str] = None
    ) -> None:
        self.storage_dir = storage_dir
        self.db_path = db_path
        self.experiments = {}
        self.in_memory = storage_dir is None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database for storing results."""
        if self.db_path:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = sqlite3.connect(self.db_path)
        else:
            conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute(
            "\n        CREATE TABLE IF NOT EXISTS experiments (\n            id TEXT PRIMARY KEY,\n            name TEXT NOT NULL,\n            description TEXT,\n            status TEXT NOT NULL,\n            start_date TEXT,\n            end_date TEXT,\n            creation_date TEXT NOT NULL,\n            data BLOB\n        )\n        "
        )
        cursor.execute(
            "\n        CREATE TABLE IF NOT EXISTS results (\n            id TEXT PRIMARY KEY,\n            experiment_id TEXT NOT NULL,\n            variant TEXT NOT NULL,\n            metric TEXT NOT NULL,\n            value REAL NOT NULL,\n            timestamp TEXT NOT NULL,\n            metadata BLOB,\n            FOREIGN KEY (experiment_id) REFERENCES experiments (id)\n        )\n        "
        )
        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a connection to the database."""
        if self.db_path:
            return sqlite3.connect(self.db_path)
        else:
            return sqlite3.connect(":memory:")

    def add_experiment(self, experiment: Experiment, save: bool = True) -> None:
        """Add an experiment to the tracker.

        Parameters
        ----------
        experiment : Experiment
            Experiment to add.
        save : bool, default=True
            Whether to save the experiment to storage.
        """
        self.experiments[experiment.id] = experiment
        if save and (not self.in_memory):
            self._save_experiment(experiment)

    def _save_experiment(self, experiment: Experiment) -> None:
        """Save an experiment to storage.

        Parameters
        ----------
        experiment : Experiment
            Experiment to save.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        data = pickle.dumps(experiment)
        cursor.execute(
            "\n         INSERT OR REPLACE INTO experiments\n         (id, name, description, status, start_date, end_date, creation_date, data)\n         VALUES (?, ?, ?, ?, ?, ?, ?, ?)\n        ",
            (
                experiment.id,
                experiment.name,
                experiment.description,
                experiment.status.value,
                experiment.start_date.isoformat() if experiment.start_date else None,
                experiment.end_date.isoformat() if experiment.end_date else None,
                experiment.creation_date.isoformat(),
                data,
            ),
        )
        conn.commit()
        conn.close()
        if self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            experiment.save(self.storage_dir)

    def get_experiment(
        self, experiment_id: str, load_if_needed: bool = True
    ) -> Experiment:
        """Get an experiment from the tracker.

        Parameters
        ----------
        experiment_id : str
            ID of the experiment to get.
        load_if_needed : bool, default=True
            Whether to load the experiment from storage if not in memory.

        Returns
        -------
        experiment : Experiment
            Experiment with the specified ID.
        """
        if experiment_id in self.experiments:
            return self.experiments[experiment_id]
        if load_if_needed and (not self.in_memory):
            experiment = self._load_experiment(experiment_id)
            if experiment:
                self.experiments[experiment_id] = experiment
                return experiment
        raise ValueError(f"Experiment with ID '{experiment_id}' not found")

    def _load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load an experiment from storage.

        Parameters
        ----------
        experiment_id : str
            ID of the experiment to load.

        Returns
        -------
        experiment : Experiment or None
            Loaded experiment, or None if not found.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "\n         SELECT data FROM experiments WHERE id = ?\n        ",
            (experiment_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            experiment = pickle.loads(row[0])
            return experiment
        if self.storage_dir:
            for filename in os.listdir(self.storage_dir):
                if filename.startswith(f"{experiment_id}_") and filename.endswith(
                    ".json"
                ):
                    filepath = os.path.join(self.storage_dir, filename)
                    return Experiment.load(filepath)
        return None

    def get_experiments(
        self, status: Optional[ExperimentStatus] = None, load_all: bool = False
    ) -> List[Experiment]:
        """Get experiments from the tracker.

        Parameters
        ----------
        status : ExperimentStatus, optional
            Status to filter experiments by.
            If None, returns all experiments.
        load_all : bool, default=False
            Whether to load all experiments from storage.

        Returns
        -------
        experiments : list
            List of experiments.
        """
        if load_all and (not self.in_memory):
            self._load_all_experiments()
        if status is None:
            return list(self.experiments.values())
        return [e for e in self.experiments.values() if e.status == status]

    def _load_all_experiments(self) -> None:
        """Load all experiments from storage."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, data FROM experiments")
        for row in cursor.fetchall():
            experiment_id, data = row
            if experiment_id not in self.experiments:
                experiment = pickle.loads(data)
                self.experiments[experiment_id] = experiment
        conn.close()
        if self.storage_dir and os.path.exists(self.storage_dir):
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_dir, filename)
                    try:
                        experiment = Experiment.load(filepath)
                        if experiment.id not in self.experiments:
                            self.experiments[experiment.id] = experiment
                    except:
                        pass

    def update_experiment(self, experiment: Experiment, save: bool = True) -> None:
        """Update an experiment in the tracker.

        Parameters
        ----------
        experiment : Experiment
            Experiment to update.
        save : bool, default=True
            Whether to save the experiment to storage.
        """
        self.experiments[experiment.id] = experiment
        if save and (not self.in_memory):
            self._save_experiment(experiment)

    def delete_experiment(self, experiment_id: str) -> None:
        """Delete an experiment from the tracker.

        Parameters
        ----------
        experiment_id : str
            ID of the experiment to delete.
        """
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
        if not self.in_memory:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
            cursor.execute(
                "DELETE FROM results WHERE experiment_id = ?", (experiment_id,)
            )
            conn.commit()
            conn.close()
            if self.storage_dir:
                for filename in os.listdir(self.storage_dir):
                    if filename.startswith(f"{experiment_id}_") and filename.endswith(
                        ".json"
                    ):
                        filepath = os.path.join(self.storage_dir, filename)
                        try:
                            os.remove(filepath)
                        except:
                            pass

    def add_result(self, result: ExperimentResult, save: bool = True) -> None:
        """Add a result to the tracker.

        Parameters
        ----------
        result : ExperimentResult
            Result to add.
        save : bool, default=True
            Whether to save the result to storage.
        """
        try:
            experiment = self.get_experiment(result.experiment_id)
            experiment.add_result(
                result.variant, result.metric, result.value, result.timestamp
            )
            if save and (not self.in_memory):
                self._save_experiment(experiment)
        except ValueError:
            pass
        if save and (not self.in_memory):
            self._save_result(result)

    def _save_result(self, result: ExperimentResult) -> None:
        """Save a result to storage.

        Parameters
        ----------
        result : ExperimentResult
            Result to save.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        metadata = pickle.dumps(result.metadata)
        cursor.execute(
            "\n         INSERT OR REPLACE INTO results\n         (id, experiment_id, variant, metric, value, timestamp, metadata)\n         VALUES (?, ?, ?, ?, ?, ?, ?)\n        ",
            (
                result.id,
                result.experiment_id,
                result.variant,
                result.metric,
                result.value,
                result.timestamp.isoformat(),
                metadata,
            ),
        )
        conn.commit()
        conn.close()

    def get_results(
        self,
        experiment_id: Optional[str] = None,
        variant: Optional[str] = None,
        metric: Optional[str] = None,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        as_dataframe: bool = True,
    ) -> Union[List[ExperimentResult], pd.DataFrame]:
        """Get results from the tracker.

        Parameters
        ----------
        experiment_id : str, optional
            ID of the experiment to get results for.
            If None, returns results for all experiments.
        variant : str, optional
            Name of the variant to get results for.
            If None, returns results for all variants.
        metric : str, optional
            Name of the metric to get results for.
            If None, returns results for all metrics.
        start_date : datetime, optional
            Start date for filtering results.
            If None, returns results from the beginning.
        end_date : datetime, optional
            End date for filtering results.
            If None, returns results until the end.
        as_dataframe : bool, default=True
            Whether to return results as a DataFrame.

        Returns
        -------
        results : list or DataFrame
            Results matching the specified criteria.
        """
        query = "SELECT id, experiment_id, variant, metric, value, timestamp, metadata FROM results"
        params: List[Any] = []
        conditions = []
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)
        if variant:
            conditions.append("variant = ?")
            params.append(variant)
        if metric:
            conditions.append("metric = ?")
            params.append(metric)
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp"
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        results: List[Any] = []
        for row in cursor.fetchall():
            result_id, exp_id, var, met, val, ts, meta = row
            result = ExperimentResult(
                experiment_id=exp_id,
                variant=var,
                metric=met,
                value=val,
                timestamp=datetime.datetime.fromisoformat(ts),
            )
            result.id = result_id
            if meta:
                result.metadata = pickle.loads(meta)
            results.append(result)
        conn.close()
        if as_dataframe:
            data: List[Any] = []
            for result in results:
                data.append(
                    {
                        "id": result.id,
                        "experiment_id": result.experiment_id,
                        "variant": result.variant,
                        "metric": result.metric,
                        "value": result.value,
                        "timestamp": result.timestamp,
                    }
                )
            return pd.DataFrame(data)
        return results

    def get_summary_statistics(
        self,
        experiment_id: Optional[str] = None,
        metric: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get summary statistics for results.

        Parameters
        ----------
        experiment_id : str, optional
            ID of the experiment to get statistics for.
            If None, returns statistics for all experiments.
        metric : str, optional
            Name of the metric to get statistics for.
            If None, returns statistics for all metrics.
        variant : str, optional
            Name of the variant to get statistics for.
            If None, returns statistics for all variants.

        Returns
        -------
        stats : DataFrame
            Summary statistics for the results.
        """
        results_df = self.get_results(
            experiment_id=experiment_id,
            metric=metric,
            variant=variant,
            as_dataframe=True,
        )
        if hasattr(results_df, "empty") and results_df.empty:
            return pd.DataFrame()
        stats: List[Any] = []
        for (exp_id, met, var), group in results_df.groupby(
            ["experiment_id", "metric", "variant"]
        ):
            values = group["value"]
            stats.append(
                {
                    "experiment_id": exp_id,
                    "metric": met,
                    "variant": var,
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

    def export_results(
        self, filepath: str, experiment_id: Optional[str] = None, format: str = "csv"
    ) -> None:
        """Export results to a file.

        Parameters
        ----------
        filepath : str
            Path to save the results to.
        experiment_id : str, optional
            ID of the experiment to export results for.
            If None, exports results for all experiments.
        format : str, default="csv"
            Format to export results in. Options: "csv", "json", "excel".
        """
        results_df = self.get_results(experiment_id=experiment_id, as_dataframe=True)
        if hasattr(results_df, "empty") and results_df.empty:
            raise ValueError("No results to export")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        if format == "csv":
            results_df.to_csv(filepath, index=False)
        elif format == "json":
            results_df.to_json(filepath, orient="records", indent=2)
        elif format == "excel":
            results_df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_results(self, filepath: str, format: str = "csv") -> None:
        """Import results from a file.

        Parameters
        ----------
        filepath : str
            Path to the file to import results from.
        format : str, default="csv"
            Format of the file. Options: "csv", "json", "excel".
        """
        if format == "csv":
            results_df = pd.read_csv(filepath)
        elif format == "json":
            results_df = pd.read_json(filepath, orient="records")
        elif format == "excel":
            results_df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        if "timestamp" in results_df.columns:
            results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])
        for _, row in results_df.iterrows():
            result = ExperimentResult(
                experiment_id=row["experiment_id"],
                variant=row["variant"],
                metric=row["metric"],
                value=row["value"],
                timestamp=row.get("timestamp"),
            )
            if "id" in row:
                result.id = row["id"]
            self.add_result(result)

    def clear(self) -> None:
        """Clear all experiments and results from the tracker."""
        self.experiments = {}
        if not self.in_memory:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM experiments")
            cursor.execute("DELETE FROM results")
            conn.commit()
            conn.close()
            if self.storage_dir and os.path.exists(self.storage_dir):
                for filename in os.listdir(self.storage_dir):
                    if filename.endswith(".json"):
                        filepath = os.path.join(self.storage_dir, filename)
                        try:
                            os.remove(filepath)
                        except:
                            pass
