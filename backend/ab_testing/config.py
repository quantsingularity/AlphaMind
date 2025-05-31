"""
Configuration management for A/B testing experiments.

This module provides classes for managing experiment configurations,
including parameter definitions, variant configurations, and
experiment settings.
"""

import os
import json
import yaml
import datetime
import uuid
from typing import Dict, List, Optional, Union, Any, Tuple
import copy


class ExperimentConfig:
    """
    Class for managing experiment configurations.
    
    This class provides methods for defining, validating, and
    managing experiment configurations, including parameters,
    variants, and settings.
    
    Parameters
    ----------
    name : str
        Name of the experiment.
    description : str, optional
        Description of the experiment.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description or ""
        self.parameters = {}
        self.variants = {}
        self.metrics = []
        self.settings = {
            "start_date": None,
            "end_date": None,
            "sample_size": None,
            "traffic_allocation": {},
            "randomization_unit": "user",
            "randomization_salt": str(uuid.uuid4()),
            "segment_filters": {}
        }
        self.creation_date = datetime.datetime.now()
    
    def add_parameter(
        self,
        name: str,
        default_value: Any,
        description: Optional[str] = None,
        type_hint: Optional[str] = None,
        constraints: Optional[Dict] = None
    ) -> None:
        """
        Add a parameter to the experiment configuration.
        
        Parameters
        ----------
        name : str
            Name of the parameter.
        default_value : any
            Default value of the parameter.
        description : str, optional
            Description of the parameter.
        type_hint : str, optional
            Type hint for the parameter (e.g., "int", "float", "str", "bool").
        constraints : dict, optional
            Constraints for the parameter (e.g., min, max, choices).
        """
        self.parameters[name] = {
            "default_value": default_value,
            "description": description or "",
            "type_hint": type_hint or type(default_value).__name__,
            "constraints": constraints or {}
        }
    
    def add_variant(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> None:
        """
        Add a variant to the experiment configuration.
        
        Parameters
        ----------
        name : str
            Name of the variant.
        parameters : dict, optional
            Parameter values for the variant.
            If None, uses default values.
        description : str, optional
            Description of the variant.
        """
        # Use default parameter values if not specified
        if parameters is None:
            parameters = {name: param["default_value"] for name, param in self.parameters.items()}
        
        # Validate parameters
        for param_name, value in parameters.items():
            if param_name not in self.parameters:
                raise ValueError(f"Parameter '{param_name}' not defined in experiment")
            
            # Validate type
            param_def = self.parameters[param_name]
            expected_type = param_def["type_hint"]
            
            if expected_type == "int" and not isinstance(value, int):
                raise TypeError(f"Parameter '{param_name}' should be an integer")
            elif expected_type == "float" and not isinstance(value, (int, float)):
                raise TypeError(f"Parameter '{param_name}' should be a float")
            elif expected_type == "str" and not isinstance(value, str):
                raise TypeError(f"Parameter '{param_name}' should be a string")
            elif expected_type == "bool" and not isinstance(value, bool):
                raise TypeError(f"Parameter '{param_name}' should be a boolean")
            
            # Validate constraints
            constraints = param_def["constraints"]
            
            if "min" in constraints and value < constraints["min"]:
                raise ValueError(f"Parameter '{param_name}' should be >= {constraints['min']}")
            
            if "max" in constraints and value > constraints["max"]:
                raise ValueError(f"Parameter '{param_name}' should be <= {constraints['max']}")
            
            if "choices" in constraints and value not in constraints["choices"]:
                raise ValueError(f"Parameter '{param_name}' should be one of {constraints['choices']}")
        
        # Add variant
        self.variants[name] = {
            "parameters": parameters,
            "description": description or ""
        }
    
    def add_metric(
        self,
        name: str,
        description: Optional[str] = None,
        higher_is_better: bool = True,
        minimum_detectable_effect: Optional[float] = None
    ) -> None:
        """
        Add a metric to the experiment configuration.
        
        Parameters
        ----------
        name : str
            Name of the metric.
        description : str, optional
            Description of the metric.
        higher_is_better : bool, default=True
            Whether higher values are better.
        minimum_detectable_effect : float, optional
            Minimum detectable effect size.
        """
        self.metrics.append({
            "name": name,
            "description": description or "",
            "higher_is_better": higher_is_better,
            "minimum_detectable_effect": minimum_detectable_effect
        })
    
    def set_traffic_allocation(
        self,
        allocations: Dict[str, float]
    ) -> None:
        """
        Set traffic allocation for variants.
        
        Parameters
        ----------
        allocations : dict
            Dictionary mapping variant names to allocation percentages (0-1).
        """
        # Validate variants
        for variant in allocations:
            if variant not in self.variants:
                raise ValueError(f"Variant '{variant}' not defined in experiment")
        
        # Validate allocations
        total_allocation = sum(allocations.values())
        if not (0.99 <= total_allocation <= 1.01):  # Allow for small floating-point errors
            raise ValueError(f"Total allocation should be 1.0, got {total_allocation}")
        
        self.settings["traffic_allocation"] = allocations
    
    def set_experiment_duration(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        duration_days: Optional[int] = None
    ) -> None:
        """
        Set experiment duration.
        
        Parameters
        ----------
        start_date : datetime, optional
            Start date of the experiment.
            If None, uses current time.
        end_date : datetime, optional
            End date of the experiment.
            If None but duration_days is provided, calculated from start_date.
        duration_days : int, optional
            Duration of the experiment in days.
            If None but end_date is provided, ignored.
        """
        start_date = start_date or datetime.datetime.now()
        
        if end_date is None and duration_days is not None:
            end_date = start_date + datetime.timedelta(days=duration_days)
        
        self.settings["start_date"] = start_date
        self.settings["end_date"] = end_date
    
    def set_sample_size(
        self,
        sample_size: int
    ) -> None:
        """
        Set sample size for the experiment.
        
        Parameters
        ----------
        sample_size : int
            Sample size for the experiment.
        """
        if sample_size <= 0:
            raise ValueError("Sample size should be positive")
        
        self.settings["sample_size"] = sample_size
    
    def set_randomization_unit(
        self,
        unit: str
    ) -> None:
        """
        Set randomization unit for the experiment.
        
        Parameters
        ----------
        unit : str
            Randomization unit (e.g., "user", "session", "device").
        """
        self.settings["randomization_unit"] = unit
    
    def add_segment_filter(
        self,
        name: str,
        filter_expression: str
    ) -> None:
        """
        Add a segment filter to the experiment.
        
        Parameters
        ----------
        name : str
            Name of the segment filter.
        filter_expression : str
            Filter expression (e.g., "country == 'US'").
        """
        self.settings["segment_filters"][name] = filter_expression
    
    def validate(self) -> List[str]:
        """
        Validate the experiment configuration.
        
        Returns
        -------
        errors : list
            List of validation errors.
        """
        errors = []
        
        # Check if at least one variant is defined
        if not self.variants:
            errors.append("No variants defined")
        
        # Check if at least one metric is defined
        if not self.metrics:
            errors.append("No metrics defined")
        
        # Check if traffic allocation is defined for all variants
        if self.settings["traffic_allocation"]:
            for variant in self.variants:
                if variant not in self.settings["traffic_allocation"]:
                    errors.append(f"No traffic allocation defined for variant '{variant}'")
        else:
            errors.append("No traffic allocation defined")
        
        # Check if start date is defined
        if self.settings["start_date"] is None:
            errors.append("No start date defined")
        
        return errors
    
    def to_dict(self) -> Dict:
        """
        Convert experiment configuration to dictionary.
        
        Returns
        -------
        config_dict : dict
            Dictionary representation of the experiment configuration.
        """
        config_dict = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "variants": self.variants,
            "metrics": self.metrics,
            "settings": {
                "start_date": self.settings["start_date"].isoformat() if self.settings["start_date"] else None,
                "end_date": self.settings["end_date"].isoformat() if self.settings["end_date"] else None,
                "sample_size": self.settings["sample_size"],
                "traffic_allocation": self.settings["traffic_allocation"],
                "randomization_unit": self.settings["randomization_unit"],
                "randomization_salt": self.settings["randomization_salt"],
                "segment_filters": self.settings["segment_filters"]
            },
            "creation_date": self.creation_date.isoformat()
        }
        
        return config_dict
    
    def to_json(self) -> str:
        """
        Convert experiment configuration to JSON.
        
        Returns
        -------
        config_json : str
            JSON representation of the experiment configuration.
        """
        return json.dumps(self.to_dict(), indent=2)
    
    def to_yaml(self) -> str:
        """
        Convert experiment configuration to YAML.
        
        Returns
        -------
        config_yaml : str
            YAML representation of the experiment configuration.
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(
        self,
        filepath: str,
        format: str = "json"
    ) -> None:
        """
        Save experiment configuration to file.
        
        Parameters
        ----------
        filepath : str
            Path to save the configuration to.
        format : str, default="json"
            Format to save the configuration in. Options: "json", "yaml".
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Save to file
        if format == "json":
            with open(filepath, "w") as f:
                f.write(self.to_json())
        elif format == "yaml":
            with open(filepath, "w") as f:
                f.write(self.to_yaml())
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(
        cls,
        filepath: str
    ) -> 'ExperimentConfig':
        """
        Load experiment configuration from file.
        
        Parameters
        ----------
        filepath : str
            Path to the configuration file.
            
        Returns
        -------
        config : ExperimentConfig
            Loaded experiment configuration.
        """
        # Determine format from file extension
        if filepath.endswith(".json"):
            with open(filepath, "r") as f:
                data = json.load(f)
        elif filepath.endswith((".yaml", ".yml")):
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        # Create configuration
        config = cls(
            name=data["name"],
            description=data["description"]
        )
        
        # Set attributes
        config.id = data["id"]
        config.parameters = data["parameters"]
        config.variants = data["variants"]
        config.metrics = data["metrics"]
        
        # Set settings
        if data["settings"]["start_date"]:
            config.settings["start_date"] = datetime.datetime.fromisoformat(data["settings"]["start_date"])
        
        if data["settings"]["end_date"]:
            config.settings["end_date"] = datetime.datetime.fromisoformat(data["settings"]["end_date"])
        
        config.settings["sample_size"] = data["settings"]["sample_size"]
        config.settings["traffic_allocation"] = data["settings"]["traffic_allocation"]
        config.settings["randomization_unit"] = data["settings"]["randomization_unit"]
        config.settings["randomization_salt"] = data["settings"]["randomization_salt"]
        config.settings["segment_filters"] = data["settings"]["segment_filters"]
        
        config.creation_date = datetime.datetime.fromisoformat(data["creation_date"])
        
        return config
    
    def clone(
        self,
        new_name: Optional[str] = None,
        new_description: Optional[str] = None
    ) -> 'ExperimentConfig':
        """
        Clone the experiment configuration.
        
        Parameters
        ----------
        new_name : str, optional
            Name for the cloned configuration.
            If None, uses the original name with " (Clone)" appended.
        new_description : str, optional
            Description for the cloned configuration.
            If None, uses the original description.
            
        Returns
        -------
        cloned_config : ExperimentConfig
            Cloned experiment configuration.
        """
        # Create new configuration
        cloned_config = ExperimentConfig(
            name=new_name or f"{self.name} (Clone)",
            description=new_description or self.description
        )
        
        # Copy attributes
        cloned_config.parameters = copy.deepcopy(self.parameters)
        cloned_config.variants = copy.deepcopy(self.variants)
        cloned_config.metrics = copy.deepcopy(self.metrics)
        cloned_config.settings = copy.deepcopy(self.settings)
        
        return cloned_config


class ExperimentConfigManager:
    """
    Class for managing multiple experiment configurations.
    
    This class provides methods for storing, retrieving, and
    managing multiple experiment configurations.
    
    Parameters
    ----------
    storage_dir : str, optional
        Directory for storing configuration files.
        If None, uses in-memory storage only.
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None
    ):
        self.storage_dir = storage_dir
        self.configs = {}
        
        # Load existing configurations if storage directory is provided
        if storage_dir and os.path.exists(storage_dir):
            self._load_configs()
    
    def _load_configs(self) -> None:
        """Load configurations from storage directory."""
        for filename in os.listdir(self.storage_dir):
            if filename.endswith((".json", ".yaml", ".yml")):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    config = ExperimentConfig.load(filepath)
                    self.configs[config.id] = config
                except:
                    # Skip files that can't be loaded
                    pass
    
    def add_config(
        self,
        config: ExperimentConfig,
        save: bool = True
    ) -> None:
        """
        Add a configuration to the manager.
        
        Parameters
        ----------
        config : ExperimentConfig
            Configuration to add.
        save : bool, default=True
            Whether to save the configuration to storage.
        """
        self.configs[config.id] = config
        
        if save and self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            filepath = os.path.join(self.storage_dir, f"{config.id}_{config.name.replace(' ', '_')}.json")
            config.save(filepath)
    
    def get_config(
        self,
        config_id: str
    ) -> ExperimentConfig:
        """
        Get a configuration from the manager.
        
        Parameters
        ----------
        config_id : str
            ID of the configuration to get.
            
        Returns
        -------
        config : ExperimentConfig
            Configuration with the specified ID.
        """
        if config_id not in self.configs:
            raise ValueError(f"Configuration with ID '{config_id}' not found")
        
        return self.configs[config_id]
    
    def get_configs(
        self,
        name_filter: Optional[str] = None
    ) -> List[ExperimentConfig]:
        """
        Get configurations from the manager.
        
        Parameters
        ----------
        name_filter : str, optional
            Filter configurations by name.
            If None, returns all configurations.
            
        Returns
        -------
        configs : list
            List of configurations.
        """
        if name_filter:
            return [c for c in self.configs.values() if name_filter.lower() in c.name.lower()]
        else:
            return list(self.configs.values())
    
    def update_config(
        self,
        config: ExperimentConfig,
        save: bool = True
    ) -> None:
        """
        Update a configuration in the manager.
        
        Parameters
        ----------
        config : ExperimentConfig
            Configuration to update.
        save : bool, default=True
            Whether to save the configuration to storage.
        """
        self.configs[config.id] = config
        
        if save and self.storage_dir:
            os.makedirs(self.storage_dir, exist_ok=True)
            filepath = os.path.join(self.storage_dir, f"{config.id}_{config.name.replace(' ', '_')}.json")
            config.save(filepath)
    
    def delete_config(
        self,
        config_id: str
    ) -> None:
        """
        Delete a configuration from the manager.
        
        Parameters
        ----------
        config_id : str
            ID of the configuration to delete.
        """
        if config_id in self.configs:
            del self.configs[config_id]
            
            if self.storage_dir:
                for filename in os.listdir(self.storage_dir):
                    if filename.startswith(f"{config_id}_") and filename.endswith((".json", ".yaml", ".yml")):
                        filepath = os.path.join(self.storage_dir, filename)
                        try:
                            os.remove(filepath)
                        except:
                            pass
    
    def export_configs(
        self,
        directory: str,
        format: str = "json"
    ) -> None:
        """
        Export all configurations to files.
        
        Parameters
        ----------
        directory : str
            Directory to export configurations to.
        format : str, default="json"
            Format to export configurations in. Options: "json", "yaml".
        """
        os.makedirs(directory, exist_ok=True)
        
        for config in self.configs.values():
            filepath = os.path.join(directory, f"{config.id}_{config.name.replace(' ', '_')}.{format}")
            config.save(filepath, format=format)
    
    def import_configs(
        self,
        directory: str
    ) -> int:
        """
        Import configurations from files.
        
        Parameters
        ----------
        directory : str
            Directory to import configurations from.
            
        Returns
        -------
        count : int
            Number of configurations imported.
        """
        count = 0
        
        for filename in os.listdir(directory):
            if filename.endswith((".json", ".yaml", ".yml")):
                filepath = os.path.join(directory, filename)
                try:
                    config = ExperimentConfig.load(filepath)
                    self.add_config(config, save=False)
                    count += 1
                except:
                    # Skip files that can't be loaded
                    pass
        
        return count
