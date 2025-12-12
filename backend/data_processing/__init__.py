"""
## AlphaMind Data Processing Framework

This package provides optimized data processing capabilities for financial applications,
including parallel processing, streaming data handling, caching mechanisms,
and pipeline monitoring tools.

### Sub-Modules Overview:
1.  **pipeline**: Defines the sequential data transformation stages (ETL).
2.  **parallel**: Handles multi-threading, multi-processing, and distributed task execution.
3.  **streaming**: Manages real-time data ingestion and processing.
4.  **caching**: Provides various caching backends for performance optimization.
5.  **monitoring**: Implements tools for performance tracking and alert generation.
"""

# Import necessary components from their respective sub-modules
# These imports assume the sub-modules (e.g., caching, pipeline) are defined within the same package structure.
from data_processing.caching import (
    CacheManager,
    CachePolicy,
    DiskCache,
    MemoryCache,
    RedisCache,
)
from data_processing.monitoring import (
    AlertManager,
    MetricsCollector,
    PerformanceTracker,
    PipelineMonitor,
)
from data_processing.parallel import (
    DistributedComputing,
    ParallelProcessor,
    TaskManager,
    WorkerPool,
)
from data_processing.pipeline import (
    DataExporter,
    DataLoader,
    DataPipeline,
    DataTransformer,
    PipelineStage,
)
from data_processing.streaming import (
    DataStream,
    KafkaStreamAdapter,
    StreamingPipeline,
    StreamProcessor,
    WebSocketStreamAdapter,
)

# Define the public interface for the 'alphamind.data_processing' package.
# This makes all listed classes directly importable from the package root:
# `from alphamind.data_processing import DataPipeline, ParallelProcessor, CacheManager`
__all__ = [
    # Parallel Processing
    "ParallelProcessor",
    "TaskManager",
    "WorkerPool",
    "DistributedComputing",
    # Streaming
    "StreamProcessor",
    "DataStream",
    "StreamingPipeline",
    "KafkaStreamAdapter",
    "WebSocketStreamAdapter",
    # Caching
    "CacheManager",
    "MemoryCache",
    "DiskCache",
    "RedisCache",
    "CachePolicy",
    # Monitoring
    "PipelineMonitor",
    "MetricsCollector",
    "PerformanceTracker",
    "AlertManager",
    # Pipeline
    "DataPipeline",
    "PipelineStage",
    "DataTransformer",
    "DataLoader",
    "DataExporter",
]
