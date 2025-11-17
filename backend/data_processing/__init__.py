"""
AlphaMind Data Processing Framework

This package provides optimized data processing capabilities for financial applications,
including parallel processing, streaming data handling, caching mechanisms,
and pipeline monitoring tools.
"""

from .caching import CacheManager, CachePolicy, DiskCache, MemoryCache, RedisCache
from .monitoring import (
    AlertManager,
    MetricsCollector,
    PerformanceTracker,
    PipelineMonitor,
)
from .parallel import DistributedComputing, ParallelProcessor, TaskManager, WorkerPool
from .pipeline import (
    DataExporter,
    DataLoader,
    DataPipeline,
    DataTransformer,
    PipelineStage,
)
from .streaming import (
    DataStream,
    KafkaStreamAdapter,
    StreamingPipeline,
    StreamProcessor,
    WebSocketStreamAdapter,
)

__all__ = [
    "ParallelProcessor",
    "TaskManager",
    "WorkerPool",
    "DistributedComputing",
    "StreamProcessor",
    "DataStream",
    "StreamingPipeline",
    "KafkaStreamAdapter",
    "WebSocketStreamAdapter",
    "CacheManager",
    "MemoryCache",
    "DiskCache",
    "RedisCache",
    "CachePolicy",
    "PipelineMonitor",
    "MetricsCollector",
    "PerformanceTracker",
    "AlertManager",
    "DataPipeline",
    "PipelineStage",
    "DataTransformer",
    "DataLoader",
    "DataExporter",
]
