"""
AlphaMind Data Processing Framework

This package provides optimized data processing capabilities for financial applications,
including parallel processing, streaming data handling, caching mechanisms,
and pipeline monitoring tools.
"""

from .parallel import (
    ParallelProcessor, 
    TaskManager, 
    WorkerPool,
    DistributedComputing
)
from .streaming import (
    StreamProcessor, 
    DataStream, 
    StreamingPipeline,
    KafkaStreamAdapter,
    WebSocketStreamAdapter
)
from .caching import (
    CacheManager,
    MemoryCache,
    DiskCache,
    RedisCache,
    CachePolicy
)
from .monitoring import (
    PipelineMonitor,
    MetricsCollector,
    PerformanceTracker,
    AlertManager
)
from .pipeline import (
    DataPipeline,
    PipelineStage,
    DataTransformer,
    DataLoader,
    DataExporter
)

__all__ = [
    'ParallelProcessor', 'TaskManager', 'WorkerPool', 'DistributedComputing',
    'StreamProcessor', 'DataStream', 'StreamingPipeline', 'KafkaStreamAdapter', 'WebSocketStreamAdapter',
    'CacheManager', 'MemoryCache', 'DiskCache', 'RedisCache', 'CachePolicy',
    'PipelineMonitor', 'MetricsCollector', 'PerformanceTracker', 'AlertManager',
    'DataPipeline', 'PipelineStage', 'DataTransformer', 'DataLoader', 'DataExporter'
]
