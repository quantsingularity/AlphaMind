"""
Parallel processing utilities for financial data.

This module provides classes and functions for parallel processing
of financial data, including task management, worker pools, and
distributed computing capabilities.
"""

import os
import time
import uuid
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Iterator
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from functools import partial
import threading
import queue


class ParallelProcessor:
    """
    Base class for parallel processing of financial data.
    
    This class provides methods for executing tasks in parallel
    using various parallelization strategies.
    
    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes or threads.
        If None, uses the number of CPU cores.
    use_threads : bool, default=False
        Whether to use threads instead of processes.
        Threads are faster for I/O-bound tasks, processes for CPU-bound tasks.
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_threads: bool = False
    ):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threads = use_threads
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Apply a function to each item in parallel.
        
        Parameters
        ----------
        func : callable
            Function to apply to each item.
        items : list
            List of items to process.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.
            
        Returns
        -------
        results : list
            List of results.
        """
        if not items:
            return []
        
        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)
        
        # Execute in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            results = list(executor.map(func, items))
        
        return results
    
    def apply(
        self,
        funcs: List[Callable],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Apply multiple functions in parallel.
        
        Parameters
        ----------
        funcs : list
            List of functions to execute.
        *args : tuple
            Additional positional arguments to pass to each function.
        **kwargs : dict
            Additional keyword arguments to pass to each function.
            
        Returns
        -------
        results : list
            List of results.
        """
        if not funcs:
            return []
        
        # Create partial functions with additional arguments
        if args or kwargs:
            funcs = [partial(func, *args, **kwargs) for func in funcs]
        
        # Execute in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            futures = [executor.submit(func) for func in funcs]
            results = [future.result() for future in as_completed(futures)]
        
        return results
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        by: Optional[Union[str, List[str]]] = None,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process a DataFrame in parallel.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame to process.
        func : callable
            Function to apply to each group or partition.
        by : str or list, optional
            Column(s) to group by. If None, splits the DataFrame into partitions.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.
            
        Returns
        -------
        result_df : DataFrame
            Processed DataFrame.
        """
        if df.empty:
            return df
        
        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)
        
        if by is not None:
            # Group by specified column(s)
            groups = df.groupby(by)
            group_names = list(groups.groups.keys())
            
            # Process each group in parallel
            with self.executor_class(max_workers=self.n_workers) as executor:
                futures = {executor.submit(func, groups.get_group(name)): name for name in group_names}
                results = {}
                
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results[name] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing group {name}: {e}")
                        results[name] = None
            
            # Combine results
            valid_results = [result for result in results.values() if result is not None]
            if not valid_results:
                return pd.DataFrame()
            
            return pd.concat(valid_results)
        
        else:
            # Split DataFrame into partitions
            n_rows = len(df)
            chunk_size = max(1, n_rows // self.n_workers)
            chunks = [df.iloc[i:i+chunk_size] for i in range(0, n_rows, chunk_size)]
            
            # Process each partition in parallel
            with self.executor_class(max_workers=self.n_workers) as executor:
                futures = [executor.submit(func, chunk) for chunk in chunks]
                results = []
                
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error(f"Error processing partition: {e}")
            
            # Combine results
            if not results:
                return pd.DataFrame()
            
            return pd.concat(results)
    
    def process_dask_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process a DataFrame using Dask for parallelization.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame to process.
        func : callable
            Function to apply to each partition.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.
            
        Returns
        -------
        result_df : DataFrame
            Processed DataFrame.
        """
        if df.empty:
            return df
        
        # Convert to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=self.n_workers)
        
        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)
        
        # Apply function to each partition
        result_df = dask_df.map_partitions(func).compute()
        
        return result_df
    
    def parallel_apply(
        self,
        df: pd.DataFrame,
        func: Callable,
        axis: int = 0,
        *args,
        **kwargs
    ) -> pd.Series:
        """
        Apply a function along an axis of a DataFrame in parallel.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame to process.
        func : callable
            Function to apply to each row or column.
        axis : int, default=0
            Axis along which to apply the function (0 for rows, 1 for columns).
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.
            
        Returns
        -------
        result : Series
            Series with the results.
        """
        if df.empty:
            return pd.Series()
        
        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)
        
        # Get items to process
        if axis == 0:
            items = [df.iloc[i] for i in range(len(df))]
        else:
            items = [df[col] for col in df.columns]
        
        # Process in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            results = list(executor.map(func, items))
        
        # Create Series with results
        if axis == 0:
            return pd.Series(results, index=df.index)
        else:
            return pd.Series(results, index=df.columns)


class TaskManager:
    """
    Manager for parallel task execution with dependencies.
    
    This class provides methods for defining and executing tasks
    with dependencies in parallel.
    
    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes or threads.
        If None, uses the number of CPU cores.
    use_threads : bool, default=False
        Whether to use threads instead of processes.
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_threads: bool = False
    ):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threads = use_threads
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        self.tasks = {}
        self.dependencies = {}
        self.results = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_task(
        self,
        task_id: str,
        func: Callable,
        dependencies: Optional[List[str]] = None,
        *args,
        **kwargs
    ) -> None:
        """
        Add a task to the manager.
        
        Parameters
        ----------
        task_id : str
            Unique identifier for the task.
        func : callable
            Function to execute.
        dependencies : list, optional
            List of task IDs that this task depends on.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.
        """
        if task_id in self.tasks:
            raise ValueError(f"Task with ID '{task_id}' already exists")
        
        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)
        
        self.tasks[task_id] = func
        self.dependencies[task_id] = dependencies or []
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute all tasks in parallel, respecting dependencies.
        
        Returns
        -------
        results : dict
            Dictionary mapping task IDs to results.
        """
        # Reset results
        self.results = {}
        
        # Find tasks with no dependencies
        ready_tasks = [task_id for task_id, deps in self.dependencies.items() if not deps]
        pending_tasks = {task_id for task_id in self.tasks if task_id not in ready_tasks}
        
        # Execute tasks in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            futures = {}
            
            while ready_tasks or futures:
                # Submit ready tasks
                for task_id in ready_tasks:
                    func = self.tasks[task_id]
                    futures[executor.submit(func)] = task_id
                
                ready_tasks = []
                
                # Wait for a task to complete
                if futures:
                    done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    
                    for future in done:
                        task_id = futures.pop(future)
                        
                        try:
                            self.results[task_id] = future.result()
                        except Exception as e:
                            self.logger.error(f"Error executing task {task_id}: {e}")
                            self.results[task_id] = None
                        
                        # Find tasks that are now ready
                        for pending_id in list(pending_tasks):
                            if all(dep in self.results for dep in self.dependencies[pending_id]):
                                ready_tasks.append(pending_id)
                                pending_tasks.remove(pending_id)
        
        return self.results
    
    def get_result(
        self,
        task_id: str
    ) -> Any:
        """
        Get the result of a task.
        
        Parameters
        ----------
        task_id : str
            ID of the task.
            
        Returns
        -------
        result : any
            Result of the task.
        """
        if task_id not in self.results:
            raise ValueError(f"No result found for task '{task_id}'")
        
        return self.results[task_id]
    
    def clear(self) -> None:
        """Clear all tasks and results."""
        self.tasks = {}
        self.dependencies = {}
        self.results = {}


class WorkerPool:
    """
    Pool of workers for parallel task execution.
    
    This class provides methods for managing a pool of workers
    for executing tasks in parallel.
    
    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes or threads.
        If None, uses the number of CPU cores.
    use_threads : bool, default=False
        Whether to use threads instead of processes.
    """
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_threads: bool = False
    ):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threads = use_threads
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def start(self) -> None:
        """Start the worker pool."""
        if self.running:
            return
        
        self.running = True
        
        # Create and start workers
        for _ in range(self.n_workers):
            if self.use_threads:
                worker = threading.Thread(target=self._worker_loop)
            else:
                worker = mp.Process(target=self._worker_loop)
            
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self) -> None:
        """Stop the worker pool."""
        if not self.running:
            return
        
        self.running = False
        
        # Add termination signals to the queue
        for _ in range(self.n_workers):
            self.task_queue.put(None)
        
        # Wait for workers to terminate
        for worker in self.workers:
            worker.join()
        
        self.workers = []
    
    def _worker_loop(self) -> None:
        """Worker loop for processing tasks."""
        while self.running:
            try:
                # Get a task from the queue
                task = self.task_queue.get(timeout=1)
                
                # Check for termination signal
                if task is None:
                    break
                
                # Execute the task
                task_id, func, args, kwargs = task
                
                try:
                    result = func(*args, **kwargs)
                    self.result_queue.put((task_id, result, None))
                except Exception as e:
                    self.result_queue.put((task_id, None, str(e)))
                
            except queue.Empty:
                # No tasks available, continue waiting
                continue
    
    def submit(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Submit a task to the worker pool.
        
        Parameters
        ----------
        func : callable
            Function to execute.
        *args : tuple
            Positional arguments to pass to the function.
        **kwargs : dict
            Keyword arguments to pass to the function.
            
        Returns
        -------
        task_id : str
            Unique identifier for the task.
        """
        if not self.running:
            self.start()
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        
        # Add the task to the queue
        self.task_queue.put((task_id, func, args, kwargs))
        
        return task_id
    
    def get_result(
        self,
        task_id: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Tuple[str, Any, Optional[str]]:
        """
        Get a result from the worker pool.
        
        Parameters
        ----------
        task_id : str, optional
            ID of the task to get the result for.
            If None, returns the next available result.
        timeout : float, optional
            Maximum time to wait for a result.
            If None, waits indefinitely.
            
        Returns
        -------
        result : tuple
            Tuple containing (task_id, result, error).
        """
        if not self.running:
            raise RuntimeError("Worker pool is not running")
        
        if task_id is None:
            # Get the next available result
            try:
                return self.result_queue.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("Timed out waiting for a result")
        
        else:
            # Wait for a specific task to complete
            start_time = time.time()
            
            while timeout is None or time.time() - start_time < timeout:
                try:
                    result = self.result_queue.get(timeout=1)
                    
                    if result[0] == task_id:
                        return result
                    
                    # Put the result back in the queue
                    self.result_queue.put(result)
                    
                except queue.Empty:
                    # No results available, continue waiting
                    continue
            
            raise TimeoutError(f"Timed out waiting for task {task_id}")
    
    def wait_all(
        self,
        timeout: Optional[float] = None
    ) -> Dict[str, Tuple[Any, Optional[str]]]:
        """
        Wait for all submitted tasks to complete.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for all tasks to complete.
            If None, waits indefinitely.
            
        Returns
        -------
        results : dict
            Dictionary mapping task IDs to (result, error) tuples.
        """
        if not self.running:
            raise RuntimeError("Worker pool is not running")
        
        results = {}
        start_time = time.time()
        
        while self.task_queue.qsize() > 0 or self.result_queue.qsize() > 0:
            if timeout is not None and time.time() - start_time > timeout:
                break
            
            try:
                task_id, result, error = self.result_queue.get(timeout=1)
                results[task_id] = (result, error)
            except queue.Empty:
                # No results available, continue waiting
                continue
        
        return results


class DistributedComputing:
    """
    Distributed computing for financial data processing.
    
    This class provides methods for distributed computing
    using Dask for financial data processing.
    
    Parameters
    ----------
    scheduler : str, optional
        Dask scheduler to use. If None, uses local scheduler.
    n_workers : int, optional
        Number of worker processes.
        If None, uses the number of CPU cores.
    """
    
    def __init__(
        self,
        scheduler: Optional[str] = None,
        n_workers: Optional[int] = None
    ):
        self.scheduler = scheduler
        self.n_workers = n_workers or mp.cpu_count()
        self.client = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize Dask client
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize Dask client."""
        try:
            from dask.distributed import Client, LocalCluster
            
            if self.scheduler:
                # Connect to existing scheduler
                self.client = Client(self.scheduler)
            else:
                # Create local cluster
                cluster = LocalCluster(n_workers=self.n_workers)
                self.client = Client(cluster)
            
            self.logger.info(f"Initialized Dask client: {self.client}")
            
        except ImportError:
            self.logger.warning("Dask distributed not available, using local scheduler")
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process a DataFrame using Dask for distributed computing.
        
        Parameters
        ----------
        df : DataFrame
            DataFrame to process.
        func : callable
            Function to apply to each partition.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.
            
        Returns
        -------
        result_df : DataFrame
            Processed DataFrame.
        """
        if df.empty:
            return df
        
        # Convert to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=self.n_workers)
        
        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)
        
        # Apply function to each partition
        result_df = dask_df.map_partitions(func).compute()
        
        return result_df
    
    def submit(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Submit a function for distributed execution.
        
        Parameters
        ----------
        func : callable
            Function to execute.
        *args : tuple
            Positional arguments to pass to the function.
        **kwargs : dict
            Keyword arguments to pass to the function.
            
        Returns
        -------
        future : Future
            Future object representing the computation.
        """
        if self.client:
            return self.client.submit(func, *args, **kwargs)
        else:
            # Fall back to local execution
            return dask.delayed(func)(*args, **kwargs)
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Apply a function to each item in parallel.
        
        Parameters
        ----------
        func : callable
            Function to apply to each item.
        items : list
            List of items to process.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.
            
        Returns
        -------
        results : list
            List of results.
        """
        if not items:
            return []
        
        if self.client:
            # Use Dask client for distributed execution
            futures = self.client.map(func, items, *args, **kwargs)
            return self.client.gather(futures)
        else:
            # Fall back to local execution
            results = []
            for item in items:
                result = func(item, *args, **kwargs)
                results.append(result)
            return results
    
    def close(self) -> None:
        """Close the Dask client."""
        if self.client:
            self.client.close()
            self.client = None
