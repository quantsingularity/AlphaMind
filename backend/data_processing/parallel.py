"""
Parallel processing utilities for financial data.

This module provides classes and functions for parallel processing
of financial data, including task management, worker pools, and
distributed computing capabilities.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import concurrent.futures
from functools import partial
import logging
import multiprocessing as mp
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import dask
import dask.dataframe as dd
import pandas as pd


# Set up logging for this module
logger = logging.getLogger("alphamind.parallel")


class ParallelProcessor:
    """
    Base class for parallel processing of financial data.

    This class provides methods for executing tasks in parallel
    using various parallelization strategies (threads or processes).

    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes or threads.
        If None, uses the number of CPU cores.
    use_threads : bool, default=False
        Whether to use threads instead of processes.
        Threads are generally better for I/O-bound tasks (e.g., waiting for data),
        processes for CPU-bound tasks (e.g., complex calculations) due to the GIL.
    """

    def __init__(self, n_workers: Optional[int] = None, use_threads: bool = False):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threads = use_threads
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        self.logger = logger.getChild(self.__class__.__name__)
        self.logger.info(
            f"Initialized with {self.n_workers} workers using "
            f"{'Threads' if use_threads else 'Processes'}"
        )

    def map(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        Apply a function to each item in parallel.

        Parameters
        ----------
        func : callable
            Function to apply to each item. The first argument of `func` should be an item from `items`.
        items : list
            List of items to process.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        results : list
            List of results, maintaining the order of the input `items`.
        """
        if not items:
            return []

        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)

        # Execute in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            # `executor.map` handles arguments and returns results in input order
            results = list(executor.map(func, items))

        return results

    def apply(self, funcs: List[Callable], *args, **kwargs) -> List[Any]:
        """
        Apply multiple functions in parallel.

        Parameters
        ----------
        funcs : list
            List of callable functions to execute.
        *args : tuple
            Additional positional arguments to pass to each function.
        **kwargs : dict
            Additional keyword arguments to pass to each function.

        Returns
        -------
        results : list
            List of results, ordered by completion time.
        """
        if not funcs:
            return []

        # Create partial functions with additional arguments
        if args or kwargs:
            funcs = [partial(func, *args, **kwargs) for func in funcs]

        # Execute in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            futures = [executor.submit(func) for func in funcs]
            # `as_completed` yields results as they finish, which can be faster than waiting for all
            results = [future.result() for future in as_completed(futures)]

        return results

    def process_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        by: Optional[Union[str, List[str]]] = None,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Process a DataFrame in parallel by splitting it into partitions or groups.

        Parameters
        ----------
        df : DataFrame
            DataFrame to process.
        func : callable
            Function to apply to each group (if `by` is not None) or partition (if `by` is None).
        by : str or list, optional
            Column(s) to group by. If None, splits the DataFrame into partitions by row count.
        *args : tuple
            Additional positional arguments to pass to the function.
        **kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        result_df : DataFrame
            Processed DataFrame, concatenated from the results of the parallel executions.
        """
        if df.empty:
            return df

        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)

        if by is not None:
            # --- Group-based processing (e.g., by 'Symbol') ---
            groups = df.groupby(by)
            group_names = list(groups.groups.keys())
            items_to_process = [groups.get_group(name) for name in group_names]

            self.logger.info(f"Processing {len(group_names)} groups in parallel...")

            # Process each group in parallel
            with self.executor_class(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(func, item): group_names[i]
                    for i, item in enumerate(items_to_process)
                }
                results_map = {}

                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        results_map[name] = future.result()
                    except Exception as e:
                        self.logger.error(f"Error processing group {name}: {e}")
                        results_map[name] = None

            # Combine results, ensuring only valid results are concatenated
            valid_results = [
                result for result in results_map.values() if result is not None
            ]
            if not valid_results:
                return pd.DataFrame()

            # Result is concatenated, restoring original index/structure might require post-processing
            return pd.concat(valid_results)

        else:
            # --- Partition-based processing (by row chunk) ---
            n_rows = len(df)
            chunk_size = max(1, n_rows // self.n_workers)
            chunks = [df.iloc[i : i + chunk_size] for i in range(0, n_rows, chunk_size)]

            self.logger.info(f"Processing {len(chunks)} partitions in parallel...")

            # Process each partition in parallel using map for ordered results
            results = self.map(func, chunks)

            # Combine results
            valid_results = [result for result in results if result is not None]
            if not valid_results:
                return pd.DataFrame()

            return pd.concat(valid_results)

    def process_dask_dataframe(
        self, df: pd.DataFrame, func: Callable, *args, **kwargs
    ) -> pd.DataFrame:
        """
        Process a DataFrame using Dask for streamlined, optimized parallelization.
        This uses Dask's internal schedulers (threads/processes) which are generally
        more efficient than the concurrent.futures executors for DataFrame tasks.

        Parameters
        ----------
        df : DataFrame
            DataFrame to process.
        func : callable
            Function to apply to each partition. Must return a Pandas object.
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

        # Convert to Dask DataFrame, partitioning by the worker count
        dask_df = dd.from_pandas(df, npartitions=self.n_workers)

        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)

        # Apply function to each partition and then compute the final result
        # Dask handles the parallel execution based on its configured scheduler (local process/thread pool by default)
        result_df = dask_df.map_partitions(func).compute()

        return result_df

    def parallel_apply(
        self, df: pd.DataFrame, func: Callable, axis: int = 0, *args, **kwargs
    ) -> pd.Series:
        """
        Apply a function along an axis of a DataFrame (row or column) in parallel.
        This is essentially a parallel version of `df.apply(func, axis=axis)`.

        Parameters
        ----------
        df : DataFrame
            DataFrame to process.
        func : callable
            Function to apply to each row (axis=0) or column (axis=1).
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

        # Get items (rows or columns) to process
        if axis == 0:
            items = [df.iloc[i] for i in range(len(df))]
            index = df.index
            self.logger.info(f"Applying function to {len(items)} rows in parallel...")
        else:
            items = [df[col] for col in df.columns]
            index = df.columns
            self.logger.info(
                f"Applying function to {len(items)} columns in parallel..."
            )

        # Process in parallel using map to preserve order
        with self.executor_class(max_workers=self.n_workers) as executor:
            results = list(executor.map(func, items))

        # Create Series with results
        return pd.Series(results, index=index)


# --- Dependency-Based Task Management ---


class TaskManager:
    """
    Manager for parallel task execution with dependencies.

    This class provides methods for defining and executing tasks
    in parallel, respecting a defined dependency graph.

    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes or threads.
        If None, uses the number of CPU cores.
    use_threads : bool, default=False
        Whether to use threads instead of processes.
    """

    def __init__(self, n_workers: Optional[int] = None, use_threads: bool = False):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threads = use_threads
        # Must import concurrent.futures explicitly for wait function
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        self.tasks: Dict[str, Callable] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.results: Dict[str, Any] = {}
        self.logger = logger.getChild(self.__class__.__name__)

    def add_task(
        self,
        task_id: str,
        func: Callable,
        dependencies: Optional[List[str]] = None,
        *args,
        **kwargs,
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
            List of task IDs that this task depends on (must be executed first).
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
            Dictionary mapping task IDs to results. Returns None for failed tasks.
        """
        # Reset results
        self.results = {}
        self.logger.info(
            f"Starting execution of {len(self.tasks)} tasks with dependencies."
        )

        # Find tasks with no dependencies (initial ready tasks)
        ready_tasks = [
            task_id for task_id, deps in self.dependencies.items() if not deps
        ]
        pending_tasks = {
            task_id for task_id in self.tasks if task_id not in ready_tasks
        }

        # Execute tasks in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            futures: Dict[concurrent.futures.Future, str] = {}
            total_tasks = len(self.tasks)
            completed_count = 0

            while ready_tasks or futures:
                # Submit ready tasks
                for task_id in ready_tasks:
                    self.logger.debug(f"Submitting task: {task_id}")
                    # Prepare the function call: include results of dependencies as arguments if needed
                    # NOTE: This implementation does not currently pass results to dependent tasks,
                    # but assumes the function will fetch/use results from a shared resource/state.
                    func = self.tasks[task_id]
                    futures[executor.submit(func)] = task_id

                ready_tasks = []  # Clear the queue of newly submitted tasks

                # Wait for a task to complete
                if futures:
                    # Wait for the first submitted task to complete
                    done, _ = concurrent.futures.wait(
                        futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for future in done:
                        task_id = futures.pop(future)
                        completed_count += 1

                        try:
                            self.results[task_id] = future.result()
                            self.logger.info(
                                f"Completed task: {task_id} ({completed_count}/{total_tasks})"
                            )
                        except Exception as e:
                            self.logger.error(
                                f"Error executing task {task_id}: {e}", exc_info=True
                            )
                            self.results[task_id] = (
                                None  # Mark as completed with failure result
                            )

                        # Find tasks that are now ready
                        for pending_id in list(pending_tasks):
                            # Check if all dependencies for the pending task are now in results
                            if all(
                                dep in self.results
                                for dep in self.dependencies[pending_id]
                            ):
                                ready_tasks.append(pending_id)
                                pending_tasks.remove(pending_id)
                elif pending_tasks:
                    # Should not happen if dependencies are resolvable, indicates deadlock or issue
                    self.logger.error(
                        f"Deadlock detected or remaining pending tasks: {pending_tasks}"
                    )
                    break

        self.logger.info(
            f"Execution finished. Total tasks: {total_tasks}, Completed: {completed_count}"
        )
        return self.results

    def get_result(self, task_id: str) -> Any:
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

        Raises
        ------
        ValueError: If no result is found for the task.
        """
        if task_id not in self.results:
            raise ValueError(
                f"No result found for task '{task_id}'. Task may not have run or failed."
            )

        return self.results[task_id]

    def clear(self) -> None:
        """Clear all tasks, dependencies, and results."""
        self.tasks = {}
        self.dependencies = {}
        self.results = {}
        self.logger.info("Task Manager cleared.")


# --- Queue-Based Worker Pool ---


class WorkerPool:
    """
    Pool of workers for parallel task execution using queues.

    This class provides methods for managing a pool of persistent workers
    (threads or processes) that pull tasks from a queue.

    Parameters
    ----------
    n_workers : int, optional
        Number of worker processes or threads.
        If None, uses the number of CPU cores.
    use_threads : bool, default=False
        Whether to use threads instead of processes.
    """

    def __init__(self, n_workers: Optional[int] = None, use_threads: bool = False):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_threads = use_threads
        # Use appropriate queues based on Thread or Process
        QueueClass = queue.Queue if use_threads else mp.Queue
        self.task_queue: QueueClass = QueueClass()
        self.result_queue: QueueClass = QueueClass()
        self.workers: List[Union[threading.Thread, mp.Process]] = []
        self.running = False
        self.logger = logger.getChild(self.__class__.__name__)

    def start(self) -> None:
        """Start the worker pool."""
        if self.running:
            return

        self.running = True
        WorkerClass = threading.Thread if self.use_threads else mp.Process

        # Create and start workers
        for i in range(self.n_workers):
            worker = WorkerClass(target=self._worker_loop, name=f"Worker-{i}")
            worker.daemon = True  # Allows program to exit even if workers are running
            worker.start()
            self.workers.append(worker)

        self.logger.info(
            f"Started worker pool with {self.n_workers} {'threads' if self.use_threads else 'processes'}."
        )

    def stop(self) -> None:
        """Stop the worker pool."""
        if not self.running:
            return

        self.running = False

        # Add termination signals (None) to the queue for each worker
        for _ in range(self.n_workers):
            self.task_queue.put(None)

        # Wait for workers to terminate
        for worker in self.workers:
            # Use a timeout to prevent indefinite blocking if a worker is stuck
            worker.join(timeout=5)

        self.workers = []
        self.logger.info("Worker pool stopped.")

    def _worker_loop(self) -> None:
        """Worker loop for processing tasks from the queue."""
        while self.running:
            try:
                # Get a task from the queue with a timeout to allow checking `self.running`
                task = self.task_queue.get(timeout=0.1)

                # Check for termination signal
                if task is None:
                    break

                # Execute the task: task is a tuple (task_id, func, args, kwargs)
                task_id, func, args, kwargs = task

                try:
                    result = func(*args, **kwargs)
                    self.result_queue.put(
                        (task_id, result, None)
                    )  # (id, result, error=None)
                except Exception as e:
                    self.logger.error(f"Task {task_id} failed: {e}", exc_info=True)
                    self.result_queue.put(
                        (task_id, None, str(e))
                    )  # (id, result=None, error)
                finally:
                    # Signal that the task is complete for queue management
                    self.task_queue.task_done()

            except queue.Empty:
                # No tasks available, continue waiting
                continue
            except Exception as e:
                # Catch other potential worker errors
                self.logger.critical(
                    f"Critical error in worker loop: {e}", exc_info=True
                )
                break  # Exit the loop if a critical error occurs

    def submit(self, func: Callable, *args, **kwargs) -> str:
        """
        Submit a task to the worker pool. Starts the pool if it's not running.

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
        self.logger.debug(f"Submitted task {task_id}")

        return task_id

    def get_result(
        self, task_id: Optional[str] = None, timeout: Optional[float] = None
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

        Returns
        -------
        result : tuple
            Tuple containing (task_id, result, error string).

        Raises
        ------
        RuntimeError: If the worker pool is not running.
        TimeoutError: If the wait times out.
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
            # We must poll the queue and temporarily store other results
            other_results = []

            while timeout is None or time.time() - start_time < timeout:
                try:
                    result = self.result_queue.get(timeout=1)

                    if result[0] == task_id:
                        # Return all temporarily stored results back to the queue
                        for other_res in other_results:
                            self.result_queue.put(other_res)
                        return result

                    # Store other results found while searching
                    other_results.append(result)

                except queue.Empty:
                    continue

            # If loop exits due to timeout, return stored results to queue and raise error
            for other_res in other_results:
                self.result_queue.put(other_res)

            raise TimeoutError(f"Timed out waiting for task {task_id}")

    def wait_all(
        self, timeout: Optional[float] = None
    ) -> Dict[str, Tuple[Any, Optional[str]]]:
        """
        Wait for all submitted tasks to complete.

        Note: This is non-blocking with respect to *future* submissions,
        but blocks until the current tasks are done or the timeout is reached.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for all currently submitted tasks to complete.

        Returns
        -------
        results : dict
            Dictionary mapping task IDs to (result, error) tuples.
        """
        if not self.running:
            raise RuntimeError("Worker pool is not running")

        results = {}
        start_time = time.time()

        # Calculate the number of results expected (tasks currently in queue + those that finished)
        initial_tasks = self.task_queue.qsize() + len(results)

        while len(results) < initial_tasks:
            if timeout is not None and time.time() - start_time > timeout:
                self.logger.warning(f"wait_all timed out after {timeout} seconds.")
                break

            try:
                # Wait for results based on the remaining timeout
                remaining_timeout = (
                    timeout - (time.time() - start_time) if timeout is not None else 1
                )
                task_id, result, error = self.result_queue.get(
                    timeout=remaining_timeout
                )
                results[task_id] = (result, error)
            except queue.Empty:
                # If the result queue is empty, and the task queue is also empty, we might be done
                if self.task_queue.qsize() == 0:
                    # This can be a false negative if workers are still computing tasks
                    if len(results) == initial_tasks:
                        break  # All results collected
                    else:
                        # Continue waiting for in-progress tasks
                        continue
                continue
            except TimeoutError:
                # If `get` times out, check condition again
                continue

        return results


# --- Distributed Computing with Dask ---


class DistributedComputing:
    """
    Distributed computing for financial data processing using Dask.

    This class provides methods for distributed computing
    using Dask, leveraging a cluster for scaling beyond a single machine.
    Requires the `dask.distributed` package.

    Parameters
    ----------
    scheduler : str, optional
        Dask scheduler address (e.g., 'tcp://<ip>:<port>').
        If None, a local Dask cluster is created.
    n_workers : int, optional
        Number of worker processes for the local cluster.
        If None, uses the number of CPU cores.
    """

    def __init__(
        self, scheduler: Optional[str] = None, n_workers: Optional[int] = None
    ):
        self.scheduler = scheduler
        self.n_workers = n_workers or mp.cpu_count()
        self.client = None
        self.cluster = None
        self.logger = logger.getChild(self.__class__.__name__)

        # Initialize Dask client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Dask client."""
        try:
            # Local import to prevent hard dependency on dask.distributed
            # if only Dask core is installed.
            from dask.distributed import Client, LocalCluster

            if self.scheduler:
                # Connect to existing scheduler
                self.client = Client(self.scheduler)
                self.logger.info(f"Connected to Dask scheduler: {self.scheduler}")
            else:
                # Create local cluster
                self.cluster = LocalCluster(n_workers=self.n_workers)
                self.client = Client(self.cluster)
                self.logger.info(
                    f"Created local Dask cluster with {self.n_workers} workers."
                )

            self.logger.info(f"Initialized Dask client: {self.client}")

        except ImportError:
            self.logger.warning(
                "Dask distributed not available. DistributedComputing will not function."
            )
        except Exception as e:
            self.logger.error(f"Error initializing Dask client: {e}", exc_info=True)

    def process_dataframe(
        self, df: pd.DataFrame, func: Callable, *args, **kwargs
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
        if not self.client:
            self.logger.warning(
                "Dask client not available. Falling back to local pandas execution."
            )
            # Fall back to local application if Dask isn't available
            return func(df, *args, **kwargs)

        # Convert to Dask DataFrame, using the client's worker count as a default for partitions
        npartitions = self.n_workers
        dask_df = dd.from_pandas(df, npartitions=npartitions)

        # Create partial function with additional arguments
        if args or kwargs:
            func = partial(func, *args, **kwargs)

        # Apply function to each partition and then compute the final result
        result_df = dask_df.map_partitions(func).compute()

        return result_df

    def submit(self, func: Callable, *args, **kwargs) -> Any:
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
        future : dask.distributed.Future or dask.delayed
            Future object representing the computation. Use `.result()` or `.compute()` to retrieve the result.
        """
        if self.client:
            # Use Dask client for immediate submission to the cluster
            return self.client.submit(func, *args, **kwargs)
        else:
            # Fall back to local execution using Dask's delayed mechanism
            return dask.delayed(func)(*args, **kwargs)

    def map(self, func: Callable, items: List[Any], *args, **kwargs) -> List[Any]:
        """
        Apply a function to each item in parallel using the Dask cluster.

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
            return self.client.gather(
                futures
            )  # Gathers results back to the local client
        else:
            # Fall back to local execution using a simple map
            return [func(item, *args, **kwargs) for item in items]

    def close(self) -> None:
        """Close the Dask client and cluster (if locally managed)."""
        if self.client:
            self.client.close()
            self.client = None
        if self.cluster:
            self.cluster.close()
            self.cluster = None

        self.logger.info("Dask client and cluster closed.")
