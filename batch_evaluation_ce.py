#!/usr/bin/env python3
"""
Batch evaluation script for cross-encoder models using multiple GPUs.
This script runs evaluations in parallel across available GPUs.
"""

import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration for all models and datasets to evaluate
MODELS_CONFIG = [
    {
        "model_name": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        "name": "mDeBERTa-v3-base-mnli-xnli",
        "entailment": 0,
    },
    {
        "model_name": "cross-encoder/nli-deberta-v3-small",
        "name": "nli-deberta-v3-small",
        "entailment": 1,
    },
    {
        "model_name": "cross-encoder/nli-deberta-v3-base",
        "name": "nli-deberta-v3-base",
        "entailment": 1,
    },
    {
        "model_name": "alexandrainst/scandi-nli-small",
        "name": "scandi-nli-small",
        "entailment": 0,
    },
    {
        "model_name": "alexandrainst/scandi-nli-large-v2",
        "name": "scandi-nli-large-v2",
        "entailment": 0,
    },
    {
        "model_name": "facebook/bart-large-mnli",
        "name": "bart-large-mnli",
        "entailment": 2,
    },
]

DATASETS = [
    "agnews",
    "imdb",
    "amazon_massive_intent",
    "dbpedia",
    "events_biotech",
    "yahoo",
]

SOFTMAX_DATA = [
    "events_biotech",
]
# ACTIVATIONS = ["softmax", "sigmoid"]


def run_evaluation(
    model_config: Dict, dataset: str, activation: str, device_pos: int
) -> Tuple[bool, str]:
    """
    Run a single evaluation task.

    Args:
        model_config: Dictionary containing model configuration
        dataset: Dataset name to evaluate on
        activation: Activation function to use
        device_pos: GPU device position (0 or 1)

    Returns:
        Tuple of (success, output_message)
    """
    model_name = model_config["model_name"]
    name = model_config["name"]
    entailment = model_config["entailment"]

    results_dir = f"results/evaluation/cross_encoder/{name}/{dataset}"

    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python",
        "evaluate_ce_agnews.py",
        "--model_path",
        model_name,
        "--data",
        dataset,
        "--results_dir",
        results_dir,
        "--activation",
        activation,
        "--entailment",
        str(entailment),
        "--device_pos",
        str(device_pos),
    ]

    try:
        print(f"\n{'='*80}")
        print(f"GPU {device_pos}: 🚀 STARTING EVALUATION")
        print(f"Model: {name}")
        print(f"Dataset: {dataset}")
        print(f"Activation: {activation}")
        print(f"Entailment: {entailment}")
        print(f"Results dir: {results_dir}")
        print(f"{'='*80}")

        # Run the command with shared stdout/stderr so we can see real-time output
        result = subprocess.run(
            cmd,
            timeout=3600,  # 1 hour timeout
            # Don't capture output - let it go to the main process stdout/stderr
            stdout=None,  # Use parent's stdout
            stderr=None,  # Use parent's stderr
            text=True,
        )

        if result.returncode == 0:
            success_msg = (
                f"GPU {device_pos}: ✅ COMPLETED {name} on {dataset} with {activation}"
            )
            print(f"\n{'='*80}")
            print(success_msg)
            print(f"{'='*80}\n")
            return True, success_msg
        else:
            error_msg = f"GPU {device_pos}: ❌ FAILED {name} on {dataset} with {activation} (exit code: {result.returncode})"
            print(f"\n{'='*80}")
            print(error_msg)
            print(f"{'='*80}\n")
            return False, error_msg

    except subprocess.TimeoutExpired:
        timeout_msg = f"GPU {device_pos}: ⏰ TIMEOUT {name} on {dataset} with {activation} (>1 hour)"
        print(f"\n{'='*80}")
        print(timeout_msg)
        print(f"{'='*80}\n")
        return False, timeout_msg
    except Exception as e:
        exception_msg = f"GPU {device_pos}: 💥 EXCEPTION {name} on {dataset} with {activation}: {str(e)}"
        print(f"\n{'='*80}")
        print(exception_msg)
        print(f"{'='*80}\n")
        return False, exception_msg


def create_task_queue() -> queue.Queue:
    """
    Create a queue of all evaluation tasks.

    Returns:
        Queue containing tuples of (model_config, dataset, activation)
    """
    task_queue = queue.Queue()
    for model_config in MODELS_CONFIG:
        for dataset in DATASETS:
            activation = "softmax" if dataset in SOFTMAX_DATA else "sigmoid"
            task_queue.put((model_config, dataset, activation))
    return task_queue


def gpu_worker(
    gpu_id: int,
    task_queue: queue.Queue,
    results_list: List,
    results_lock: threading.Lock,
) -> None:
    """
    Worker function for a GPU that processes tasks from the queue.

    Args:
        gpu_id: GPU device ID (0 or 1)
        task_queue: Queue containing tasks to process
        results_list: Shared list to store results
        results_lock: Lock for thread-safe access to results_list
    """
    task_count = 0
    while True:
        try:
            # Get next task from queue (with timeout to avoid hanging)
            model_config, dataset, activation = task_queue.get(timeout=1)
            task_count += 1

            remaining_tasks = task_queue.qsize()
            print(
                f"\n🔄 GPU {gpu_id}: Picking up task #{task_count} ({remaining_tasks} remaining in queue)"
            )
            print(f"📋 Task: {model_config['name']} → {dataset} ({activation})")

            # Run the evaluation
            start_time = time.time()
            success, message = run_evaluation(model_config, dataset, activation, gpu_id)
            end_time = time.time()

            duration = end_time - start_time
            print(f"⏱️  GPU {gpu_id}: Task completed in {duration:.1f} seconds")

            # Store result in thread-safe manner
            with results_lock:
                results_list.append((success, message))

            # Mark task as done
            task_queue.task_done()

            # Small delay to avoid potential issues
            time.sleep(1)

        except queue.Empty:
            # No more tasks in queue, worker can exit
            print(
                f"\n🏁 GPU {gpu_id}: No more tasks available, worker finishing (completed {task_count} tasks)"
            )
            break
        except Exception as e:
            print(f"\n💥 GPU {gpu_id}: Worker error: {e}")
            break


def progress_monitor(
    task_queue: queue.Queue,
    results_list: List,
    results_lock: threading.Lock,
    total_tasks: int,
    start_time: float,
) -> None:
    """
    Monitor progress and print periodic updates.

    Args:
        task_queue: The task queue to monitor
        results_list: List of completed results
        results_lock: Lock for thread-safe access
        total_tasks: Total number of tasks
        start_time: When the batch started
    """
    while True:
        time.sleep(30)  # Update every 30 seconds

        with results_lock:
            completed_tasks = len(results_list)

        remaining_tasks = task_queue.qsize()

        if completed_tasks >= total_tasks:
            break

        elapsed_time = time.time() - start_time

        # Calculate progress stats
        progress_percent = (completed_tasks / total_tasks) * 100

        if completed_tasks > 0:
            avg_time_per_task = elapsed_time / completed_tasks
            estimated_remaining_time = avg_time_per_task * remaining_tasks

            print(f"\n📊 PROGRESS UPDATE ({time.strftime('%H:%M:%S')})")
            print(
                f"✅ Completed: {completed_tasks}/{total_tasks} ({progress_percent:.1f}%)"
            )
            print(f"⏳ Remaining: {remaining_tasks} tasks")
            print(
                f"⏱️  Elapsed: {elapsed_time:.0f}s | Est. remaining: {estimated_remaining_time:.0f}s"
            )
            print(f"🔄 Avg time per task: {avg_time_per_task:.1f}s")
            print("-" * 50)


def main():
    """
    Main function to run batch evaluation with dynamic task assignment.
    """
    print("🚀 Starting batch evaluation for cross-encoder models")
    print(f"📊 Total models: {len(MODELS_CONFIG)}")
    print(f"📊 Total datasets: {len(DATASETS)}")
    # print(f"📊 Total activations: {len(ACTIVATIONS)}")

    # Create task queue
    task_queue = create_task_queue()
    total_tasks = task_queue.qsize()
    print(f"📊 Total evaluation tasks: {total_tasks}")

    # Show estimated time
    estimated_time_per_task = 120  # seconds (rough estimate)
    estimated_total_time = (total_tasks * estimated_time_per_task) / 2  # 2 GPUs
    print(f"⏱️  Estimated total time: {estimated_total_time/3600:.1f} hours")

    # Setup for dynamic task execution
    num_gpus = 2
    results_list = []
    results_lock = threading.Lock()

    print(f"🔧 Using {num_gpus} GPUs with dynamic task assignment")
    print("🔧 Each GPU will pick up tasks as they become available")
    print("🔧 Real-time output from evaluations will be shown below")

    # Start timing
    start_time = time.time()

    # Create and start worker threads for each GPU
    threads = []
    for gpu_id in range(num_gpus):
        thread = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, task_queue, results_list, results_lock),
            name=f"GPU-{gpu_id}-Worker",
        )
        thread.start()
        threads.append(thread)
        print(f"🚀 Started worker for GPU {gpu_id}")

    # Start progress monitor thread
    progress_thread = threading.Thread(
        target=progress_monitor,
        args=(task_queue, results_list, results_lock, total_tasks, start_time),
        name="Progress-Monitor",
        daemon=True,
    )
    progress_thread.start()

    # Wait for all tasks to complete
    print(f"\n⏳ Waiting for all {total_tasks} tasks to complete...")
    print("=" * 80)
    task_queue.join()

    # Wait for all worker threads to finish
    for thread in threads:
        thread.join()

    end_time = time.time()

    # Summary
    successful_tasks = sum(1 for success, _ in results_list if success)
    failed_tasks = total_tasks - successful_tasks

    print("\n" + "=" * 80)
    print("📈 BATCH EVALUATION SUMMARY")
    print("=" * 80)
    print(
        f"⏱️  Total time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/3600:.2f} hours)"
    )
    print(f"✅ Successful tasks: {successful_tasks}/{total_tasks}")
    print(f"❌ Failed tasks: {failed_tasks}/{total_tasks}")
    if total_tasks > 0:
        print(f"📊 Success rate: {successful_tasks/total_tasks*100:.1f}%")
        print(f"🏃 Average time per task: {(end_time - start_time)/total_tasks:.1f}s")

    # Show failed tasks
    if failed_tasks > 0:
        print("\n❌ Failed tasks:")
        for success, message in results_list:
            if not success:
                print(f"   {message}")

    print("\n🎉 Batch evaluation completed!")


if __name__ == "__main__":
    main()
