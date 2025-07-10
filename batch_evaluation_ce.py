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

ACTIVATIONS = ["softmax", "sigmoid"]


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
        print(
            f"GPU {device_pos}: Starting evaluation for {name} on {dataset} with {activation}"
        )

        # Run the command
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            success_msg = (
                f"GPU {device_pos}: ✓ Completed {name} on {dataset} with {activation}"
            )
            print(success_msg)
            return True, success_msg
        else:
            error_msg = f"GPU {device_pos}: ✗ Failed {name} on {dataset} with {activation}\nError: {result.stderr}"
            print(error_msg)
            return False, error_msg

    except subprocess.TimeoutExpired:
        timeout_msg = (
            f"GPU {device_pos}: ⏰ Timeout for {name} on {dataset} with {activation}"
        )
        print(timeout_msg)
        return False, timeout_msg
    except Exception as e:
        exception_msg = f"GPU {device_pos}: ❌ Exception for {name} on {dataset} with {activation}: {str(e)}"
        print(exception_msg)
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
            for activation in ACTIVATIONS:
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
    while True:
        try:
            # Get next task from queue (with timeout to avoid hanging)
            model_config, dataset, activation = task_queue.get(timeout=1)

            # Run the evaluation
            success, message = run_evaluation(model_config, dataset, activation, gpu_id)

            # Store result in thread-safe manner
            with results_lock:
                results_list.append((success, message))

            # Mark task as done
            task_queue.task_done()

            # Small delay to avoid potential issues
            time.sleep(1)

        except queue.Empty:
            # No more tasks in queue, worker can exit
            print(f"GPU {gpu_id}: No more tasks, worker finishing")
            break
        except Exception as e:
            print(f"GPU {gpu_id}: Worker error: {e}")
            break


def main():
    """
    Main function to run batch evaluation with dynamic task assignment.
    """
    print("🚀 Starting batch evaluation for cross-encoder models")
    print(f"📊 Total models: {len(MODELS_CONFIG)}")
    print(f"📊 Total datasets: {len(DATASETS)}")
    print(f"📊 Total activations: {len(ACTIVATIONS)}")

    # Create task queue
    task_queue = create_task_queue()
    total_tasks = task_queue.qsize()
    print(f"📊 Total evaluation tasks: {total_tasks}")

    # Setup for dynamic task execution
    num_gpus = 2
    results_list = []
    results_lock = threading.Lock()

    print(f"🔧 Using {num_gpus} GPUs with dynamic task assignment")
    print("🔧 Each GPU will pick up tasks as they become available")

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

    # Wait for all tasks to complete
    print("⏳ Waiting for all tasks to complete...")
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
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"✅ Successful tasks: {successful_tasks}/{total_tasks}")
    print(f"❌ Failed tasks: {failed_tasks}/{total_tasks}")
    if total_tasks > 0:
        print(f"📊 Success rate: {successful_tasks/total_tasks*100:.1f}%")

    # Show failed tasks
    if failed_tasks > 0:
        print("\n❌ Failed tasks:")
        for success, message in results_list:
            if not success:
                print(f"   {message}")

    print("\n🎉 Batch evaluation completed!")


if __name__ == "__main__":
    main()
