import json
import os
import glob
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from datasets import Dataset
from tqdm import tqdm
from loguru import logger
import random
from sklearn.model_selection import train_test_split


@dataclass
class DatasetConfig:
    batch_dir: str
    output_dir: str = "data/prepared_datasets"
    test_ratio: float = 0.2
    min_label_frequency: int = 3  # Minimum times a label must appear
    random_seed: int = 42


class DatasetPreparator:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.all_labels = set()
        self.label_counts = Counter()
        self.label_to_texts = defaultdict(list)
        self.text_to_labels = {}
        
        # Set random seed for reproducibility
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
        
        logger.info(f"Initialized DatasetPreparator with config: {config}")
    
    def load_raw_data(self) -> Tuple[List[str], List[List[str]]]:
        """Load all JSON batch files and extract texts and labels."""
        texts = []
        labels = []
        
        batch_files = glob.glob(os.path.join(self.config.batch_dir, "*.json"))
        batch_files.sort()
        
        logger.info(f"Found {len(batch_files)} batch files in {self.config.batch_dir}")
        
        for batch_file in tqdm(batch_files, desc="Loading batches"):
            try:
                with open(batch_file, 'r') as f:
                    batch_data = json.load(f)
                    
                for item in batch_data:
                    if 'sentence' in item and 'labels' in item:
                        text = item['sentence']
                        item_labels = item['labels']
                        
                        texts.append(text)
                        labels.append(item_labels)
                        
                        # Build label statistics and mappings
                        self.text_to_labels[text] = item_labels
                        for label in item_labels:
                            self.all_labels.add(label)
                            self.label_counts[label] += 1
                            self.label_to_texts[label].append(text)
                            
            except Exception as e:
                logger.warning(f"Error loading {batch_file}: {e}")
                continue
        
        logger.success(f"Loaded {len(texts)} texts with {len(self.all_labels)} unique labels")
        return texts, labels
    
    def filter_rare_labels(self, texts: List[str], labels: List[List[str]]) -> Tuple[List[str], List[List[str]], Set[str]]:
        """Filter out labels that appear less than min_frequency times."""
        frequent_labels = {label for label, count in self.label_counts.items() 
                          if count >= self.config.min_label_frequency}
        
        filtered_texts = []
        filtered_labels = []
        
        for text, text_labels in zip(texts, labels):
            # Keep only frequent labels
            filtered_text_labels = [label for label in text_labels if label in frequent_labels]
            
            # Only keep texts that have at least one frequent label
            if filtered_text_labels:
                filtered_texts.append(text)
                filtered_labels.append(filtered_text_labels)
        
        logger.info(f"Filtered labels: {len(self.all_labels)} -> {len(frequent_labels)}")
        logger.info(f"Filtered texts: {len(texts)} -> {len(filtered_texts)}")
        
        return filtered_texts, filtered_labels, frequent_labels
    
    def split_by_labels(self, texts: List[str], labels: List[List[str]], frequent_labels: Set[str]) -> Tuple[
        List[str], List[List[str]], List[str], List[List[str]], Set[str], Set[str]
    ]:
        """
        Split data ensuring test set contains completely unseen labels.
        """
        logger.info("Creating train/test split with unseen labels...")
        
        # Convert to list for sampling
        frequent_labels_list = list(frequent_labels)
        
        # Calculate number of labels for test
        num_test_labels = max(1, int(len(frequent_labels_list) * self.config.test_ratio))
        
        # Randomly sample labels for test set
        test_labels = set(random.sample(frequent_labels_list, num_test_labels))
        train_labels = frequent_labels - test_labels
        
        logger.info(f"Train labels: {len(train_labels)}")
        logger.info(f"Test labels: {len(test_labels)} (completely unseen during training)")
        
        # Split texts based on label assignment
        train_texts = []
        train_text_labels = []
        test_texts = []
        test_text_labels = []
        
        for text, text_labels in zip(texts, labels):
            # Check if text has any test labels
            has_test_labels = any(label in test_labels for label in text_labels)
            # Check if text has any train labels
            has_train_labels = any(label in train_labels for label in text_labels)
            
            if has_test_labels and not has_train_labels:
                # Text only has test labels -> goes to test set
                test_labels_only = [label for label in text_labels if label in test_labels]
                test_texts.append(text)
                test_text_labels.append(test_labels_only)
                
            elif has_train_labels and not has_test_labels:
                # Text only has train labels -> goes to train set
                train_labels_only = [label for label in text_labels if label in train_labels]
                train_texts.append(text)
                train_text_labels.append(train_labels_only)
                
            elif has_train_labels and has_test_labels:
                # Text has both -> split by probability
                if random.random() < 0.7:  # 70% chance to train
                    train_labels_only = [label for label in text_labels if label in train_labels]
                    if train_labels_only:
                        train_texts.append(text)
                        train_text_labels.append(train_labels_only)
                else:  # 30% chance to test
                    test_labels_only = [label for label in text_labels if label in test_labels]
                    if test_labels_only:
                        test_texts.append(text)
                        test_text_labels.append(test_labels_only)
        
        logger.info("Final split:")
        logger.info(f"  Train: {len(train_texts)} texts")
        logger.info(f"  Test: {len(test_texts)} texts")
        
        return train_texts, train_text_labels, test_texts, test_text_labels, train_labels, test_labels
    
    def create_contrastive_dataset(self, texts: List[str], labels: List[List[str]], 
                                  available_labels: Set[str], dataset_name: str) -> Dataset:
        """Create contrastive learning dataset with (anchor, positive) pairs only."""
        logger.info(f"Creating {dataset_name} contrastive dataset...")
        
        dataset_dict = {
            'anchor': [],
            'positive': []
        }
        
        total_pairs = 0
        
        for text, text_labels in tqdm(zip(texts, labels), total=len(texts), 
                                     desc=f"Creating {dataset_name} pairs"):
            for positive_label in text_labels:
                dataset_dict['anchor'].append(text)
                dataset_dict['positive'].append(positive_label)
                total_pairs += 1
        
        logger.info(f"Created {total_pairs} (anchor, positive) pairs for {dataset_name}")
        logger.info("  Note: Negatives will be handled automatically by the loss function")
        return Dataset.from_dict(dataset_dict)
    
    def create_evaluation_pairs(self, texts: List[str], labels: List[List[str]], 
                               available_labels: Set[str]) -> Tuple[List[Tuple[str, str]], List[int]]:
        """Create evaluation pairs for binary classification (includes negatives for evaluation only)."""
        eval_pairs = []
        eval_labels = []
        
        for text, text_labels in zip(texts, labels):
            # Positive pairs (text matches its true labels)
            for label in text_labels:
                eval_pairs.append((text, label))
                eval_labels.append(1)
            
            # Negative pairs (text does NOT match other labels - for evaluation only)
            negative_labels = [label for label in available_labels if label not in text_labels]
            if negative_labels:
                # Sample fewer negatives for evaluation efficiency
                num_negatives = min(3, len(negative_labels))
                sampled_negatives = random.sample(negative_labels, num_negatives)
                
                for neg_label in sampled_negatives:
                    eval_pairs.append((text, neg_label))
                    eval_labels.append(0)
        
        logger.info(f"Created evaluation pairs: {len([l for l in eval_labels if l == 1])} positive, {len([l for l in eval_labels if l == 0])} negative")
        return eval_pairs, eval_labels
    
    def prepare_datasets(self):
        """Main method to prepare and save all datasets."""
        logger.info("Starting dataset preparation...")
        
        # Load raw data
        texts, labels = self.load_raw_data()
        
        # Filter rare labels
        texts, labels, frequent_labels = self.filter_rare_labels(texts, labels)
        
        # Split by labels (ensuring test has unseen labels)
        train_texts, train_labels, test_texts, test_labels, train_label_set, test_label_set = \
            self.split_by_labels(texts, labels, frequent_labels)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create contrastive datasets (anchor, positive pairs only)
        train_dataset = self.create_contrastive_dataset(
            train_texts, train_labels, train_label_set, "training"
        )
        test_dataset = self.create_contrastive_dataset(
            test_texts, test_labels, test_label_set, "testing"
        )
        
        # Create evaluation pairs
        train_eval_pairs, train_eval_labels = self.create_evaluation_pairs(
            train_texts, train_labels, train_label_set
        )
        test_eval_pairs, test_eval_labels = self.create_evaluation_pairs(
            test_texts, test_labels, test_label_set
        )
        
        # Save datasets
        train_dataset.save_to_disk(os.path.join(self.config.output_dir, "train_contrastive"))
        test_dataset.save_to_disk(os.path.join(self.config.output_dir, "test_contrastive"))
        
        # Save evaluation data
        eval_data = {
            "train_eval_pairs": train_eval_pairs,
            "train_eval_labels": train_eval_labels,
            "test_eval_pairs": test_eval_pairs,
            "test_eval_labels": test_eval_labels
        }
        
        with open(os.path.join(self.config.output_dir, "evaluation_data.json"), 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        # Save label mappings
        label_info = {
            "train_labels": list(train_label_set),
            "test_labels": list(test_label_set),
            "all_labels": list(frequent_labels),
            "label_counts": dict(self.label_counts),
            "num_train_texts": len(train_texts),
            "num_test_texts": len(test_texts),
            "num_train_pairs": len(train_dataset),
            "num_test_pairs": len(test_dataset)
        }
        
        with open(os.path.join(self.config.output_dir, "label_info.json"), 'w') as f:
            json.dump(label_info, f, indent=2)
        
        # Save raw splits for inspection
        raw_splits = {
            "train": [{"text": text, "labels": labels} for text, labels in zip(train_texts, train_labels)],
            "test": [{"text": text, "labels": labels} for text, labels in zip(test_texts, test_labels)]
        }
        
        with open(os.path.join(self.config.output_dir, "raw_splits.json"), 'w') as f:
            json.dump(raw_splits, f, indent=2)
        
        # Print summary
        logger.success("Dataset preparation completed!")
        logger.info("Summary:")
        logger.info(f"  Train labels: {len(train_label_set)}")
        logger.info(f"  Test labels: {len(test_label_set)} (unseen)")
        logger.info(f"  Train texts: {len(train_texts)}")
        logger.info(f"  Test texts: {len(test_texts)}")
        logger.info(f"  Train pairs: {len(train_dataset)}")
        logger.info(f"  Test pairs: {len(test_dataset)}")
        logger.info(f"  Output directory: {self.config.output_dir}")
        
        # Show some example test labels to verify they're different
        logger.info(f"Example train labels: {list(train_label_set)[:10]}")
        logger.info(f"Example test labels: {list(test_label_set)[:10]}")
        
        return {
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "train_labels": train_label_set,
            "test_labels": test_label_set,
            "label_info": label_info
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare datasets for zero-shot classification training")
    parser.add_argument("--batch-dir", type=str, required=True, help="Directory containing batch JSON files")
    parser.add_argument("--output-dir", type=str, default="data/prepared_datasets", help="Output directory for prepared datasets")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Ratio of labels to reserve for testing")
    parser.add_argument("--min-label-freq", type=int, default=3, help="Minimum label frequency to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    config = DatasetConfig(
        batch_dir=args.batch_dir,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        min_label_frequency=args.min_label_freq,
        random_seed=args.seed
    )
    
    preparator = DatasetPreparator(config)
    preparator.prepare_datasets()


if __name__ == "__main__":
    main()
