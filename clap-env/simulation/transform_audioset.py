#!/usr/bin/env python3
import sys
import os
import pandas as pd
import numpy as np

SYS_SALT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if SYS_SALT_ROOT not in sys.path:
    sys.path.append(SYS_SALT_ROOT)

from salt import compare_labels
from py_salt import event_mapping

e = event_mapping.EventExplorer()

# Original parquet file - DO NOT MODIFY
ORIGINAL_PARQUET = "/home/lucaa/urban-noise-classification/audioset/audioset_eval.parquet"


def create_top_parents_file():
    df = pd.read_parquet(ORIGINAL_PARQUET)
    
    def get_top_parent(label):
        try:
            std_label = e.get_std_label_from_dataset_label(label.strip())
            paths = e.get_paths_to_label(std_label)
            if paths and len(paths[0]) > 0:
                top = paths[0][0]
                # Special case: map "water" to "natural_sounds" since water sounds are natural
                if top == 'water':
                    return 'natural_sounds'
                # Return the top parent for all other labels
                return top
        except:
            pass
        return None
    
    def transform_labels(human_labels):
        # Handle numpy arrays, lists, or single values
        if human_labels is None or (isinstance(human_labels, float) and pd.isna(human_labels)):
            return []
        
        # Convert to list if it's a numpy array or other iterable
        if not isinstance(human_labels, list):
            try:
                human_labels = list(human_labels)
            except:
                human_labels = [human_labels]
        
        # Transform each label to its top parent
        top_parents = [get_top_parent(str(label)) for label in human_labels]
        return [p for p in top_parents if p is not None]
    
    df['human_labels'] = df['human_labels'].apply(transform_labels)
    
    output_path = os.path.join(os.path.dirname(__file__), "audioset_eval_top.parquet")
    df.to_parquet(output_path, index=False)


def create_mixed_and_non_mixed_files():
    input_path = os.path.join(os.path.dirname(__file__), "audioset_eval_top.parquet")
    df = pd.read_parquet(input_path)
    
    salt_to_audioset = {
        'human_sounds': 'Human sounds',
        'animal': 'Animal',
        'music': 'Music',
        'natural_sounds': 'Natural sounds',
        'source-ambiguous_sounds': 'Source-ambiguous sounds',
        'channel_environment_and_background': 'Channel, environment and background',
        'sound_of_things': 'Sounds of things'
    }
    
    def process_labels_mixed(human_labels):
        if human_labels is None:
            return []
        if not isinstance(human_labels, list):
            try:
                human_labels = list(human_labels)
            except:
                human_labels = [human_labels]
        
        unique_labels = list(set(human_labels))
        if len(unique_labels) > 1:
            result = ['Mixed']
        else:
            result = unique_labels
        
        return [salt_to_audioset.get(label, label) for label in result]
    
    def process_labels_non_mixed(human_labels):
        if human_labels is None:
            return []
        if not isinstance(human_labels, list):
            try:
                human_labels = list(human_labels)
            except:
                human_labels = [human_labels]
        
        unique_labels = list(set(human_labels))
        return [salt_to_audioset.get(label, label) for label in unique_labels]
    
    # Create mixed version
    df_mixed = df.copy()
    df_mixed['human_labels'] = df_mixed['human_labels'].apply(process_labels_mixed)
    output_path_mixed = os.path.join(os.path.dirname(__file__), "audioset_eval_top_mixed.parquet")
    df_mixed.to_parquet(output_path_mixed, index=False)
    
    # Create non-mixed version
    df_non_mixed = df.copy()
    df_non_mixed['human_labels'] = df_non_mixed['human_labels'].apply(process_labels_non_mixed)
    output_path_non_mixed = os.path.join(os.path.dirname(__file__), "audioset_eval_top_non_mixed.parquet")
    df_non_mixed.to_parquet(output_path_non_mixed, index=False)


def remove_redundant_parents():
    df = pd.read_parquet(ORIGINAL_PARQUET)
    
    def get_label_path(label):
        try:
            std_label = e.get_std_label_from_dataset_label(str(label).strip())
            paths = e.get_paths_to_label(std_label)
            if paths and len(paths[0]) > 0:
                return paths[0]  # Return the path as a list
        except:
            pass
        return None
    
    def remove_parents(human_labels):
        if human_labels is None or (isinstance(human_labels, float) and pd.isna(human_labels)):
            return []
        
        # Convert to list if needed
        if not isinstance(human_labels, list):
            try:
                human_labels = list(human_labels)
            except:
                human_labels = [human_labels]
        
        if len(human_labels) == 0:
            return []
        
        # Get paths for all labels
        label_paths = {}
        for label in human_labels:
            path = get_label_path(label)
            if path:
                label_paths[label] = path
        
        # Find labels to keep (those with no children in the list)
        labels_to_keep = []
        for label, path in label_paths.items():
            is_parent = False
            # Check if any other label's path starts with this label's path (is a descendant)
            for other_label, other_path in label_paths.items():
                if label != other_label and len(other_path) > len(path):
                    # Check if other_path starts with path
                    if other_path[:len(path)] == path:
                        is_parent = True
                        break
            
            if not is_parent:
                labels_to_keep.append(label)
        
        return labels_to_keep if labels_to_keep else human_labels
    
    df['human_labels'] = df['human_labels'].apply(remove_parents)
    
    output_path = os.path.join(os.path.dirname(__file__), "audioset_eval_no_redundant_parents.parquet")
    df.to_parquet(output_path, index=False)


def remove_mixed_rows():
    input_path = os.path.join(os.path.dirname(__file__), "audioset_eval_top_mixed.parquet")
    df = pd.read_parquet(input_path)
    
    def has_mixed(human_labels):
        if human_labels is None:
            return False
        if isinstance(human_labels, (list, np.ndarray)):
            return "Mixed" in human_labels
        return human_labels == "Mixed"
    
    # Filter out rows with "Mixed"
    df_filtered = df[~df['human_labels'].apply(has_mixed)]
    
    # Count all labels in filtered dataset
    all_labels = []
    for labels in df_filtered['human_labels']:
        if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
            all_labels.extend(labels)
        elif labels is not None and not (isinstance(labels, float) and pd.isna(labels)):
            all_labels.append(labels)
    
    label_counts = pd.Series(all_labels).value_counts()
    total_labels = len(all_labels)
    
    output_path = os.path.join(os.path.dirname(__file__), "audioset_eval_top_mixed_no_mixed.parquet")
    df_filtered.to_parquet(output_path, index=False)
    
    print(f"Removed {len(df) - len(df_filtered)} rows with 'Mixed'. Saved {len(df_filtered)} rows to {output_path}")
    print(f"\nLabel distribution (Total labels: {total_labels}):")
    print(f"{'Label':<50} {'Count':<15} {'Percentage':<10}")
    print("-" * 75)
    for label, count in label_counts.items():
        pct = (count / total_labels) * 100 if total_labels > 0 else 0.0
        print(f"{label:<50} {count:<15} {pct:>6.2f}%")


if __name__ == "__main__":
    remove_mixed_rows()
