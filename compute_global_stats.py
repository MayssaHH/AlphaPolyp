import os
import numpy as np
import pickle
import csv
import argparse

def load_label_map(csv_file):
    """Load label map from CSV"""
    label_map = {}
    try:
        with open(csv_file, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                try:
                    label_map[row['Filename']] = [
                        float(row['logVolume']),
                        float(row['x']),
                        float(row['y']),
                        float(row['z'])
                    ]
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping invalid row in CSV: {e}")
                    continue
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return None
    
    print(f"Loaded {len(label_map)} labels from CSV")
    return label_map

def filter_labeled(img_dir, label_map):
    """Return sorted list of filenames that have labels"""
    out = []
    if not os.path.exists(img_dir):
        print(f"ERROR: Image directory does not exist: {img_dir}")
        return out
        
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(('.jpg','.png','.jpeg')): 
            continue
        base = os.path.splitext(fn)[0]
        base_no_zeros = base.lstrip('0')  # Remove leading zeros
        if base_no_zeros in label_map:
            out.append(fn)
    return sorted(out)

def get_reg_labels(file_list, label_map):
    """Return regression labels for file list"""
    regs = []
    for fname in file_list:
        base = os.path.splitext(fname)[0]
        base_no_zeros = base.lstrip('0')  # Remove leading zeros
        if base_no_zeros in label_map:
            regs.append(label_map[base_no_zeros])
    return np.array(regs, dtype=np.float32)

def compute_global_statistics(real_img_dir, synth_img_dir, csv_labels):
    """Compute global regression statistics across all data"""
    print("Computing Global Regression Statistics")
    
    # Load label map
    label_map = load_label_map(csv_labels)
    if label_map is None:
        return None
    # Get file lists
    real_files = filter_labeled(real_img_dir, label_map)
    synth_files = filter_labeled(synth_img_dir, label_map)
    if not real_files or not synth_files:
        print("ERROR: No labeled files found")
        return None
    
    print(f"Found {len(real_files)} real files and {len(synth_files)} synthetic files")
    print(real_files)
    print(synth_files)
    # Collect all regression labels
    all_regs = []
    
    print("Processing real files...")
    real_regs = get_reg_labels(real_files, label_map)
    print(real_regs)
    all_regs.append(real_regs)
    
    print("Processing synthetic files...")
    synth_regs = get_reg_labels(synth_files, label_map)
    all_regs.append(synth_regs)
    print(all_regs)
    # Combine and compute statistics
    if all_regs:
        all_regs = np.vstack(all_regs)
        stats = {
            'min': np.min(all_regs, axis=0),
            'max': np.max(all_regs, axis=0),
            'mean': np.mean(all_regs, axis=0),
            'std': np.std(all_regs, axis=0),
            'range': np.max(all_regs, axis=0) - np.min(all_regs, axis=0)
        }
        
        # Check that stats arrays are not empty before printing
        if (len(stats['min']) == 0 or len(stats['max']) == 0 or len(stats['mean']) == 0):
            print("No statistics available to display (empty data).")
        else:
            print("\nGlobal regression statistics:")
            print(f"  Volume: min={stats['min'][0]:.2f}, max={stats['max'][0]:.2f}, mean={stats['mean'][0]:.2f}")
            print(f"  X-dim:  min={stats['min'][1]:.2f}, max={stats['max'][1]:.2f}, mean={stats['mean'][1]:.2f}")
            print(f"  Y-dim:  min={stats['min'][2]:.2f}, max={stats['max'][2]:.2f}, mean={stats['mean'][2]:.2f}")
            print(f"  Z-dim:  min={stats['min'][3]:.2f}, max={stats['max'][3]:.2f}, mean={stats['mean'][3]:.2f}")
        
        return stats
    else:
        print("ERROR: No regression data found")
        return None

def main():
    """Main function to compute global statistics"""
    # Argument parsing
    parser = argparse.ArgumentParser(description='Compute global regression statistics for AlphaPolyp dataset')
    parser.add_argument('--root', type=str, required=True, help='Root path to data (drive_base)')
    parser.add_argument('--csv', type=str, required=True, help='CSV file with labels')
    args = parser.parse_args()

    drive_base = args.root
    real_img_dir = os.path.join(drive_base, 'cyclegan_images')
    synth_img_dir = os.path.join(drive_base, 'images')
    csv_labels = args.csv
    
    # Validate paths
    paths_to_check = [
        (real_img_dir, "Real images directory"),
        (synth_img_dir, "Synthetic images directory"),
        (csv_labels, "CSV labels file")
    ]
    
    missing_paths = []
    for path, description in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(f"{description}: {path}")
    
    if missing_paths:
        print("ERROR: Missing required paths:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\nPlease ensure all paths exist before running.")
        return
    
    # Compute statistics
    stats = compute_global_statistics(real_img_dir, synth_img_dir, csv_labels)
    
    if stats is not None:
        # Save statistics
        output_file = 'global_regression_stats.pkl'
        pickle.dump(stats, open(output_file, 'wb'))
        print(f"\nSaved global regression statistics to {output_file}")
        
        # Also save as text for easy viewing
        txt_file = 'global_regression_stats.txt'
        with open(txt_file, 'w') as f:
            f.write("Global Regression Statistics\n")
            f.write("===========================\n\n")
            if (len(stats['min']) == 0 or len(stats['max']) == 0 or len(stats['mean']) == 0 or len(stats['std']) == 0):
                f.write("No statistics available to display (empty data).\n")
            else:
                f.write(f"Volume: min={stats['min'][0]:.6f}, max={stats['max'][0]:.6f}, mean={stats['mean'][0]:.6f}, std={stats['std'][0]:.6f}\n")
                f.write(f"X-dim:  min={stats['min'][1]:.6f}, max={stats['max'][1]:.6f}, mean={stats['mean'][1]:.6f}, std={stats['std'][1]:.6f}\n")
                f.write(f"Y-dim:  min={stats['min'][2]:.6f}, max={stats['max'][2]:.6f}, mean={stats['mean'][2]:.6f}, std={stats['std'][2]:.6f}\n")
                f.write(f"Z-dim:  min={stats['min'][3]:.6f}, max={stats['max'][3]:.6f}, mean={stats['mean'][3]:.6f}, std={stats['std'][3]:.6f}\n")
        
        print(f"Saved statistics summary to {txt_file}")
    else:
        print("Failed to compute global statistics")

if __name__ == "__main__":
    main() 