import random
import shutil
from pathlib import Path

def split_dataset():
    # Define paths
    base_dir = Path("dataset")
    raw_dir = base_dir / "raw"
    train_dir = base_dir / "train"
    val_dir = base_dir / "validation"
    test_dir = base_dir / "test"
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    
    # Find all class subdirectories across raw, train, validation, and test directories
    classes_set = set()
    for folder in [raw_dir, train_dir, val_dir, test_dir]:
        if folder.exists():
            for p in folder.iterdir():
                if p.is_dir():
                    classes_set.add(p.name)
                    
    classes = sorted(list(classes_set))
    
    if not classes:
        print("No class directories found.")
        return
        
    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    
    # Pre-scan to gather all image files and their current locations
    class_data = {}
    total_images_count = 0
    
    for class_name in classes:
        image_locations = {}
        for folder in [raw_dir, train_dir, val_dir, test_dir]:
            class_folder = folder / class_name
            if class_folder.exists():
                for f in class_folder.iterdir():
                    if f.is_file() and f.suffix.lower() in image_extensions:
                        # Store current path
                        image_locations[f.name] = f
                        
        # Get sorted list of names for deterministic shuffling
        sorted_names = sorted(list(image_locations.keys()))
        class_data[class_name] = {
            "sorted_names": sorted_names,
            "locations": image_locations
        }
        total_images_count += len(sorted_names)
        
    print(f"Total images found across all categories: {total_images_count}")
    
    # Track statistics
    summary = {}
    total_files_processed = 0
    
    # Set random seed for reproducibility of shuffle order across runs
    random.seed(42)
    
    for class_name in classes:
        sorted_names = class_data[class_name]["sorted_names"]
        image_locations = class_data[class_name]["locations"]
        
        # Shuffle deterministically
        random.shuffle(sorted_names)
        
        n = len(sorted_names)
        if n == 0:
            summary[class_name] = {"moved": 0, "skipped": 0, "train": 0, "val": 0, "test": 0}
            continue
            
        # Calculate split sizes
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        n_test = n - n_train - n_val
        
        # Ensure class directories in destination folders exist
        class_train_dir = train_dir / class_name
        class_val_dir = val_dir / class_name
        class_test_dir = test_dir / class_name
        
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_val_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0
        skipped_count = 0
        
        for i, img_name in enumerate(sorted_names):
            # Determine target directory
            if i < n_train:
                target_dir = class_train_dir
            elif i < n_train + n_val:
                target_dir = class_val_dir
            else:
                target_dir = class_test_dir
                
            target_path = target_dir / img_name
            current_path = image_locations[img_name]
            
            if current_path.resolve() == target_path.resolve():
                skipped_count += 1
            else:
                try:
                    shutil.move(str(current_path), str(target_path))
                    moved_count += 1
                except Exception as e:
                    print(f"\nError moving {current_path} to {target_path}: {e}")
                    
            total_files_processed += 1
            if total_files_processed % 500 == 0:
                print(f"Progress: Processed {total_files_processed}/{total_images_count} files...")
                
        summary[class_name] = {
            "moved": moved_count,
            "skipped": skipped_count,
            "train": n_train,
            "val": n_val,
            "test": n_test
        }
        
    # Print the split summary
    print("\nSplit Summary:")
    print("-" * 75)
    print(f"{'Class':<20} | {'Moved':<8} | {'Skipped':<8} | {'Train Target':<12} | {'Val Target':<10} | {'Test Target':<10}")
    print("-" * 75)
    for class_name, stats in summary.items():
        print(f"{class_name:<20} | {stats['moved']:<8} | {stats['skipped']:<8} | {stats['train']:<12} | {stats['val']:<10} | {stats['test']:<10}")
    print("-" * 75)

if __name__ == "__main__":
    split_dataset()
