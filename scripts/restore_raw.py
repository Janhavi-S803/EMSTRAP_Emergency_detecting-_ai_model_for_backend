from pathlib import Path
import shutil

raw_dir = Path("dataset/raw")
train_dir = Path("dataset/train")
val_dir = Path("dataset/validation")

raw_dir.mkdir(parents=True, exist_ok=True)

for source in [train_dir, val_dir]:
    for cls in source.iterdir():
        if cls.is_dir():
            target = raw_dir / cls.name
            target.mkdir(parents=True, exist_ok=True)

            for img in cls.iterdir():
                if img.is_file():
                    destination = target / img.name

                    # avoid duplicates
                    if not destination.exists():
                        shutil.copy2(img, destination)

print("✓ Raw dataset restored successfully!")