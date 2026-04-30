import os
from pathlib import Path

# ─────────────────────────────────────────
# Dataset paths
# ─────────────────────────────────────────
MONTGOMERY_PATH = Path("datasets/montgomery/CXR_png")
SHENZHEN_PATH   = Path("datasets/shenzhen/CXR_png")

def explore_dataset(path, name):
    print(f"\n{'='*50}")
    print(f"📁 Dataset: {name}")
    print(f"{'='*50}")

    if not path.exists():
        print(f"❌ ERROR: Folder not found at {path}")
        return [], []

    # Get all PNG files
    all_images = list(path.glob("*.png"))

    if len(all_images) == 0:
        print(f"❌ ERROR: No PNG images found in {path}")
        print("   Make sure images are directly inside the folder")
        return [], []

    # Separate TB and Normal based on filename
    # _0.png = Normal, _1.png = TB
    normal = [f for f in all_images if f.stem.endswith('_0')]
    tb     = [f for f in all_images if f.stem.endswith('_1')]
    other  = [f for f in all_images
              if not f.stem.endswith('_0')
              and not f.stem.endswith('_1')]

    print(f"✅ Total images  : {len(all_images)}")
    print(f"   Normal (0)   : {len(normal)}")
    print(f"   TB     (1)   : {len(tb)}")

    if other:
        print(f"   ⚠️  Unknown    : {len(other)}")
        print(f"   Unknown files: {[f.name for f in other[:3]]}")

    # Show sample filenames
    print(f"\n📄 Sample Normal files:")
    for f in normal[:3]:
        print(f"   {f.name}")

    print(f"\n📄 Sample TB files:")
    for f in tb[:3]:
        print(f"   {f.name}")

    return normal, tb


def main():
    print("\n🔍 Lung3D AI — Dataset Explorer")
    print("Checking your datasets...\n")

    # Explore both datasets
    mont_normal, mont_tb = explore_dataset(MONTGOMERY_PATH, "Montgomery")
    shen_normal, shen_tb = explore_dataset(SHENZHEN_PATH,   "Shenzhen")

    # Combined summary
    print(f"\n{'='*50}")
    print(f"📊 COMBINED SUMMARY")
    print(f"{'='*50}")

    total_normal = len(mont_normal) + len(shen_normal)
    total_tb     = len(mont_tb)     + len(shen_tb)
    total        = total_normal + total_tb

    print(f"✅ Total images  : {total}")
    print(f"   Normal       : {total_normal}")
    print(f"   TB           : {total_tb}")

    if total > 0:
        tb_pct     = round(total_tb     / total * 100, 1)
        normal_pct = round(total_normal / total * 100, 1)
        print(f"\n📈 Class Balance:")
        print(f"   Normal : {normal_pct}%")
        print(f"   TB     : {tb_pct}%")

        if abs(tb_pct - normal_pct) > 20:
            print(f"\n⚠️  Warning: Dataset is imbalanced!")
            print(f"   We'll handle this during training.")
        else:
            print(f"\n✅ Dataset is reasonably balanced!")

    print(f"\n{'='*50}")
    if total > 0:
        print("✅ Dataset looks good! Ready for training.")
    else:
        print("❌ Dataset has issues. Fix them before continuing.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()