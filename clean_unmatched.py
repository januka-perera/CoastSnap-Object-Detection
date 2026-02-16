"""Delete images that do not have a corresponding .mat file, matched by ID.

The ID is the first part of the filename (before the first dot), e.g.
  1598992500.Wed.Sep.02_06_35_00.AEST.2020.manly.snap.CherylWhite.jpg -> ID: 1598992500
  1598992500.Wed.Sep.02_06_35_00.AEST.2020.manly.plan.CherylWhite.mat -> ID: 1598992500

Usage:
    python clean_unmatched.py <images_dir> <mat_dir>
"""
import os
import sys


def get_id(filename):
    """Extract the numeric ID (first part before the dot) from a filename."""
    return filename.split(".")[0]


def main():
    if len(sys.argv) != 3:
        print("Usage: python clean_unmatched.py <images_dir> <mat_dir>")
        sys.exit(1)

    images_dir = sys.argv[1]
    mat_dir = sys.argv[2]

    if not os.path.isdir(images_dir):
        print("Error: '%s' is not a directory" % images_dir)
        sys.exit(1)
    if not os.path.isdir(mat_dir):
        print("Error: '%s' is not a directory" % mat_dir)
        sys.exit(1)

    # Collect IDs from .mat files
    mat_ids = set()
    for name in os.listdir(mat_dir):
        if name.endswith(".mat"):
            mat_ids.add(get_id(name))

    # Find images without a matching .mat ID
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    to_delete = []
    kept = 0
    for name in os.listdir(images_dir):
        path = os.path.join(images_dir, name)
        if not os.path.isfile(path):
            continue
        if not name.lower().endswith(image_exts):
            continue
        if get_id(name) not in mat_ids:
            to_delete.append(path)
        else:
            kept += 1

    print("Mat files found: %d" % len(mat_ids))
    print("Images with match: %d" % kept)
    print("Images without match: %d" % len(to_delete))

    if not to_delete:
        print("No files to delete.")
        return

    print("\nFiles to delete:")
    for f in to_delete:
        print("  " + os.path.basename(f))

    answer = input("\nProceed? (y/n): ").strip().lower()
    if answer == "y":
        for f in to_delete:
            os.remove(f)
        print("Deleted %d files." % len(to_delete))
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
