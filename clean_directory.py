"""Delete files that are not .mat files or that contain '_registered' in the filename.

Usage:
    python clean_directory.py <directory>

Only keeps .mat files that do NOT have '_registered' in the name.
Prints what will be deleted and asks for confirmation before proceeding.
"""
import os
import sys


def get_files_to_delete(directory):
    to_delete = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if not name.endswith(".mat") or "_registered" in name:
            to_delete.append(path)
    return to_delete


def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_directory.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Error: '%s' is not a directory" % directory)
        sys.exit(1)

    to_delete = get_files_to_delete(directory)

    if not to_delete:
        print("No files to delete.")
        return

    print("The following %d files will be deleted:" % len(to_delete))
    for f in to_delete:
        print("  " + f)

    answer = input("\nProceed? (y/n): ").strip().lower()
    if answer == "y":
        for f in to_delete:
            os.remove(f)
        print("Deleted %d files." % len(to_delete))
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
