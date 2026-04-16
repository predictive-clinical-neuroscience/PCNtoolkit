# To run this script, execute "python doc/convert_notebooks.py" from the root
# of the repository.

import glob
import os
import shutil
import subprocess
import sys

EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)
TUTORIALS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "pages", "tutorials")
)

os.makedirs(TUTORIALS_DIR, exist_ok=True)


def clean_tutorials_dir() -> None:
    """Remove previously generated RST files and support png's."""
    # Delete every generated .rst file in the tutorials output dir
    for f in glob.glob(os.path.join(TUTORIALS_DIR, "*.rst")):
        os.remove(f)
    # Delete every notebook-support directory that ends with "_files/".
    # This deletes all the png's
    for d in glob.glob(os.path.join(TUTORIALS_DIR, "*_files")):
        shutil.rmtree(d)


def convert_notebooks() -> None:
    """Convert every example notebook to RST using nbconvert.
    """
    # Collect all notebooks in the examples directory
    notebooks = glob.glob(os.path.join(EXAMPLES_DIR, "*.ipynb"))

    for nb_path in notebooks:
        # Derive the rst filename from the notebook name
        nb_name = os.path.splitext(os.path.basename(nb_path))[0]

        subprocess.run(
            [
                sys.executable, # resolve "jupyter" from the virtual
                                # environment that is running this script
                "-m",
                "jupyter",
                "nbconvert", # nbconvert needs the package "pandoc". Please
                             # install with
                             # "conda install -c conda-forge pandoc"
                "--to",
                "rst",
                nb_path,
                "--output",
                nb_name,
                "--output-dir",
                TUTORIALS_DIR,
            ],
            check=True,
        )


def fix_image_paths() -> None:
    """Fix image paths in generated RST files in Windows.
    """    
    # Iterate over every generated RST file in the tutorials dir
    for rst_path in glob.glob(os.path.join(TUTORIALS_DIR, "*.rst")):
        # Read the file content as text
        with open(rst_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        # Fix Windows paths
        fixed = content.replace("%5C", "/")

        # Only write back if the content actually changed
        if fixed != content:
            with open(rst_path, "w", encoding="utf-8") as fh:
                fh.write(fixed)


if __name__ == "__main__":
    clean_tutorials_dir()
    convert_notebooks()
    if os.name == "nt": # Only fix image paths on Windows
        fix_image_paths()
