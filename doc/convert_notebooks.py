# To run this script, execute "python doc/convert_notebooks.py" from the root
# of the repository.

import glob
import os
import shutil
import subprocess
import sys

import nbformat
from nbconvert.preprocessors import ClearOutputPreprocessor

EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)
TUTORIALS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "tutorials")
)
# Downloadable copies of the notebooks inside the doc folder
NOTEBOOKS_DIR = os.path.join(TUTORIALS_DIR, "notebooks")

# Notebooks that are not ready for the website. Names are without the
# ".ipynb" extension.
SKIP_NOTEBOOKS = {
    "14_longitudinal_modelling",
}

os.makedirs(TUTORIALS_DIR, exist_ok=True)


def clean_tutorials_dir() -> None:
    """Remove previously generated RST files and support png's."""
    # Delete every generated .rst file in the tutorials output dir,
    # but not the index.rst.
    for f in glob.glob(os.path.join(TUTORIALS_DIR, "*.rst")):
        # Skip the index.rst
        if os.path.basename(f) == "index.rst":
            continue
        os.remove(f)
    # Delete every notebook-support directory that ends with "_files/".
    # This deletes all the png's
    for d in glob.glob(os.path.join(TUTORIALS_DIR, "*_files")):
        shutil.rmtree(d)
    # Delete the downloadable notebooks, so renamed or removed notebooks
    # do not linger on the website.
    if os.path.isdir(NOTEBOOKS_DIR):
        shutil.rmtree(NOTEBOOKS_DIR)


def convert_notebooks() -> None:
    """Convert every example notebook to RST using nbconvert.
    """
    # Collect all notebooks in the examples directory
    notebooks = glob.glob(os.path.join(EXAMPLES_DIR, "*.ipynb"))

    for nb_path in notebooks:
        # Derive the rst filename from the notebook name
        nb_name = os.path.splitext(os.path.basename(nb_path))[0]

        if nb_name in SKIP_NOTEBOOKS:
            print(f"Skipping {nb_name}")
            continue

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


def write_stripped_notebooks() -> None:
    """
    Write downloadable copies of the notebooks without their outputs.

    That way the notebooks take less memory and are faster to download.
    """
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

    for nb_path in glob.glob(os.path.join(EXAMPLES_DIR, "*.ipynb")):
        nb_name = os.path.splitext(os.path.basename(nb_path))[0]
        if nb_name in SKIP_NOTEBOOKS:
            continue

        notebook = nbformat.read(nb_path, as_version=4)
        
        # remove the outputs and execution counts from the notebook
        notebook, _ = ClearOutputPreprocessor().preprocess(notebook, {})

        # remove the Papermill metadata that CI injects
        notebook.metadata.pop("papermill", None)
        for cell in notebook.cells:
            cell.get("metadata", {}).pop("papermill", None)
            cell.get("metadata", {}).pop("execution", None)

        nbformat.write(
            notebook, os.path.join(NOTEBOOKS_DIR, f"{nb_name}.ipynb")
        )


def insert_download_link() -> None:
    """Put a notebook download link just below each tutorial title."""
    for rst_path in glob.glob(os.path.join(TUTORIALS_DIR, "*.rst")):
        nb_name = os.path.splitext(os.path.basename(rst_path))[0]
        if nb_name == "index":
            continue

        if not os.path.exists(os.path.join(NOTEBOOKS_DIR, f"{nb_name}.ipynb")):
            continue

        with open(rst_path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()

        title_end = find_title_end(lines)
        if title_end is None:
            # The notebook has no heading to put the link under, so it
            # also does not render as a proper page. Flag it rather than
            # silently shipping a tutorial without a download link.
            print(f"No title in {nb_name}, skipping download link")
            continue

        # add a download link just below the title with a container class 
        # that can be styled in CSS.
        link = [
            "",
            ".. container:: notebook-download",
            "",
            "   :download:`Download Jupyter notebook "
            f"<notebooks/{nb_name}.ipynb>`",
        ]
        lines[title_end + 1 : title_end + 1] = link

        with open(rst_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def find_title_end(lines: list[str]) -> int | None:
    """Find the line index of a document title's underline.

    nbconvert writes the notebook's first heading as an RST title: a line
    of text followed by a line of repeated punctuation of at least the
    same length.

    Parameters
    ----------
    lines : list[str]
        Lines of the generated RST file.

    Returns
    -------
    int | None
        Index of the underline, or None if no title was found.
    """
    for i, line in enumerate(lines[:-1]):
        text, underline = line.strip(), lines[i + 1].strip()
        if not text or not underline:
            continue
        # An underline is one punctuation character repeated, spanning at
        # least the title text.
        if (
            underline[0] in "=-~^\"'`#*+"
            and underline == underline[0] * len(underline)
            and len(underline) >= len(text)
        ):
            return i + 1
    return None


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
    write_stripped_notebooks()
    insert_download_link()
    if os.name == "nt": # Only fix image paths on Windows
        fix_image_paths()
