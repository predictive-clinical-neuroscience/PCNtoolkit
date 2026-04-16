"""Render CITATION.cff from template and git history.

Reads contributor data from ``git log``, orders
authors by total lines added (descending), and
generates CITATION.cff from a Jinja2 template with
rich author metadata (ORCID, affiliations).
"""

import re
import subprocess
from collections import defaultdict
from datetime import date

import toml
from jinja2 import Template

# --------------------------------------------------
# CFF author metadata keyed by canonical name.
# Only authors with known ORCID or affiliations need
# an entry here.  Unknown git contributors will still
# appear in the output with just their name.
# --------------------------------------------------
AUTHOR_METADATA: dict[str, dict] = {
    "Andre Marquand": {
        "family-names": "Marquand",
        "given-names": "Andre",
        "orcid": (
            "https://orcid.org/0000-0001-5903-203X"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition, and Behavior",
            "Radboud University Medical Center",
            "King's College London",
        ],
    },
    "Stijn de Boer": {
        "family-names": "de Boer",
        "given-names": "Stijn",
        "orcid": (
            "https://orcid.org/0000-0002-8657-8959"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition and Behaviour",
            "Radboud University Medical Center",
        ],
    },
    "Konstantinos Tsilimparis": {
        "family-names": "Tsilimparis",
        "given-names": "Konstantinos",
        "orcid": (
            "https://orcid.org/0009-0008-5734-7538"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition and Behaviour",
            "Radboud University Medical Center",
        ],
    },
    "Seyed Mostafa Kia": {
        "family-names": "Kia",
        "given-names": "Seyed Mostafa",
        "orcid": (
            "https://orcid.org/0000-0002-7128-814X"
        ),
        "affiliations": [
            "Tilburg University",
            "Donders Institute for Brain,"
            " Cognition, and Behavior",
            "University Medical Center Utrecht",
        ],
    },
    "Saige Rutherford": {
        "family-names": "Rutherford",
        "given-names": "Saige",
        "orcid": (
            "https://orcid.org/0000-0003-3006-9044"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition, and Behavior",
            "Radboud University Medical Center",
            "University of Michigan",
        ],
    },
    "Charlotte Fraza": {
        "family-names": "Fraza",
        "given-names": "Charlotte",
        "orcid": (
            "https://orcid.org/0000-0002-7088-9250"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition, and Behavior",
            "Radboud University Medical Center",
        ],
    },
    "Barbora Rehak Buckova": {
        "family-names": "Rehak Buckova",
        "given-names": "Barbora",
        "orcid": (
            "https://orcid.org/0000-0001-5619-3946"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition, and Behavior",
            "Radboud University Medical Center",
            "National Institute of Mental Health:"
            " Klecany, CZ",
            "Czech Technical University in Prague:"
            " Prague, CZ",
            "Institute of Computer Science:"
            " Prague, CZ",
        ],
    },
    "Pieter Barkema": {
        "family-names": "Barkema",
        "given-names": "Pieter",
        "affiliations": [
            "University College London",
            "Donders Center for Brain,"
            " Cognition, and Behavior",
        ],
    },
    "Thomas Wolfers": {
        "family-names": "Wolfers",
        "given-names": "Thomas",
    },
    "Johanna Bayer": {
        "family-names": "Bayer",
        "given-names": "Johanna",
        "orcid": (
            "https://orcid.org/0000-0003-4891-6256"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition and Behaviour",
        ],
    },
    "Maarten Mennes": {
        "family-names": "Mennes",
        "given-names": "Maarten",
        "orcid": (
            "https://orcid.org/0000-0002-7279-3439"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition and Behaviour",
            "Radboud University",
            "SBGneuro Ltd.",
        ],
    },
    "Hester Huijsdens": {
        "family-names": "Huijsdens",
        "given-names": "Hester",
        "orcid": (
            "https://orcid.org/0000-0001-7039-8390"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition and Behaviour",
        ],
    },
    "Pierre Berthet": {
        "family-names": "Berthet",
        "given-names": "Pierre",
        "orcid": (
            "https://orcid.org/0000-0002-6878-6842"
        ),
        "affiliations": [
            "Donders Institute for Brain,"
            " Cognition and Behaviour",
            "University of Oslo",
            "Stockholm University",
            "Universite de Bordeaux",
            "Universite Pierre Mendes-France",
            "Universite Savoie Mont-Blanc",
        ],
    }
}
# --------------------------------------------------
# Map git author names → canonical names.
# Handles contributors who commit under different
# usernames or email-linked display names.
# --------------------------------------------------
NAME_MAP: dict[str, str] = {
    "amarquand": "Andre Marquand",
    "andre": "Andre Marquand",
    "Stijn": "Stijn de Boer",
    "Augub": "Stijn de Boer",
    "AuguB": "Stijn de Boer",
    "S.M.Kia": "Seyed Mostafa Kia",
    "contsili": "Konstantinos Tsilimparis",
    "saigerutherford": "Saige Rutherford",
    "RindKind": "Thomas Wolfers",
    "Hesterhuijsdens": "Hester Huijsdens",
    "pierre berthet": "Pierre Berthet",
    "Barbora": "Barbora Rehak Buckova",
    "Barbora Buckova": "Barbora Rehak Buckova",
    "likeajumprope": "Johanna Bayer",
    "PBarkema": "Pieter Barkema",
    "lindenmp": "Linden Parkes",
}

# Git names to exclude (bots, placeholder entries).
EXCLUDE: set[str] = {"dependabot[bot]", "="}


def get_lines_per_author() -> dict[str, int]:
    """Get total lines added per canonical author.

    Runs ``git log --numstat`` across all refs and
    sums the additions for each author, deduplicating
    names via NAME_MAP.

    Returns
    -------
    dict[str, int]
        Mapping of canonical author name to total
        lines added.
    """
    # Run git log with numstat to get per-file
    # insertions and deletions for every commit
    # that was merged into the current branch.
    result = subprocess.run(
        [
            "git", "log",
            "--format=%aN",
            "--numstat",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Accumulate lines added per author name.
    lines_by_author: dict[str, int] = defaultdict(int)
    current_author: str | None = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        # Skip empty lines between commits.
        if not stripped:
            continue
        # Numstat lines: adds<TAB>dels<TAB>filename.
        # Binary files show "-" instead of numbers.
        match = re.match(
            r"^(\d+|-)\t(\d+|-)\t", stripped
        )
        if match:
            # Only count text-file additions.
            if (
                current_author
                and match.group(1) != "-"
            ):
                additions = int(match.group(1))
                # Resolve to canonical name.
                canonical = NAME_MAP.get(
                    current_author, current_author
                )
                lines_by_author[canonical] += additions
        else:
            # This line is an author name.
            current_author = stripped
    return dict(lines_by_author)


def format_author_yaml(
    name: str,
    meta: dict | None,
) -> str:
    """Format a single author as CFF YAML lines.

    Parameters
    ----------
    name : str
        Canonical author name.
    meta : dict | None
        Author metadata dict, or None for unknown
        authors whose names are guessed from git.

    Returns
    -------
    str
        Indented YAML lines for one author entry.
    """
    yaml_lines: list[str] = []
    if meta:
        # Use the curated metadata.
        yaml_lines.append(
            '  - family-names: "'
            f'{meta["family-names"]}"'
        )
        yaml_lines.append(
            '    given-names: "'
            f'{meta["given-names"]}"'
        )
        if "orcid" in meta:
            yaml_lines.append(
                f'    orcid: "{meta["orcid"]}"'
            )
        for aff in meta.get("affiliations", []):
            yaml_lines.append(
                f'    affiliation: "{aff}"'
            )
    else:
        # Unknown author: split on last space.
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            yaml_lines.append(
                f'  - family-names: "{parts[1]}"'
            )
            yaml_lines.append(
                f'    given-names: "{parts[0]}"'
            )
        else:
            # Single-word name / username.
            yaml_lines.append(
                f'  - family-names: "{name}"'
            )
    return "\n".join(yaml_lines)


def build_authors_yaml(
    lines_by_author: dict[str, int],
) -> str:
    """Build CFF authors YAML ordered by lines added.

    Parameters
    ----------
    lines_by_author : dict[str, int]
        Canonical name → total lines added.

    Returns
    -------
    str
        YAML string for the ``authors:`` section body.
    """
    # Sort by lines contributed (descending).
    sorted_authors = sorted(
        lines_by_author.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    # Build a YAML entry for each contributor.
    entries: list[str] = []
    for name, lines in sorted_authors:
        # Skip excluded names.
        if name in EXCLUDE:
            continue
        meta = AUTHOR_METADATA.get(name)
        entries.append(format_author_yaml(name, meta))
        # Print contributor ranking to stdout.
        print(f"  {name}: {lines} lines added")
    return "\n".join(entries)


def main() -> None:
    """Render CITATION.cff from template and git."""
    # Load version from pyproject.toml.
    pyproject = toml.load("pyproject.toml")
    meta = pyproject["project"]
    version = meta["version"]

    # Collect contributor stats from git history.
    print("Collecting git contributor stats...")
    lines_by_author = get_lines_per_author()
    authors_yaml = build_authors_yaml(lines_by_author)

    # Template context with version, date, DOI, and
    # the generated authors YAML block.
    context = {
        "version": version,
        "date": date.today().isoformat(),
        "doi": "10.5281/zenodo.5207839",
        "authors_yaml": authors_yaml,
    }

    # Load and render the Jinja2 template.
    with open(
        "citation.cff.in", encoding="utf-8"
    ) as f:
        template = Template(f.read())

    # Write rendered output to CITATION.cff.
    with open(
        "CITATION.cff", "w", encoding="utf-8"
    ) as f:
        f.write(template.render(**context))

    print("CITATION.cff generated.")


if __name__ == "__main__":
    main()
