"""Age-to-age z-score correlation matrices for longitudinal scores."""

from __future__ import annotations

import json
import pickle
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pcntoolkit.math_functions.velocity import compute_correlation_matrix

# Import heavy toolkit types only during static type checking.
if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData


class CorrelationMatrix:
    """Age-to-age z-score correlation matrix estimated from a longitudinal
    cohort.

    The matrix records how strongly a subject's z-scores at two ages correlate.

    Do not call ``CorrelationMatrix(...)`` directly. Use :meth:`compute` to
    estimate one from a cohort, or :meth:`load` to read one from a file.

    Examples
    --------
    Estimate a matrix from your own longitudinal controls:

    >>> from pcntoolkit import CorrelationMatrix
    >>> corr = CorrelationMatrix.compute(controls, bandwidth=5)
    >>> corr.n_subjects
    67

    Save it, and load it back later:

    >>> corr.save("my_matrix")
    >>> corr = CorrelationMatrix.load("my_matrix")

    Either way, pass it to a score:

    >>> from pcntoolkit import ZGainScore
    >>> zgain = ZGainScore(model, corr)

    Parameters
    ----------
    matrix : xr.DataArray
        Correlation matrix, filled in by :meth:`compute` and :meth:`load`.
    covariate : str, default "age"
        Covariate the matrix is indexed by.
    bandwidth : int, optional
        Age-offset range within which correlations were estimated directly.
        Wider offsets were interpolated.
    max_correlation : float, default 0.99
        Upper bound applied by :meth:`get`, keeping the correlation away from 1
        so that ``sqrt(1 - r**2)`` stays away from zero.
    n_subjects : int, optional
        Number of subjects in the cohort the matrix was estimated from.
    estimated_range : tuple of int, optional
        ``(min, max)`` covariate values actually observed in that cohort. Not
        the same as the matrix coordinates, which always start at 0.

    Attributes
    ----------
    matrix : xr.DataArray
        The correlations.

    Raises
    ------
    ValueError
        If ``max_correlation`` is not strictly between 0 and 1.
    """

    _METADATA_FILE = "correlation_matrix.json"

    def __init__(
        self,
        matrix: xr.DataArray,
        *,
        covariate: str = "age",
        bandwidth: int | None = None,
        max_correlation: float = 0.99,
        n_subjects: int | None = None,
        estimated_range: tuple[int, int] | None = None,
    ) -> None:
        # Keep the clipping threshold inside a mathematically safe range.
        if not 0.0 < max_correlation < 1.0:
            raise ValueError("max_correlation must be strictly between 0 and 1.")

        self.matrix = matrix
        self.covariate = covariate
        self.bandwidth = bandwidth
        self.max_correlation = max_correlation
        self.n_subjects = n_subjects
        self.estimated_range = estimated_range

    @classmethod
    def compute(
        cls,
        data: NormData,
        *,
        bandwidth: int = 5,
        covariate: str = "age",
        max_correlation: float = 0.99,
    ) -> CorrelationMatrix:
        """Estimate a correlation matrix from a longitudinal cohort.

        Parameters
        ----------
        data : NormData
            Longitudinal cohort with predicted z-scores and repeated visits per
            subject.
        bandwidth : int, default 5
            Age-offset range, in years, for direct correlation estimates.
            Larger offsets are interpolated.
        covariate : str, default "age"
            Covariate to index the matrix by.
        max_correlation : float, default 0.99
            Upper bound applied by :meth:`get`.

        Returns
        -------
        CorrelationMatrix
            The estimated matrix, carrying the cohort size and observed
            covariate range.

        Raises
        ------
        ValueError
            If ``data`` has no predictions, is not longitudinal, or if
            ``max_correlation`` is out of range.
        """
        # Imported here to avoid a circular import at module load.
        from pcntoolkit.longitudinal_score.longitudinal_score import (
            LongitudinalScore,
        )

        LongitudinalScore._check_is_predicted(data)
        LongitudinalScore._check_is_longitudinal(data)

        matrix = compute_correlation_matrix(data, bandwidth, covariate)

        # Record the ages actually observed. The matrix coordinates always run
        # from 0, so anything below the youngest subject is extrapolated.
        observed = np.round(np.asarray(data.X.sel(covariates=covariate).values, dtype=float)).astype(int)
        estimated_range = (int(observed.min()), int(observed.max()))

        return cls(
            matrix,
            covariate=covariate,
            bandwidth=bandwidth,
            max_correlation=max_correlation,
            n_subjects=len(np.unique(data.get_subject_ids())),
            estimated_range=estimated_range,
        )

    @classmethod
    def load(cls, path: str) -> CorrelationMatrix:
        """Load a correlation matrix from a directory written by :meth:`save`.

        The directory holds one CSV per response variable plus a
        ``correlation_matrix.json`` with the metadata::

            <path>/correlation_matrix.json
            <path>/<response_var>.csv

        Parameters
        ----------
        path : str
            Directory written by :meth:`save`.

        Returns
        -------
        CorrelationMatrix
            The loaded matrix, with the metadata recorded at save time.

        Raises
        ------
        FileNotFoundError
            If ``path`` holds no ``correlation_matrix.json``.
        """
        directory = Path(path)
        metadata_path = directory / cls._METADATA_FILE

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"'{path}' is not a saved correlation matrix: it has no "
                f"{cls._METADATA_FILE}. Write one with CorrelationMatrix.save()."
            )

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        covariate = metadata.get("covariate", "age")
        response_vars = list(metadata["response_vars"])

        # One CSV per response variable, stacked back into the 3-D layout
        # produced by compute_correlation_matrix.
        frames = []
        for response_var in response_vars:
            csv_path = directory / f"{response_var}.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(
                    f"'{metadata_path}' lists response variable "
                    f"'{response_var}', but '{csv_path}' is missing."
                )
            # round_trip parsing: the default parser is a hair imprecise, which
            # would make a loaded matrix score differently from the saved one.
            frames.append(
                pd.read_csv(csv_path, index_col=0, float_precision="round_trip")
            )

        # Coordinates come from the CSV header, so they survive a round-trip
        # even if the matrix does not start at 0.
        first = frames[0]
        coords = np.asarray(first.columns, dtype=int)
        for response_var, frame in zip(response_vars, frames, strict=True):
            if frame.shape != first.shape:
                raise ValueError(
                    f"'{response_var}.csv' has shape {frame.shape}, but "
                    f"'{response_vars[0]}.csv' has {first.shape}. Every "
                    "response variable must cover the same covariate values."
                )

        values = np.stack([frame.to_numpy(dtype=float) for frame in frames])
        matrix = xr.DataArray(
            values,
            dims=("response_vars", f"{covariate}_1", f"{covariate}_2"),
            coords={
                "response_vars": response_vars,
                f"{covariate}_1": coords,
                f"{covariate}_2": coords,
            },
        )

        estimated_range = metadata.get("estimated_range")
        return cls(
            matrix,
            covariate=covariate,
            bandwidth=metadata.get("bandwidth"),
            max_correlation=metadata.get("max_correlation", 0.99),
            n_subjects=metadata.get("n_subjects"),
            estimated_range=tuple(estimated_range) if estimated_range else None,
        )

    @classmethod
    def load_velocity_pickle(cls, path: str) -> CorrelationMatrix:
        """Load a correlation matrix from a pickled velocity model.

        .. note::
           This reads one specific layout: the velocity models produced by
           Johanna Bayer's pipeline. It is not a general-purpose loader. Use it
           once to convert such a file, then save it with :meth:`save` and load
           the result with :meth:`load`.

        Those files hold a dict whose ``A_sparse_predict`` entry is the
        correlations, stored lower-triangular with a zero diagonal. They are
        mirrored and given a unit diagonal here to match the layout produced by
        :func:`~pcntoolkit.math_functions.velocity.compute_correlation_matrix`.

        The region name is not inside the file. It appears only in the directory
        name, laid out as ``.../batch_<n>_<region>/Velocity/<file>.pkl``, so it
        is read from there. The region must match the one the normative model
        was fitted on, since scoring selects the matrix by that label.

        Parameters
        ----------
        path : str
            Path to the ``.pkl`` file, inside a ``batch_<n>_<region>``
            directory.

        Returns
        -------
        CorrelationMatrix
            The loaded matrix. ``n_subjects`` and ``estimated_range`` are
            ``None``, as the file does not carry them.

        Raises
        ------
        KeyError
            If the file has no ``A_sparse_predict`` entry.
        ValueError
            If no ``batch_<n>_<region>`` directory is found in ``path``.
        """
        response_var = cls._response_var_from_path(path)

        with open(path, "rb") as f:
            file_contents = pickle.load(f)

        matrix = cls._matrix_from_velocity_file(file_contents, response_var)
        # Non-zero entries only reach so far from the diagonal; that distance is
        # the bandwidth the correlations were estimated over.
        values = matrix.values[0]
        rows, cols = np.asarray(values).nonzero()
        bandwidth = int(np.abs(rows - cols).max()) if rows.size else None

        return cls(matrix, bandwidth=bandwidth)

    @staticmethod
    def _response_var_from_path(path: str) -> str:
        """Read the region name out of a ``batch_<n>_<region>`` directory."""
        # Walk up from the file: the batch directory is a parent, not always the
        # immediate one (files sit in .../batch_1_lh_G_front/Velocity/x.pkl).
        for part in reversed(Path(path).resolve().parts):
            match = re.fullmatch(r"batch_\d+_(.+)", part)
            if match:
                return match.group(1)
        raise ValueError(
            f"Could not read a region name from '{path}'. This loader expects a "
            "'batch_<n>_<region>' directory in the path, as written by the "
            "velocity modelling pipeline."
        )

    @staticmethod
    def _matrix_from_velocity_file(
        file_contents: Any,
        response_var: str,
    ) -> xr.DataArray:
        """Convert the contents of a velocity file into a labelled correlation matrix."""
        if not hasattr(file_contents, "get") or "A_sparse_predict" not in file_contents:
            available = (
                list(file_contents.keys())
                if hasattr(file_contents, "keys")
                else type(file_contents).__name__
            )
            raise KeyError(
                "The pickle has no 'A_sparse_predict' entry, so it does not "
                f"contain correlations this can read. Found: {available}."
            )

        sparse = file_contents["A_sparse_predict"]
        # Accept both scipy sparse matrices and plain arrays.
        lower = np.asarray(
            sparse.toarray() if hasattr(sparse, "toarray") else sparse,
            dtype=float,
        )
        if lower.ndim != 2 or lower.shape[0] != lower.shape[1]:
            raise ValueError(f"'A_sparse_predict' must be a square matrix; got shape {lower.shape}.")

        # Stored as the lower triangle with an empty diagonal.
        full = lower + lower.T
        np.fill_diagonal(full, 1.0)

        ages = np.arange(full.shape[0])
        return xr.DataArray(
            full[np.newaxis, :, :],
            dims=("response_vars", "age_1", "age_2"),
            coords={
                "response_vars": [response_var],
                "age_1": ages,
                "age_2": ages,
            },
        )

    def save(self, path: str) -> None:
        """Save the matrix as CSV files plus a JSON metadata file.

        Writes one CSV per response variable, with the covariate values as both
        the header row and the index column, and a ``correlation_matrix.json``
        holding everything the CSVs cannot carry::

            <path>/correlation_matrix.json
            <path>/<response_var>.csv

        The format is plain text so the correlations can be read without
        pcntoolkit, and so that loading one does not execute code the way
        unpickling does.

        Parameters
        ----------
        path : str
            Directory to write to. Created if it does not exist.

        Returns
        -------
        None
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        response_vars = [str(r) for r in self.matrix.coords["response_vars"].values]
        coords = np.asarray(self.matrix.coords[f"{self.covariate}_1"].values, dtype=int)

        for response_var in response_vars:
            frame = pd.DataFrame(
                self.matrix.sel(response_vars=response_var).values,
                index=coords,
                columns=coords,
            )
            frame.index.name = self.covariate
            # No float_format: pandas then writes the shortest string that reads
            # back as the same float64, so a saved matrix gives bit-identical
            # scores to the one in memory.
            frame.to_csv(directory / f"{response_var}.csv")

        metadata = {
            "covariate": self.covariate,
            "response_vars": response_vars,
            "bandwidth": self.bandwidth,
            "max_correlation": self.max_correlation,
            "n_subjects": self.n_subjects,
            "estimated_range": (
                list(self.estimated_range) if self.estimated_range else None
            ),
        }
        with open(directory / self._METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    def get(self, response_var: str, cov_1: int, cov_2: int) -> float:
        """Read one correlation, clamped to the matrix and clipped for safety.

        Parameters
        ----------
        response_var : str
            Response variable to read.
        cov_1, cov_2 : int
            The two covariate values (e.g. ages) to correlate.

        Returns
        -------
        float
            The correlation, clipped to ``+/- max_correlation`` so that
            ``sqrt(1 - r**2)`` stays away from zero.

        Raises
        ------
        KeyError
            If the matrix holds no correlations for the model's ``response_var``.
        """
        # check that the response variable from the model is available in the 
        # correlation matrix too.
        available = [str(r) for r in self.matrix.coords["response_vars"].values]
        if response_var not in available:
            raise KeyError(
                f"This correlation matrix has no data for '{response_var}'. "
                f"It holds: {available}. The matrix and the normative model "
                "must cover the same regions."
            )

        max_cov = int(self.matrix[f"{self.covariate}_1"].max())
        clamped = [int(np.clip(c, 0, max_cov)) for c in (cov_1, cov_2)]

        # Warn when reading outside the range the cohort actually covered. No
        # subject was observed there, so the value is extrapolated.
        if self.estimated_range is not None:
            low, high = self.estimated_range
            outside = [c for c in (cov_1, cov_2) if not low <= c <= high]
            if outside:
                warnings.warn(
                    f"{self.covariate} {outside} falls outside the range "
                    f"correlations were estimated over ({low}-{high}). "
                    "The value used is extrapolated.",
                    UserWarning,
                    stacklevel=2,
                )

        value = float(self.matrix.sel(response_vars=response_var).values[clamped[0], clamped[1]])
        return float(np.clip(value, -self.max_correlation, self.max_correlation))
