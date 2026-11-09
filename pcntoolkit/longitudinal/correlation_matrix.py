"""Age-to-age z-score correlation matrices for longitudinal scores."""

from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from pcntoolkit.math_functions.velocity import compute_correlation_matrix

from .longitudinal_score import LongitudinalScore

# Import heavy toolkit types only during static type checking.
if TYPE_CHECKING:
    from pcntoolkit.dataio.norm_data import NormData


class CorrelationMatrix:
    """How strongly a person's z-scores at two timepoints correlate.


    Do not call ``CorrelationMatrix(...)`` directly. Use ``compute`` to
    estimate one from your data, or ``load`` to read saved CSV files.

    Examples
    --------
    Estimate a matrix from your own longitudinal controls:

    >>> from pcntoolkit import CorrelationMatrix
    >>> corr = CorrelationMatrix.compute(controls, bandwidth=5)
    >>> corr.n_subjects
    67

    Save it as one CSV per region, and load the folder back later:

    >>> corr.save("my_matrices")
    >>> corr = CorrelationMatrix.load("my_matrices")

    Or load a single region:

    >>> corr = CorrelationMatrix.load("my_matrices/rh_G_front_middle_thickness.csv")

    Either way, pass it to a velocity model:

    >>> from pcntoolkit import VelocityModel
    >>> velocity = VelocityModel(model, corr)

    Parameters
    ----------
    matrix : xr.DataArray
        The correlations, filled in by ``compute`` and ``load``.
    covariate : str, default "age"
        Covariate the matrix is indexed by.
    bandwidth : int, optional
        Widest age gap that has a correlation. Wider gaps are 0.
    max_correlation : float, default 0.99
        Cap on correlations read with ``get_correlation``, so the z-gain never
        divides by zero (as it divides by sqrt(1 - r²))
    n_subjects : int, optional
        Number of subjects the matrix was estimated from.
    estimated_range : tuple of int, optional
        ``(min, max)`` ages actually seen in those subjects. Values outside it
        are extrapolated. The matrix itself always starts at age 0.

    Attributes
    ----------
    matrix : xr.DataArray
        The correlations.

    Raises
    ------
    ValueError
        If ``max_correlation`` is not strictly between 0 and 1.
    """

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
        """Estimate the correlation matrix from your own longitudinal data.

        .. warning::
           This method is still under development and might change in the future.

        Parameters
        ----------
        data : NormData
            Subjects with two or more visits and z-scores already predicted.
        bandwidth : int, default 5
            Widest age gap, in years, to estimate correlations for.
            Correlations for wider gaps are set to 0.
        covariate : str, default "age"
            Covariate to compute correlations over.
        max_correlation : float, default 0.99
            Cap on correlations read with ``get_correlation``, so the z-gain never
            divides by zero (as it divides by sqrt(1 - r²))

        Returns
        -------
        CorrelationMatrix
            The matrix, plus the number of subjects and the age range it was
            estimated from.

        Raises
        ------
        ValueError
            If ``data`` has no predictions, is not longitudinal, or if
            ``max_correlation`` is out of range.
        """
        # Correlation matrix computation needs Z-scores to be present in the data.
        LongitudinalScore._check_is_predicted(data, required=("Z",))
        LongitudinalScore._check_is_longitudinal(data)

        matrix = compute_correlation_matrix(data, bandwidth, covariate)

        # Record the ages actually observed. The matrix coordinates always run
        # from 0, so anything below the youngest subject is extrapolated.
        observed = np.round(
            np.asarray(data.X.sel(covariates=covariate).values, dtype=float)
        ).astype(int)
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
    def load(
        cls,
        path: str,
        *,
        covariate: str = "age",
        max_correlation: float = 0.99,
    ) -> CorrelationMatrix:
        """Load correlation matrices from a folder of CSVs, or from one CSV.

        Each CSV holds one response variable and is named after it, as
        written by ``save``::

            <path>/<response_var>.csv

        A folder loads every ``.csv`` in it, so keep only correlation matrices
        there. A file loads that one response variable. The file name, without
        ``.csv``, must match the response variable in the normative model the
        matrix is used with.

        Only the correlations are stored, so ``bandwidth``, ``n_subjects`` and
        ``estimated_range`` are ``None``.

        Parameters
        ----------
        path : str
            Folder written by ``save``, or one CSV inside it.
        covariate : str, default "age"
            Covariate the matrix is indexed by.
        max_correlation : float, default 0.99
            Cap on correlations read with ``get_correlation``, so the z-gain never
            divides by zero (as it divides by sqrt(1 - r²))

        Returns
        -------
        CorrelationMatrix
            The loaded matrix, with one entry along ``response_vars`` per CSV.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist, or is a folder without ``.csv`` files.
        ValueError
            If a CSV's header row and first column are not the covariate
            values 0, 1, ..., N-1, if the CSVs differ in size, or if a CSV has
            empty cells, correlations outside -1 to 1, or is not symmetric.
        """
        source = Path(path)
        if source.is_dir():
            csv_paths = sorted(source.glob("*.csv"))
            if not csv_paths:
                raise FileNotFoundError(
                    f"'{path}' holds no .csv files. Write correlation matrices "
                    "there with CorrelationMatrix.save()."
                )
        elif source.is_file():
            csv_paths = [source]
        else:
            raise FileNotFoundError(f"'{path}' does not exist.")
        # The response variable names are derived from the CSV file names.
        response_vars = [csv_path.stem for csv_path in csv_paths]

        dfs = []
        for csv_path in csv_paths:
            # round_trip parsing: the default parser is a hair imprecise, which
            # would make a loaded matrix score differently from the saved one.
            df = pd.read_csv(csv_path, index_col=0, float_precision="round_trip")
            # get_correlation() reads cells by position, so covariate value c
            # must sit in row c and column c.
            expected = [str(c) for c in range(len(df))]
            if list(df.columns) != expected or list(df.index.astype(str)) != expected:
                raise ValueError(
                    f"'{csv_path}' is not a correlation matrix: its header row "
                    "and first column must both be the covariate values "
                    f"0, 1, ..., {len(df) - 1}."
                )
            cls._check_correlation_matrix_is_valid(
                df.to_numpy(dtype=float), csv_path, covariate
            )
            dfs.append(df)

        first = dfs[0]
        for response_var, df in zip(response_vars, dfs, strict=True):
            if df.shape != first.shape:
                raise ValueError(
                    f"'{response_var}' has shape {df.shape}, but "
                    f"'{response_vars[0]}' has {first.shape}. Every "
                    "response variable must cover the same covariate values."
                )

        # Stack into the 3-D layout produced by compute_correlation_matrix.
        values = np.stack([df.to_numpy(dtype=float) for df in dfs])
        coords = np.arange(values.shape[1])
        matrix = xr.DataArray(
            values,
            dims=("response_vars", f"{covariate}_1", f"{covariate}_2"),
            coords={
                "response_vars": response_vars,
                f"{covariate}_1": coords,
                f"{covariate}_2": coords,
            },
        )

        return cls(matrix, covariate=covariate, max_correlation=max_correlation)

    @classmethod
    def load_velocity_pickle(cls, path: str) -> CorrelationMatrix:
        """Load a correlation matrix from a pickled velocity model.

        Note: this reads one specific layout: the velocity models produced by
        Johanna Bayer's pipeline. It is not a general-purpose loader. Use it
        once to convert such a file, then save it with ``save`` and load
        the result with ``load``.

        Those files hold a dict whose ``A_sparse_predict`` entry is the
        correlations, stored lower-triangular with a zero diagonal. They are
        mirrored and given a unit diagonal here to match the layout produced by
        ``compute_correlation_matrix``.

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
            The loaded matrix.

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
        return cls(matrix)

    @staticmethod
    def _check_correlation_matrix_is_valid(
        values: np.ndarray, source: Path, covariate: str
    ) -> None:
        """Check that the correlation matrix is valid.

        Raises a ValueError if there are empty cells, values outside [-1, 1],
        or if the matrix is not symmetric.
        """
        empty = np.argwhere(np.isnan(values))
        if empty.size:
            i, j = empty[0]
            raise ValueError(
                f"'{source}' has an empty cell at {covariate} {i} and {j}. "
                "Every cell must hold a correlation."
            )

        out_of_range = np.argwhere(np.abs(values) > 1)
        if out_of_range.size:
            i, j = out_of_range[0]
            raise ValueError(
                f"'{source}' has a correlation of {values[i, j]:g} at {covariate} "
                f"{i} and {j}. Correlations must be between -1 and 1."
            )

        if not np.allclose(values, values.T):
            raise ValueError(
                f"'{source}' is not symmetric. The correlation at "
                f"({covariate}_1, {covariate}_2) must equal the one at "
                f"({covariate}_2, {covariate}_1)."
            )

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
            raise ValueError(
                f"'A_sparse_predict' must be a square matrix; got shape {lower.shape}."
            )

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
        """Save the matrix as one CSV per response variable in a folder::

            <path>/<response_var>.csv

        Each CSV has the covariate values (0, 1, ..., N-1) as both the header
        row and the first column, with an empty top-left cell.

        Parameters
        ----------
        path : str
            Folder to write to. Created if it does not exist.

        Returns
        -------
        None
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        response_vars = [str(r) for r in self.matrix.coords["response_vars"].values]
        coords = np.asarray(self.matrix.coords[f"{self.covariate}_1"].values, dtype=int)

        for response_var in response_vars:
            df = pd.DataFrame(
                self.matrix.sel(response_vars=response_var).values,
                index=coords,
                columns=coords,
            )
            # Keep full precision and use \n line endings on every OS.
            df.to_csv(directory / f"{response_var}.csv", lineterminator="\n")

    def get_correlation(
        self, response_var: str, timepoint_1: int, timepoint_2: int
    ) -> float:
        """Read the correlation between two timepoints for one response variable.

        For example, ``corr.get_correlation("lh_thickness", 40, 41)`` is how strongly a
        z-score at age 40 predicts the z-score at age 41. Ages beyond the
        matrix are moved to its nearest edge.

        Parameters
        ----------
        response_var : str
            Response variable to read.
        timepoint_1, timepoint_2 : int
            The two timepoints to get the correlation for.

        Returns
        -------
        float
            The correlation, kept between ``-max_correlation`` and
            ``+max_correlation`` (default 0.99).

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
        clamped = [int(np.clip(t, 0, max_cov)) for t in (timepoint_1, timepoint_2)]

        # Warn when reading outside the range the cohort actually covered. No
        # subject was observed there, so the value is extrapolated.
        if self.estimated_range is not None:
            low, high = self.estimated_range
            outside = [t for t in (timepoint_1, timepoint_2) if not low <= t <= high]
            if outside:
                warnings.warn(
                    f"{self.covariate} {outside} falls outside the range "
                    f"correlations were estimated over ({low}-{high}). "
                    "The value used is extrapolated.",
                    UserWarning,
                    stacklevel=2,
                )

        value = float(
            self.matrix.sel(response_vars=response_var).values[clamped[0], clamped[1]]
        )
        return float(np.clip(value, -self.max_correlation, self.max_correlation))
