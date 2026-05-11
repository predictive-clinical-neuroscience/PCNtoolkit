from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import BSpline

from pcntoolkit.util.output import Errors, Output
from pcntoolkit.util.migration import registry

def create_basis_function(
    basis_type: str | dict | None,
    basis_column: int = 0,
    **kwargs,
) -> BasisFunction:
    if isinstance(basis_type, dict):
        return BasisFunction.from_dict(basis_type)
    elif basis_type in ["polynomial", "PolynomialBasisFunction"]:
        return PolynomialBasisFunction(basis_column, **kwargs)
    elif basis_type in ["bspline", "BsplineBasisFunction"]:
        new_knots = kwargs.pop("knots", [])
        if isinstance(new_knots, list):
            new_knots = np.array(new_knots)
        return BsplineBasisFunction(basis_column, **kwargs, knots=new_knots)
    elif basis_type in ["Composite", "CompositeBasis"]:
        parts = [BasisFunction.from_dict(p) for p in kwargs['parts']]
        return CompositeBasisFunction(parts)
    else:
        return LinearBasisFunction(basis_column)

class BasisFunction(ABC):
    def __init__(
        self,
        basis_column: int = 0,
        **kwargs,
    ):
        self.basis_column = basis_column
        self.is_fitted: bool = kwargs.get("is_fitted", False)
        self.basis_name: str = kwargs.get("basis_name", "basis")
        self.min: float = kwargs.get("min", 0)
        self.max: float = kwargs.get("max", 1)
        self.compute_min: bool = self.min == 0
        self.compute_max: bool = self.max == 1

    @classmethod
    def from_dict(
        cls, my_dict: dict, version: str | None = None
    ) -> "BasisFunction":
        # Apply any registered BasisFunction migrations for this version.
        my_dict = registry.migrate(
            "BasisFunction", my_dict, version=version
        )
        basis_function_type = my_dict["basis_function"]
        basis_function = create_basis_function(basis_function_type, **my_dict)
        return basis_function

    @classmethod
    def from_args(cls, name: str, args: dict) -> BasisFunction:
        basis_function_type = args.pop(name, "linear")
        if basis_function_type == "bspline":
            nknots = args.pop(f"{name}_nknots", 3)
            degree = args.pop(f"{name}_degree", 3)
            left_expand = args.pop(f"{name}_left_expand", 0.05)
            right_expand = args.pop(f"{name}_right_expand", 0.05)
            knot_method = args.pop(f"{name}_knot_method", "uniform")
            basis_column = args.pop(f"{name}_basis_column", None)
            return BsplineBasisFunction(
                basis_column=basis_column,
                degree=degree,
                nknots=nknots,
                left_expand=left_expand,
                right_expand=right_expand,
                knot_method=knot_method,
            )
        elif basis_function_type == "polynomial":
            degree = args.pop(f"{name}_degree", 3)
            basis_column = args.pop(f"{name}_basis_column", None)
            return PolynomialBasisFunction(basis_column=basis_column, degree=degree)
        else:
            basis_column = args.pop(f"{name}_basis_column", None)
            return LinearBasisFunction(basis_column=basis_column)

    def fit(self, X: np.ndarray) -> None:
        if len(X.shape) == 1:
            X = X[:,None]
        array = X[:, self.basis_column]
        if self.compute_min:
            self.min = np.min(array)
        if self.compute_max:
            self.max = np.max(array)
        self._fit(array)
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(Output.error(Errors.ERROR_BASIS_FUNCTION_NOT_FITTED))
        if len(X.shape) == 1:
            X = X[:,None]
        all_arrays = []
        for i in range(X.shape[1]):
            if i == self.basis_column:
                array = X[:, i]
                squeezed = np.squeeze(array)
                if squeezed.ndim > 1:
                    raise ValueError(Output.error(Errors.ERROR_DATA_MUST_BE_1D))
                transformed_array = self._transform(array)
                if transformed_array.ndim == 1:
                    transformed_array = transformed_array.reshape(-1, 1)
                all_arrays.append(transformed_array)
            else:
                copied_array = copy.deepcopy(X[:, i])
                all_arrays.append(copied_array[:, None])
        return np.concatenate(all_arrays, axis=1)


    @abstractmethod
    def _fit(self, data: np.ndarray) -> None:
        pass

    @abstractmethod
    def _transform(self, data: np.ndarray) -> np.ndarray:
        pass

    def to_dict(self) -> dict:
        mydict = copy.deepcopy(self.__dict__)
        mydict["basis_function"] = self.__class__.__name__
        return mydict

    @property
    @abstractmethod
    def dimension(self):
        pass

class PolynomialBasisFunction(BasisFunction):
    def __init__(
        self,
        basis_column: int = 0,
        degree: int = 3,
        **kwargs,
    ):
        super().__init__(basis_column, **kwargs)
        self.degree = degree
        self.basis_name = "poly"

    def _fit(self, data: np.ndarray) -> None:
        pass

    def _transform(self, data: np.ndarray) -> np.ndarray:
        transformed_array = np.power.outer(data, np.arange(1, self.degree + 1))
        return transformed_array

    @property
    def dimension(self):
        return self.degree


class BsplineBasisFunction(BasisFunction):
    def __init__(
        self,
        basis_column: int = 0,
        degree: int = 3,
        nknots: int = 5,
        left_expand: float = 0.05,
        right_expand: float = 0.05,
        knot_method: str = "uniform",
        knots:  np.ndarray | list = None,  # type: ignore
        **kwargs,
    ):
        super().__init__(basis_column, **kwargs)
        self.degree = degree
        self.nknots = nknots
        self.left_expand = left_expand
        self.right_expand = right_expand
        self.knot_method = knot_method
        if knots is not None:
            self.knots = list(knots)
        else:
            self.knots = None 
        self.basis_name = "bspline"

    def _fit(self, data: np.ndarray) -> None:
        mymin = self.min
        mymax = self.max
        delta = mymax - mymin
        aug_min = mymin - delta * self.left_expand
        aug_max = mymax + delta * self.right_expand
        if self.knot_method == "uniform":
            knots = np.linspace(aug_min, aug_max, self.nknots)
        elif self.knot_method == "quantile":
            knots = np.percentile(data, np.linspace(0, 100, self.nknots))
        knots = np.concatenate([[aug_min] * self.degree, knots, [aug_max] * self.degree])
        self.knots = list(knots)

    def _transform(self, data: np.ndarray) -> np.ndarray:
        spline = BSpline.design_matrix(data, self.knots, self.degree, extrapolate=True).toarray()
        return np.concatenate((data.reshape(-1, 1), spline), axis=1)

    def to_dict(self) -> dict:
        mydict = super().to_dict()
        mydict["degree"] = self.degree
        mydict["nknots"] = self.nknots
        mydict["left_expand"] = self.left_expand
        mydict["right_expand"] = self.right_expand
        mydict["knot_method"] = self.knot_method
        if self.knots is not None:
            mydict["knots"] = list(self.knots)    
        else:
            mydict["knots"] = None       
        return mydict

    @property
    def dimension(self):
        return self.degree + self.nknots


class LinearBasisFunction(BasisFunction):
    def __init__(self, basis_column: int = 0, **kwargs):
        super().__init__(basis_column, **kwargs)
        self.basis_name = "linear"

    def _fit(self, data: np.ndarray) -> None:
        pass

    def _transform(self, data: np.ndarray) -> np.ndarray:
        return data

    @property
    def dimension(self):
        return 1

class CompositeBasisFunction(BasisFunction):
    def __init__(self, parts):
        super().__init__(basis_column=0)
        self.parts = parts

    def _fit(self, data, i):
        pass

    def _transform(self, data, i):
        pass

    def fit(self, X):
        if len(X.shape) == 1:
            X = X[:,None]
        for bf in self.parts:
            basis_column = bf.basis_column
            bf.basis_column = 0
            bf.fit(X[:,basis_column])
            bf.basis_column = basis_column
        self.is_fitted = True

    def transform(self, X):
        if len(X.shape) == 1:
            X = X[:,None]
        mats = []
        for c in range(X.shape[1]):
            transformed = False
            for bf in self.parts:
                if bf.basis_column == c:
                    bf.basis_column = 0
                    mats.append(bf.transform(X[:,c]))
                    bf.basis_column=c
                    transformed=True
            if not transformed:
                mats.append(X[:,c])
        return np.concatenate(mats, axis=1)

    def to_dict(self):
        return {"basis_function": "CompositeBasis", "parts": [bf.to_dict() for bf in self.parts]}

    @property
    def dimension(self):
        return sum([p.dimension for p in self.parts])