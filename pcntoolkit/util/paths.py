"""Path-related utilities for PCNtoolkit."""

import os


def get_default_home_dir() -> str:
    """Get the default home directory for PCNtoolkit.

    Returns
    -------
    str
        The default home directory path
    """
    if home_dir := os.environ.get("PCN_HOME_DIR"):
        return home_dir

    # Try user's home directory
    home_dir = os.path.expanduser("~/.pcntoolkit")
    try:
        os.makedirs(home_dir, exist_ok=True)
        return home_dir
    except (OSError, PermissionError):
        pass

    # Fallback to local directory
    return "./pcntoolkit"


def get_default_log_dir() -> str:
    """Get the default log directory for PCNtoolkit.

    Returns
    -------
    str
        The default log directory path
    """
    # Try user's home directory
    home_log_dir = os.path.join(get_default_home_dir(), "logs")
    try:
        os.makedirs(home_log_dir, exist_ok=True)
        return home_log_dir
    except (OSError, PermissionError):
        pass

    # Fallback to local directory
    return "./logs"


def get_default_temp_dir() -> str:
    """Get the default temp directory for PCNtoolkit.

    The temp directory is determined in the following order:
    1. PCN_TEMP_DIR environment variable if set
    2. ~/.pcntoolkit/temp if the directory exists or can be created
    3. ./temp as a fallback

    Returns
    -------
    str
        The default temp directory path
    """
    # Try user's home directory
    home_temp_dir = os.path.join(get_default_home_dir(), "temp")
    try:
        os.makedirs(home_temp_dir, exist_ok=True)
        return home_temp_dir
    except (OSError, PermissionError):
        pass

    # Fallback to local directory
    return "./temp"


def get_default_save_dir() -> str:
    """Get the default save directory for normative models.

    The save directory is determined in the following order:
    1. PCN_SAVE_DIR environment variable if set
    2. ~/.pcntoolkit/saves if the directory exists or can be created
    3. ./saves as a fallback

    Returns
    -------
    str
        The default save directory path
    """
    # Try user's home directory
    home_save_dir = os.path.join(get_default_home_dir(), "saves")
    try:
        os.makedirs(home_save_dir, exist_ok=True)
        return home_save_dir
    except (OSError, PermissionError):
        pass

    # Fallback to local directory
    return "./saves"


def ensure_dir_exists(path: str) -> None:
    """Ensure that a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str
        The directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_save_subdirs(save_dir: str) -> tuple[str, str, str]:
    """Get the standard subdirectories for saving model data.

    Parameters
    ----------
    save_dir : str
        The base save directory

    Returns
    -------
    tuple[str, str, str]
        Tuple of (model_dir, results_dir, plots_dir)
    """
    model_dir = os.path.join(save_dir, "model")
    results_dir = os.path.join(save_dir, "results")
    plots_dir = os.path.join(save_dir, "plots")
    return model_dir, results_dir, plots_dir
