import numpy as np
from astropy.table import Table


def load_sky_catalog(path, ra_col="RA", dec_col="DEC", extra_cols=None):
    """Load a galaxy catalog with sky coordinates from a FITS file.

    Reads a FITS table and returns sky positions as numpy arrays.
    Additional columns (e.g. redshift, velocity, mass) can be requested
    by name and are returned in a dictionary.

    Args:
        path (str or Path): Path to the FITS file.
        ra_col (str): Name of the right ascension column. Must be in
            degrees. Defaults to ``"RA"``.
        dec_col (str): Name of the declination column. Must be in
            degrees. Defaults to ``"DEC"``.
        extra_cols (list of str, optional): Names of additional columns
            to return, e.g. ``["Z", "MASS"]``. Column names must match
            exactly. Defaults to None.

    Returns:
        tuple: A tuple ``(ra, dec, extras)`` where

            - **ra** (*np.ndarray, shape (N,)*) -- Right ascension in
              degrees.
            - **dec** (*np.ndarray, shape (N,)*) -- Declination in
              degrees.
            - **extras** (*dict*) -- Dictionary of additional columns
              keyed by column name. Empty if ``extra_cols`` is None.

    Raises:
        ValueError: If ``ra_col``, ``dec_col``, or any column in
            ``extra_cols`` is not found in the file. The error message
            lists all available column names.

    Examples:
        Load positions only:

        >>> ra, dec, _ = load_sky_catalog("catalog.fits")

        Load positions and redshift:

        >>> ra, dec, extras = load_sky_catalog(
        ...     "catalog.fits",
        ...     ra_col="RA",
        ...     dec_col="DEC",
        ...     extra_cols=["Z"]
        ... )
        >>> redshift = extras["Z"]
    """

    table = Table.read(path)
    for col in [ra_col, dec_col] + (extra_cols or []):
        if col not in table.colnames:
            raise ValueError(
                f"Column '{col}' not found. "
                f"Available columns: {table.colnames}"
            )

    ra  = np.asarray(table[ra_col])
    dec = np.asarray(table[dec_col])

    extras = {}
    for col in (extra_cols or []):
        extras[col] = np.asarray(table[col])

    return ra, dec, extras



def load_cartesian_catalog(path, pos_key="pos", extra_keys=None):
    """Load a galaxy catalog with Cartesian positions from an npz file.

    Reads an npz file and returns galaxy positions as a numpy
    array of shape (N, 3). Positions are assumed to be comoving
    coordinates in units of Mpc/h. Additional arrays (e.g. velocities,
    masses) can be requested by key and are returned in a dictionary.

    Args:
        path (str or Path): Path to the npz file.
        pos_key (str): Key for the position array in the npz file.
            The array must have shape (N, 3). Defaults to ``"pos"``.
        extra_keys (list of str, optional): Keys of additional arrays
            to return, e.g. ``["vel", "mass"]``. Keys must match
            exactly. Defaults to None.

    Returns:
        tuple: A tuple ``(pos, extras)`` where

            - **pos** (*np.ndarray, shape (N, 3)*) -- Comoving Cartesian
              positions in Mpc/h.
            - **extras** (*dict*) -- Dictionary of additional arrays
              keyed by their npz key. Empty if ``extra_keys`` is None.

    Raises:
        KeyError: If ``pos_key`` or any key in ``extra_keys`` is not
            found in the npz file. The error message lists all available
            keys.
        ValueError: If the position array does not have shape (N, 3).

    Examples:
        Load positions only:

        >>> pos, _ = load_cartesian_catalog("catalog.npz")

        Load positions and line-of-sight velocities:

        >>> pos, extras = load_cartesian_catalog(
        ...     "catalog.npz",
        ...     pos_key="pos_g",
        ...     extra_keys=["v_los"]
        ... )
        >>> v_los = extras["v_los"]
    """
    
    data = np.load(path)

    if pos_key not in data:
        raise KeyError(
            f"Key '{pos_key}' not found in npz file."
            f"Available keys: {list(data.keys())}")

    pos = data[pos_key]    
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected pos array of shape (N, 3), got {pos.shape}.")

    extras = {}
    for key in (extra_keys or []):
        if key not in data:
            raise KeyError(
                f"Key '{key}' not found in npz file. "
                f"Available keys: {list(data.keys())}")
        extras[key] = data[key]
    
    return pos, extras