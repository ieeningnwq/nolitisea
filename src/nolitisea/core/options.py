"""Command-line option parsing helpers.

Replaces ``check_option`` / ``scan_help`` / ``test_outfile`` from the
TISEAN routines.
"""


def check_option(argv, name, n_args):
    """Find an option and consume its arguments.

    Parameters
    ----------
    argv : list[str]
        Argument vector to scan.
    name : str
        Option flag (e.g. ``"-m"``).
    n_args : int
        Number of arguments the option expects.

    Returns
    -------
    str or list[str] or None
        The option value(s), or ``None`` when the option is absent.
    """
    raise NotImplementedError


def scan_help(argv):
    """Detect a ``-h`` help flag in ``argv``.

    Parameters
    ----------
    argv : list[str]
        Argument vector.

    Returns
    -------
    bool
        ``True`` when help is requested.
    """
    raise NotImplementedError


def test_outfile(path):
    """Verify that ``path`` is writable.

    Parameters
    ----------
    path : str
        Output file path.

    Returns
    -------
    bool
        ``True`` when the file can be written.
    """
    raise NotImplementedError
