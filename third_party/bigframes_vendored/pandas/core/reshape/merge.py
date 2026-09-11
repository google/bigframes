# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/reshape/merge.py
"""
SQL-style merge routines
"""

from __future__ import annotations


def merge(
    left,
    right,
    how="inner",
    on=None,
    *,
    left_on=None,
    right_on=None,
    left_index: bool = False,
    right_index: bool = False,
    sort=False,
    suffixes=("_x", "_y"),
    indicator: bool | str = False,
):
    """
    Merge DataFrame objects with a database-style join.

    The join is done on columns or indexes. If joining columns on
    columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
    on indexes or indexes on a column or columns, the index will be passed on.
    When performing a cross merge, no column specifications to merge on are
    allowed.

    .. note::
        A named Series object is treated as a DataFrame with a single named column.

    .. warning::
        If both key columns contain rows where the key is a null value, those
        rows will be matched against each other. This is different from usual SQL
        join behaviour and can lead to unexpected results.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> df1 = bpd.DataFrame({'a': ['foo', 'bar'], 'b': [1, 2]})
        >>> df2 = bpd.DataFrame({'a': ['foo', 'baz'], 'c': [3, 4]})
        >>> bpd.merge(df1, df2, how='left', on='a')
             a  b     c
        0  foo  1     3
        1  bar  2  <NA>
        <BLANKLINE>
        [2 rows x 3 columns]

        >>> bpd.merge(df1, df2, how='left', on='a', indicator=True)
             a  b     c     _merge
        0  foo  1     3       both
        1  bar  2  <NA>  left_only
        <BLANKLINE>
        [2 rows x 4 columns]

        >>> bpd.merge(df1, df2, how='left', on='a', indicator='indicator_column')
             a  b     c indicator_column
        0  foo  1     3             both
        1  bar  2  <NA>        left_only
        <BLANKLINE>
        [2 rows x 4 columns]

    Args:
        left:
            The primary object to be merged.
        right:
            Object to merge with.
        how:
            ``{'left', 'right', 'outer', 'inner'}, default 'inner'``
            Type of merge to be performed.
            ``left``: use only keys from left frame, similar to a SQL left outer join;
            preserve key order.
            ``right``: use only keys from right frame, similar to a SQL right outer join;
            preserve key order.
            ``outer``: use union of keys from both frames, similar to a SQL full outer
            join; sort keys lexicographically.
            ``inner``: use intersection of keys from both frames, similar to a SQL inner
            join; preserve the order of the left keys.
            ``cross``: creates the cartesian product from both frames, preserves the order
            of the left keys.

        on (label or list of labels):
            Columns to join on. It must be found in both DataFrames. Either on or left_on + right_on
            must be passed in.
        left_on (label or list of labels):
            Columns to join on in the left DataFrame. Either on or left_on + right_on
            must be passed in.
        right_on (label or list of labels):
            Columns to join on in the right DataFrame. Either on or left_on + right_on
            must be passed in.
        left_index (bool, default False):
            Use the index from the left DataFrame as the join key.
        right_index (bool, default False):
            Use the index from the right DataFrame as the join key.
        sort:
            Default False. Sort the join keys lexicographically in the
            result DataFrame. If False, the order of the join keys depends
            on the join type (how keyword).
        suffixes:
            Default ``("_x", "_y")``. A length-2 sequence where each
            element is optionally a string indicating the suffix to add to
            overlapping column names in `left` and `right` respectively.
            Pass a value of `None` instead of a string to indicate that the
            column name from `left` or `right` should be left as-is, with
            no suffix. At least one of the values must not be None.
        indicator (bool or str, default False):
            If True, adds a column to the output DataFrame called "_merge"
            with information on the source of each row. The column can be
            given a different name by providing a string argument. The column
            will have a value of "left_only" for observations whose merge key
            only appears in the left DataFrame, "right_only" for observations
            whose merge key only appears in the right DataFrame, and "both"
            if the observation's merge key appears in both DataFrames.

            .. note::
                Unlike pandas, the indicator column has a string dtype rather
                than ``Categorical``.

    Returns:
        bigframes.pandas.DataFrame:
            A DataFrame of the two merged objects.
    """
    raise NotImplementedError("abstract method")
