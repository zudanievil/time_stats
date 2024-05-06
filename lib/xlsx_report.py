"""
some neat functions that help to from reports with pandas and openpyxl
functions that use global state have underscore at the end of the name
like `WBook_` creates global `openpyxl.Workbook` without sheets
"""

import openpyxl as openpyxl
import pandas as pd

_WS = openpyxl.worksheet.worksheet.Worksheet
_WB = openpyxl.Workbook

workbook = None

def Book_() -> _WB:
    global workbook
    workbook = _WB()
    del workbook['Sheet']
    return workbook

def Sheet_(name: str, index: int = None) -> _WS:
    ws = workbook.create_sheet(name, index)
    workbook.active_sheet = ws
    return ws

def write_tbl(ws: _WS, df: pd.DataFrame, write_header = True):
    """
    write table, ignore index. no bullshit formatting.
    NB: coerces np.nan & np.inf to empty cells
    """
    if write_header:
        ws.append(df.columns.to_list())
    for _, r in df.iterrows():
        ws.append(r.to_list())
        # if you think, this is unoptimal, guess what, pandas does it cell-by-cell
        # pandas.io.excel._openpyexcel.OpenpyxlWriter._write_cells

def write_tbl_(df: pd.DataFrame) -> None:
    write_tbl(workbook.active_sheet, df)


def freeze_panes(ws: _WS, row: int = 0, col: int = 0):
    """
    freeze_panes(ws, row=1) -- freeze 1st row as header
    freeze_panes(ws, col=1) -- freeze 1st column
    freeze_paens(ws, row=2) -- freese 1st 2 rows as header
    you get the idea
    """
    ws.freeze_panes = ws.cell(row+1, col+1)

def freeze_panes_(row: int = 0, col: int = 0) -> None:
    freeze_panes(workbook.active_sheet, row, col)

def vcat(*dfs: pd.DataFrame) -> pd.DataFrame:
    "return pd.concat(dfs, axis=0, join='outer', ignore_index=True)"
    return pd.concat(dfs, axis=0, join="outer", ignore_index=True)

def hcat(*dfs: pd.DataFrame ) -> pd.DataFrame:
    "return pd.concat(dfs, axis=1, join='outer')"
    return pd.concat(dfs, axis=1, join="outer")

def splitby(
    df: pd.DataFrame, 
    by_column, 
    select_columns=slice(None, None),
) -> dict[str, pd.DataFrame]:
    "split by values in a column"
    gb = df.groupby(by_column)
    return {g: d[select_columns] for g, d in gb}

def idxcat(idx: pd.MultiIndex, sep="::") -> pd.Index:
    "join MultiIndex into string index"
    xs = [sep.join(str(x) for x in tup) for tup in idx.to_list()]
    return pd.Index(xs)

def save_(path) -> _WB:
    global workbook
    workbook.save(path)
    temp = workbook
    workbook = None
    return temp

