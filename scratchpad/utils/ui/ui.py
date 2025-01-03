from rich.console import Console
from rich.table import Table
import numpy as np

console = Console()


def make_table(title, data):
    keys = data[0].keys()
    table = Table(title=title)
    colors = ["cyan", "magenta", "green", "yellow", "blue", "red", "black"]

    for idx, column in enumerate(keys):
        table.add_column(column, justify="Right", style=colors[idx % len(colors)])

    for row in data:
        for key in keys:
            if type(row[key]) == np.float64 or type(row[key]) == float:
                row[key] = str(round(row[key], 2))
        table.add_row(*[row[key] for key in keys])
    return table
