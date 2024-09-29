import os
import sys

from loguru import logger as logger


# config = {
#     "handlers": [
#         {
#             "sink": sys.stdout,
#             "enqueue": True,
#             "format": "<level>[Rank {extra[rank]}/{extra[world]}] [{time:YYYY-MM-DD HH:mm:ss}] [{level}]:</> {message}",
#             # "filter": lambda record: record["extra"].get("rank") == "0",
#         },
#     ],
#     "extra": {
#         "rank": os.environ.get("LOCAL_RANK", 0),
#         "world": os.environ.get("WORLD_SIZE", 1),
#     },
# }
def custom_format(record):
    path = os.path.relpath(record["file"].path)
    record["extra"]["abspath"] = path
    return "{name} - {level} - {extra[abspath]} - {message}\n{exception}"


# logger.add(sys.stderr, format=custom_format)
