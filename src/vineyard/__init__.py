import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d|%(levelname)s|%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
