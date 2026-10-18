"""Train the assignment networks with TensorFlow and compare them with the own one.

Importing this package prepares the environment before TensorFlow loads: the own
implementation from ``../python`` becomes importable, the Keras settings file
stays in this project instead of the user profile, and oneDNN stays off so the
results do not depend on the CPU's instruction set.
"""

import os
import sys
from pathlib import Path

_FOLDER = Path(__file__).resolve().parents[1]

# The own implementation is no installed package; it lives next to this folder
sys.path.insert(0, str(_FOLDER.parent / "python"))

os.environ.setdefault("KERAS_HOME", str(_FOLDER / ".keras"))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# Hide TensorFlow's start-up info lines; warnings and errors still show
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
