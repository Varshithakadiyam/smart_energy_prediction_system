"""
compat.py
Compatibility bridge for macOS ARM, Python 3.13, Scikit-learn, and TensorFlow.
Prevents PyArrow mutex lock collision and routes KERAS_HOME/MPLCONFIGDIR to writable locations.
"""

import sys
import os

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['KERAS_HOME'] = os.path.abspath('.keras')

class _DummyPA:
    __version__ = '1.0.0'
    Table = type('T', (), {})
    RecordBatch = type('R', (), {})
    Array = type('A', (), {})
    ChunkedArray = type('C', (), {})

sys.modules['pyarrow'] = _DummyPA
sys.modules['pyarrow._generated_version'] = _DummyPA
