
import os, sys

# Add the path where the colabdesign module is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from colabdesign import mk_afdesign_model, clear_mem
from IPython.display import HTML
from google.colab import files
import numpy as np