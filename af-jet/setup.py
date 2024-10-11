import numpy as np
import pandas as pd
from scipy.special import softmax
import re
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import warnings
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.shared.utils import copy_dict
from colabdesign.af.alphafold.common import residue_constants
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

warnings.simplefilter(action='ignore', category=FutureWarning)
# Enable determinism
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()


#########################
# get pdb
###########################
def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  elif os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"{pdb_code}.pdb"
  else:
    os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
    return f"AF-{pdb_code}-F1-model_v3.pdb"

###########################
# offset
###########################
def add_cyclic_offset(self, bug_fix=True):
  '''add cyclic offset to connect N and C term'''
  def cyclic_offset(L):
    i = np.arange(L)
    ij = np.stack([i,i+L],-1)
    offset = i[:,None] - i[None,:]
    c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))
    if bug_fix:
      a = c_offset < np.abs(offset)
      c_offset[a] = -c_offset[a]
    return c_offset * np.sign(offset)
  idx = self._inputs["residue_index"]
  offset = np.array(idx[:,None] - idx[None,:])

  if self.protocol == "binder":
    c_offset = cyclic_offset(self._binder_len)
    offset[self._target_len:,self._target_len:] = c_offset
  self._inputs["offset"] = offset

###########################
#define helper functions
###########################
def af_cyclic_setup(pdb_name, target_chain,binder_len,target_hotspot,use_multimer,target_flexible):

  #target info
  pdb = pdb_name
  #enter PDB code or UniProt code (to fetch AlphaFoldDB model) or leave blink to upload your own
  target_chain = target_chain

  x = {"pdb_filename":pdb,
      "chain":target_chain,
      "binder_len":binder_len,
      "hotspot":target_hotspot,
      "use_multimer":use_multimer,
      "rm_target_seq":target_flexible}

  x["pdb_filename"] = get_pdb(x["pdb_filename"])

  if "x_prev" not in dir() or x != x_prev:
    clear_mem()
    model = mk_afdesign_model(
      protocol="binder",
      use_multimer=x["use_multimer"],
      num_recycles=num_recycles,
      recycle_mode="sample"
    )
    model.prep_inputs(
      **x,
      ignore_missing=False
    )
    x_prev = copy_dict(x)
    print("target length:", model._target_len)
    print("binder length:", model._binder_len)
    binder_len = model._binder_len

  # Set cyclic offset
  if cyclic_offset:
    if bugfix:
        print("Set bug fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucionated.")
        add_cyclic_offset(model, bug_fix=True)
    else:
        print("Set not bug fixed cyclic peptide complex offset. The cyclic peptide binder will be hallucionated.")
        add_cyclic_offset(model, bug_fix=False)
  else:
    print("Don't set cyclic offset. The linear peptide binder will be hallucionated.")

  return model
