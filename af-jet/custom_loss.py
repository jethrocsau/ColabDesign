import jax.numpy as jnp


def pDockQ(outputs, aux):
  #pDockQ fitted parameters per literature
  L = 0.724
  X0 = 152.611
  k = 0.052
  b = 0.018
  bL = binder_len #TO CHANGE LATER hard coding in first

  # calculate the average interface pLDDT
  plddt = aux['plddt'][-bL:]
  avg_plddt = jnp.multiply(jnp.mean(plddt),100) #scale to a 0-100 scale instead of prob
  #jax.debug.print("average plddt: {}",avg_plddt) #debug

  # calculte # of contact
  i_cmap_binder = aux['i_cmap'][-bL:,:]
  i_cmap_flatten = jnp.ravel(i_cmap_binder)
  contact_count = jnp.sum(jnp.where(i_cmap_flatten >= 0.95,i_cmap_flatten,0))
  #jax.debug.print("contact_count: {}",contact_count) #debug

  # get average plddt * log(interface contact)
  X = jnp.multiply(avg_plddt, jnp.log(contact_count+1)) #+1 to prevent Nan
  #jax.debug.print("X: {}",X) #debug

  # calculate loss
  loss = jnp.divide(L, 1 + jnp.exp(-k*(X - X0))) + b
  #jax.debug.print("loss: {}",loss)  #debug


  return {"custom_loss":loss}


# using the plddt log interface contact loss function, which removes the sigmoid function in the pdockq function
def plddt_logif(outputs, aux, binder_len):
  plddt = aux['plddt'][-bL:]
  avg_plddt = jnp.multiply(jnp.mean(plddt),100)

  # calculte # of contact
  i_cmap_binder = aux['i_cmap'][-bL:,:]
  i_cmap_flatten = jnp.ravel(i_cmap_binder)
  contact_count = jnp.sum(jnp.where(i_cmap_flatten >= 0.95,i_cmap_flatten,0))
  #jax.debug.print("contact_count: {}",contact_count) #debug

  # get average plddt * log(interface contact)
  loss = jnp.multiply(avg_plddt, jnp.log(contact_count+1))*-1 #+1 to prevent Nan
  #jax.debug.print("X: {}",X) #debug

  return {"custom_loss":loss}


def binder_l2_mse(aux, opt):
  #get hotspot array
  model_hotspots = opt['hotspot']

  #get binder and hotspot positions of the CA atoms only
  atom_positions = aux['atom_positions']
  binder_CA_positions = atom_positions[-model_1._binder_len:,1,:]
  hotspot_CA_positions = atom_positions[model_hotspots,1,:]

  # get centroid of binder
  binder_CA_centroid = jnp.average(binder_CA_positions, axis = 0)
  #jax.debug.print("binder_CA_centroid : {}",binder_CA_centroid) #debug

  #calc L2 norm
  distance_diff = hotspot_CA_positions[:, :] - binder_CA_centroid[jnp.newaxis,:]
  distance_l2 = jnp.sum(jnp.square(distance_diff),axis = 1)
  l2_mse_loss = jnp.average(distance_l2)
  #jax.debug.print("l2_mse_loss : {}",l2_mse_loss) #debug

  return {"weighted_mse_loss":l2_mse_loss}


def binder_l2_mse_weighted(aux, opt):
  #get hotspot array
  model_hotspots = opt['hotspot']

  #get binder and hotspot positions of the CA atoms only
  atom_positions = aux['atom_positions']
  binder_CA_positions = atom_positions[-model_1._binder_len:,1,:]
  hotspot_CA_positions = atom_positions[model_hotspots,1,:]

  # get centroid of binder
  binder_CA_centroid = jnp.average(binder_CA_positions, axis = 0)
  #jax.debug.print("binder_CA_centroid : {}",binder_CA_centroid) #debug

  #calc L2 norm
  distance_diff = hotspot_CA_positions[:, :] - binder_CA_centroid[jnp.newaxis,:]
  distance_l2 = jnp.sum(jnp.square(distance_diff),axis = 1)
  l2_mse_loss = jnp.average(distance_l2)
  #jax.debug.print("l2_mse_loss : {}",l2_mse_loss) #debug

  #get plddt loss
  average_binder_plddt = jnp.average(aux['plddt'][-model_1._binder_len:]) * 100
  #jax.debug.print("average_binder_plddt : {}",average_binder_plddt) #debug

  #compute weighted L2 loss
  plddt_weighted_L2 = jnp.divide(l2_mse_loss,average_binder_plddt)

  return {"weighted_mse_loss":plddt_weighted_L2}


def binder_l2_mse_plddtboosted(aux,opt):
  #get hotspot array
  model_hotspots = opt['hotspot']

  #get binder and hotspot positions of the CA atoms only
  atom_positions = aux['atom_positions']
  binder_CA_positions = atom_positions[-model_1._binder_len:,1,:]
  hotspot_CA_positions = atom_positions[model_hotspots,1,:]

  # get centroid of binder
  binder_CA_centroid = jnp.average(binder_CA_positions, axis = 0)
  #jax.debug.print("binder_CA_centroid : {}",binder_CA_centroid) #debug

  #calc L2 norm
  distance_diff = hotspot_CA_positions[:, :] - binder_CA_centroid[jnp.newaxis,:]
  distance_l2 = jnp.sum(jnp.square(distance_diff),axis = 1)
  l2_mse_loss = jnp.average(distance_l2)
  #jax.debug.print("l2_mse_loss : {}",l2_mse_loss) #debug

  #get plddt loss
  average_binder_plddt = jnp.average(aux['plddt'][-model_1._binder_len:]) * 100
  #jax.debug.print("average_binder_plddt : {}",average_binder_plddt) #debug

  #compute weighted L2 loss
  plddt_boosted_weighted_l2 = jnp.divide(l2_mse_loss,average_binder_plddt**2)

  return {"plddt_boosted_mse_loss":plddt_boosted_weighted_l2}
