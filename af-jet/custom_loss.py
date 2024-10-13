import jax.numpy as jnp

def custom_loss_callback(outputs, params):

  #pDockQ fitted parameters per literature
  L = 0.724
  X0 = 152.611
  k = 0.052
  b = 0.018

  # calculate the average interface pLDDT
  avg_plddt = 0.7 #FILL

  # calculte # of contact
  contact_count = #FILL

  # get average plddt * log(interface contact)
  X = jnp.multiply(avg_plddt, jnp.log(contact_count))

  # calculate loss
  loss = jnp.divide(L, 1 + jnp.exp(-k*(X - X0))) + b

  return {"custom_loss":loss}


#test run
