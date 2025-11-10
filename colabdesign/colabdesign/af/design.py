import jax
import jax.numpy as jnp
import numpy as np
from cvxopt import matrix, solvers

from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.utils import copy_dict, update_dict, Key, dict_to_str, to_float, softmax, categorical, to_list, copy_missing
AA =  np.array(['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X'])

####################################################
# AF_DESIGN - design functions
####################################################
#\
# \_af_design
# |\
# | \_restart
#  \
#   \_design
#    \_step
#     \_run
#      \_recycle
#       \_single
#
####################################################

def _ensure_2d(x):
    """Return (arr2d, had_batch) where arr2d has shape (L, K)."""
    if x.ndim == 3 and x.shape[0] == 1:
        return x.squeeze(0), True
    elif x.ndim == 2:
        return x, False
    else:
        raise ValueError(f"Expected (L,K) or (1,L,K), got {x.shape}")

def normalize_ablm_grad(ablm_grad, af2_grad):
    total_bind_norm = np.linalg.norm(af2_grad[0])
    total_ablm_norm = np.linalg.norm(ablm_grad)
    scale_factor = total_bind_norm / (total_ablm_norm + 1e-7)
    normalized_ablm_grad = ablm_grad * scale_factor
    return normalized_ablm_grad

def mgda(grad_list, epsilon=1e-8):
    """
    Solves multi-task gradient combination using quadratic programming with regularization.
    Implements the Multi-Task Learning as Multi-Objective Optimization approach.
    
    Args:
        grad_list (List[np.ndarray]): List of gradient vectors, one per task.
        epsilon (float): Regularization parameter to ensure numerical stability.
                        Default: 1e-8
    
    Returns:
        np.ndarray: Optimal task weights (sum to 1, all >= 0).
        np.ndarray: Combined gradient vector.
    """
    T = len(grad_list)
    grads = [g.reshape(-1, 1) for g in grad_list]
    
    # Build Gram matrix G where G[i, j] = <g_i, g_j>
    G = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            G[i, j] = float(np.dot(grads[i].T, grads[j]))

    # Apply regularization to ensure numerical stability
    # This prevents issues with singular/ill-conditioned matrices
    G_regularized = G + epsilon * np.eye(T)
    
    # Convert to cvxopt format
    P = matrix(G_regularized)
    q = matrix(np.zeros(T))
    G_cvx = matrix(-np.eye(T))       # -w <= 0 → w >= 0
    h = matrix(np.zeros(T))
    A = matrix(np.ones((1, T)))
    b = matrix(np.ones(1))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G_cvx, h, A, b)

    w_opt = np.array(sol['x']).flatten()
    g_combined = sum(w_opt[i] * grad_list[i] for i in range(T))
    return w_opt, g_combined

def pcgrad(g1, g2):
    """
    PCGrad algorithm for two gradients using numpy.
    
    Args:
        grad1 (np.ndarray): First gradient vector
        grad2 (np.ndarray): Second gradient vector
    
    Returns:
        np.ndarray: Combined gradient after conflict resolution
    """

    def cosine_similarity(g1, g2):
      dot_product = np.dot(g1, g2)
      norm1 = np.linalg.norm(g1)
      norm2 = np.linalg.norm(g2)
      return dot_product / (norm1 * norm2 + 1e-8)
    
    # Copy gradients to avoid modifying originals
    g1_modified = g1.copy()
    g2_modified = g2.copy()

    g1_f, _ = _ensure_2d(g1_modified)
    g2_f, _ = _ensure_2d(g2_modified)

    g1_f = g1_f.flatten()
    g2_f = g2_f.flatten()
    
    # Check if gradients conflict (negative dot product)
    cos_12 = cosine_similarity(g1_f, g2_f)
    cos_21 = cosine_similarity(g2_f, g1_f)  # Same as dot_product_12, but keeping for clarity

    dot_product = np.dot(g1_f, g2_f)
    
    # If g1 and g2 conflict, project g1 onto the normal plane of g2
    if cos_12 < 0:
        # Project g1 onto g2: proj_g2(g1) = (g1 · g2 / ||g2||²) * g2
        g1_proj = (dot_product / np.dot(g2_f, g2_f)) * g2
        # Remove conflicting component: g1 ← g1 - proj_g2(g1)
        g1_modified = g1_modified - g1_proj
    
    # If g2 and g1 conflict, project g2 onto the normal plane of g1
    if cos_21 < 0:
        # Project g2 onto g1: proj_g1(g2) = (g2 · g1 / ||g1||²) * g1
        g2_proj = (dot_product / np.dot(g1_f, g1_f)) * g1
        # Remove conflicting component: g2 ← g2 - proj_g1(g2)
        g2_modified = g2_modified - g2_proj
    
    # Average the modified gradients
    combined_grad = np.array([g1_modified + g2_modified])
    
    return combined_grad, g1_modified, g2_modified

class _af_design:

  def restart(self, seed=None, opt=None, weights=None,
              seq=None, mode=None, keep_history=False, reset_opt=True, **kwargs):   
    '''
    restart the optimization
    ------------
    note: model.restart() resets the [opt]ions and weights to their defaults
    use model.set_opt(..., set_defaults=True) and model.set_weights(..., set_defaults=True)
    or model.restart(reset_opt=False) to avoid this
    ------------
    seed=0 - set seed for reproducibility
    reset_opt=False - do NOT reset [opt]ions/weights to defaults
    keep_history=True - do NOT clear the trajectory/[opt]ions/weights
    '''
    # reset [opt]ions
    if reset_opt and not keep_history:
      copy_missing(self.opt, self._opt)
      self.opt = copy_dict(self._opt)
      if hasattr(self,"aux"): del self.aux
    
    if not keep_history:
      # initialize trajectory
      self._tmp = {"traj":{"seq":[],"xyz":[],"plddt":[],"pae":[],"af_grad":[],"ablm_grad":[],"total_grad":[]},
                   "log":[],
                   "best":{}}

    # update options/settings (if defined)
    self.set_opt(opt)
    self.set_weights(weights)
  
    # initialize sequence
    self.set_seed(seed)
    self.set_seq(seq=seq, mode=mode, **kwargs)

    # reset optimizer
    self._k = 0
    self.set_optimizer(learning_rate=kwargs.pop("learning_rate",None), optimizer=kwargs.pop("optimizer",None))

  def _get_model_nums(self, num_models=None, sample_models=None, models=None):
    '''decide which model params to use'''
    if num_models is None: num_models = self.opt["num_models"]
    if sample_models is None: sample_models = self.opt["sample_models"]

    ns_name = self._model_names
    ns = list(range(len(ns_name)))
    if models is not None:
      models = models if isinstance(models,list) else [models]
      ns = [ns[n if isinstance(n,int) else ns_name.index(n)] for n in models]

    m = min(num_models,len(ns))
    if sample_models and m != len(ns):
      model_nums = np.random.choice(ns,(m,),replace=False)
    else:
      model_nums = ns[:m]
    return model_nums   

  def run(self, num_recycles=None, num_models=None, sample_models=None, models=None,
          backprop=True, callback=None, model_nums=None, return_aux=False):
    '''run model to get outputs, losses and gradients'''
    
    # pre-design callbacks
    for fn in self._callbacks["design"]["pre"]: fn(self)
    
    # decide which model params to use
    if model_nums is None:
      model_nums = self._get_model_nums(num_models, sample_models, models)
    assert len(model_nums) > 0, "ERROR: no model params defined"
    
    # loop through model params
    auxs = []
    for n in model_nums:
        p = self._model_params[n]
        aux = self._recycle(p, num_recycles=num_recycles, backprop=backprop)
        auxs.append(aux)
    auxs = jax.tree_map(lambda *x: np.stack(x), *auxs)

    # Aggregate outputs (average or take first)
    def avg_or_first(x):
      if np.issubdtype(x.dtype, np.integer): return x[0]
      else: return x.mean(0)

    self.aux = jax.tree_map(avg_or_first, auxs)
    self.aux["atom_positions"] = auxs["atom_positions"][0]
    self.aux["all"] = auxs
    
    # post-design callbacks
    for fn in (self._callbacks["design"]["post"] + to_list(callback)): fn(self)
    
    # update log
    self.aux["log"] = {**self.aux["losses"]}
    self.aux["log"]["plddt"] = 1 - self.aux["log"]["plddt"]
    for k in ["loss","i_ptm","ptm"]: self.aux["log"][k] = self.aux[k]
    for k in ["hard","soft","temp"]: self.aux["log"][k] = self.opt[k]

    # compute sequence recovery
    if self.protocol in ["fixbb","partial"] or ( hasattr(self, "binder_redesign") and self.binder_redesign ):
      if self.protocol == "partial":
        aatype = self.aux["aatype"][...,self.opt["pos"]]
      else:
        aatype = self.aux["seq"]["pseudo"].argmax(-1)
      
      mask = self._wt_aatype != -1
      true = self._wt_aatype[mask]
      pred = aatype[...,mask] 
      self.aux["log"]["seqid"] = (true == pred).mean()

    self.aux["log"] = to_float(self.aux["log"])
    self.aux["log"].update({"recycles":int(self.aux["num_recycles"]),
                            "models":model_nums})
    
    if return_aux: return self.aux

  # Modify _single function:
  def _single(self, model_params, backprop=True):
    '''single pass through the model'''
    self._inputs["opt"] = self.opt
    flags  = [self._params, model_params, self._inputs, self.key()]
    if backprop:
      (loss, aux), grad = self._model["grad_fn"](*flags)
    else:
      loss, aux = self._model["fn"](*flags)
      grad = jax.tree_map(np.zeros_like, self._params)
    aux.update({"loss":loss,"grad":grad})
    return aux

  # Modify _recycle function:
  def _recycle(self, model_params, num_recycles=None, backprop=True):
    '''multiple passes through the model (aka recycle)'''
    a = self._args
    mode = a["recycle_mode"]
    if num_recycles is None:
        num_recycles = self.opt["num_recycles"]

    if mode in ["backprop","add_prev"]:
      # recycles compiled into model, only need single-pass
      aux = self._single(model_params, backprop)
    else:
      L = self._inputs["residue_index"].shape[0]
      
      # intialize previous
      if "prev" not in self._inputs or a["clear_prev"]:
        prev = {'prev_msa_first_row': np.zeros([L,256]),
                'prev_pair': np.zeros([L,L,128])}

        if a["use_initial_guess"] and "batch" in self._inputs:
          prev["prev_pos"] = self._inputs["batch"]["all_atom_positions"] 
        else:
          prev["prev_pos"] = np.zeros([L,37,3])

        if a["use_dgram"]:
          # TODO: add support for initial_guess + use_dgram
          prev["prev_dgram"] = np.zeros([L,L,64])

        if a["use_initial_atom_pos"]:
          if "batch" in self._inputs:
            self._inputs["initial_atom_pos"] = self._inputs["batch"]["all_atom_positions"] 
          else:
            self._inputs["initial_atom_pos"] = np.zeros([L,37,3]) 
      
      self._inputs["prev"] = prev
      # decide which layers to compute gradients for
      cycles = (num_recycles + 1)
      mask = [0] * cycles

      if mode == "sample":  mask[np.random.randint(0,cycles)] = 1
      if mode == "average": mask = [1/cycles] * cycles
      if mode == "last":    mask[-1] = 1
      if mode == "first":   mask[0] = 1

      # gather gradients across recycles 
      grad = []
      for m in mask:        
        if m == 0:
          aux = self._single(model_params, backprop=False)
        else:
          aux = self._single(model_params, backprop)
          grad.append(jax.tree_map(lambda x:x*m, aux["grad"]))
        self._inputs["prev"] = aux["prev"]
        if a["use_initial_atom_pos"]:
          self._inputs["initial_atom_pos"] = aux["prev"]["prev_pos"]                

      aux["grad"] = jax.tree_map(lambda *x: np.stack(x).sum(0), *grad)
    
    aux["num_recycles"] = num_recycles
    return aux

  def step(self, lr_scale=1.0, num_recycles=None, num_models=None, sample_models=None, 
           models=None, backprop=True, callback=None, save_best=False, verbose=1, design_mode=None, 
           save_filters=None, seq_entropy_threshold=None, iter=None):
    '''do one step of gradient descent'''
    
    # run
    self.run(num_recycles=num_recycles, num_models=num_models, sample_models=sample_models,
             models=models, backprop=backprop, callback=callback)
    
    effective_length = None
    ablm_grad, ll = self.ablm_model.get_ablm_grad(self.aux["seq"])

    self.aux["log"]["af_grad"] = np.array(self.aux["grad"]["seq"])
    self.aux["log"]["ablm_grad"] = np.zeros(self.aux["grad"]["seq"].shape)
    self.aux["log"]["ablm_ll"] = ll

    normalized_ablm_grad = normalize_ablm_grad(ablm_grad, self.aux["grad"]["seq"])
    scaled_ablm_grad = self.opt['_ablm_scale'] * normalized_ablm_grad

    if self.opt["grad_merge_method"]['scale']:
      # simple scaling of ablm gradient
      self.aux["grad"]["seq"] += scaled_ablm_grad
      self.aux["log"]["ablm_grad"] = np.array(scaled_ablm_grad)
    elif self.opt["grad_merge_method"]['mgda']:
      #MGDA to combine two, diff objective gradients
      w_opt, g_combined = mgda([self.aux["grad"]["seq"], scaled_ablm_grad])
      self.aux["log"]["af_grad"] = np.array(self.aux["grad"]["seq"] * w_opt[0])
      self.aux["log"]["ablm_grad"] = np.array([scaled_ablm_grad * w_opt[1]])
      self.aux["grad"]["seq"] = g_combined
    elif self.opt["grad_merge_method"]['pcgrad']:
      #PCGrad to combine two, diff objective gradients
      g_combined, bcg, igg = pcgrad(self.aux["grad"]["seq"][0], scaled_ablm_grad)
      self.aux["log"]["af_grad"] = np.array([bcg])
      self.aux["log"]["ablm_grad"] = np.array([igg])
      self.aux["grad"]["seq"] = g_combined

    self.aux["log"]["total_grad"] = np.array(self.aux["grad"]["seq"])

    self._norm_seq_grad(effective_length=effective_length)
      
    self._state, self.aux["grad"] = self._optimizer(self._state, self.aux["grad"], self._params)

    # print string of loss weights
    weight_str = ""
    for k, v in self.opt.items():
      if "weight" in k:
        weight_str += f"{k}: {v}, "

    # apply gradients
    lr = self.opt["learning_rate"] * lr_scale
    self._params = jax.tree_map(lambda x,g:x-lr*g, self._params, self.aux["grad"])    
    
    # save results
    self._save_results(save_best=save_best, verbose=verbose, design_mode=design_mode, 
                      save_filters=save_filters, seq_entropy_threshold=seq_entropy_threshold)         
    # increment
    self._k += 1

  def _print_log(self, print_str=None, aux=None):
    if aux is None: aux = self.aux
    keys = ["models","recycles","hard","soft","temp","seqid","loss",
            "seq_ent","mlm","helix","pae","i_pae","exp_res","con","i_con",
            "sc_fape","sc_rmsd","dgram_cce","fape","plddt","i_plddt","ptm",
            "beta_sheet","beta_strand", "hydrophobicity"]
    
    if "i_ptm" in aux["log"]:
      if len(self._lengths) > 1:
        keys.append("i_ptm")
      else:
        aux["log"].pop("i_ptm")

    print(dict_to_str(aux["log"], filt=self.opt["weights"],
                      print_str=print_str, keys=keys+["rmsd"], ok=["plddt","rmsd"]))
    print("Sequence:", ''.join(AA[aux['seq']['pseudo'].argmax(-1)[0]]))

  def _calculate_passing_filter(self, aux, filter, threshold):
    if filter == 'i_pae' or filter == 'pae':
      return aux['log'][filter] < threshold
    else:
      return aux['log'][filter] >= threshold
  
  def _calculate_passing(self, aux, save_filters):
    return all(self._calculate_passing_filter(aux, filter, threshold) for filter, threshold in save_filters.items())

  def _save_results(self, aux=None, save_best=False,
                    best_metric=None, metric_higher_better=False,
                    verbose=True, design_mode=None, save_filters=None, seq_entropy_threshold=None):
    if aux is None: aux = self.aux    
    self._tmp["log"].append(aux["log"])    
    if (self._k % self._args["traj_iter"]) == 0:
      # update traj
      traj = {
        "seq":   aux["seq"]["pseudo"],
        "xyz":   aux["atom_positions"][:,1,:],
        "plddt": aux["plddt"],
        "pae":   aux["pae"],
        "af_grad":   aux["log"].get("af_grad"),
        "ablm_grad": aux["log"].get("ablm_grad"),
        "total_grad": aux["log"].get("total_grad")
      } 
      for k,v in traj.items():
        if k in self._tmp["traj"]:
          if len(self._tmp["traj"][k]) == self._args["traj_max"]:
            self._tmp["traj"][k].pop(0)
          self._tmp["traj"][k].append(v)

    # save best
    if save_best:
      if best_metric is None:
        best_metric = self._args["best_metric"]
      metric = float(aux["log"][best_metric])

      cdr_softmax_pseudo = np.array(softmax(aux['seq']['pseudo']))
      cdrs = self.opt["pos"] - self._target_len
      cdr_softmax_pseudo = np.max(cdr_softmax_pseudo[:,cdrs], axis=-1)
      cdr_softmax_pseudo = np.mean(cdr_softmax_pseudo, axis=-1)
      
      if self._args["best_metric"] in ["plddt","ptm","i_ptm","seqid","composite"] or metric_higher_better:
        metric = -metric
      # two cases for saving a sequence with worse loss: 1) if the sequence passes the "one-hot-ness" threshold and passes filters, or if the sequence passes filters and the current sequence does not
      save_worse_seq_condition = (
          seq_entropy_threshold is not None and save_filters is not None and 
          cdr_softmax_pseudo > seq_entropy_threshold and 
          self._calculate_passing(aux, save_filters) and 
          (self._tmp["best"].get("mean_soft_pseudo", 0) < seq_entropy_threshold or (self._tmp["best"]["passing"] is not None and not self._tmp["best"]["passing"]))
      )

      if "metric" not in self._tmp["best"] or metric < self._tmp["best"]["metric"] or save_worse_seq_condition:
        if "metric" in self._tmp["best"] and design_mode == "hard" and not self._calculate_passing(aux, save_filters):
          print("Lowest loss sequence did not pass filters, not saving")
        else:
          self._tmp["best"]["seq"] = ''.join(AA[aux['seq']['pseudo'].argmax(-1)[0]])
          self._tmp["best"]["aux"] = copy_dict(aux)
          self._tmp["best"]["metric"] = metric
          self._tmp["best"]["log"] = copy_dict(aux["log"])
          self._tmp["best"]["mean_soft_pseudo"] = cdr_softmax_pseudo
          self._tmp["best"]["passing"] = False
          if save_filters is not None:
            self._tmp["best"]["passing"] = self._calculate_passing(aux, save_filters)
          if design_mode is not None: self._tmp["best"]["mode"] = design_mode
          print(f'============= New best sequence!===============')
      
    if verbose and ((self._k+1) % verbose) == 0:
      self._print_log(f"{self._k+1}", aux=aux)

  def predict(self, seq=None, bias=None,
              num_models=None, num_recycles=None, models=None, sample_models=False,
              dropout=False, hard=True, soft=False, temp=1,
              return_aux=False, verbose=True,  seed=None, **kwargs):
    '''predict structure for input sequence (if provided)'''

    def load_settings():    
      if "save" in self._tmp:
        [self.opt, self._args, self._params, self._inputs] = self._tmp.pop("save")

    def save_settings():
      load_settings()
      self._tmp["save"] = [copy_dict(x) for x in [self.opt, self._args, self._params, self._inputs]]

    save_settings()

    # set seed if defined
    if seed is not None: self.set_seed(seed)
    if bias is None: bias = np.zeros((self._len, self._args.get("alphabet_size",20)))
    # set [seq]uence/[opt]ions
    if seq is not None: self.set_seq(seq=seq, bias=bias)    
    self.set_opt(hard=hard, soft=soft, temp=temp, dropout=dropout, pssm_hard=True)
    self.set_args(shuffle_first=False)
    
    # run
    self.run(num_recycles=num_recycles, num_models=num_models,
             sample_models=sample_models, models=models, backprop=False, **kwargs)


    if verbose: self._print_log("predict")

    load_settings()

    # return (or save) results
    if return_aux: return self.aux

  # ---------------------------------------------------------------------------------
  # example design functions
  # ---------------------------------------------------------------------------------
  def design(self, iters=100,
             soft=0.0, e_soft=None,
             temp=1.0, e_temp=None,
             hard=0.0, e_hard=None,
             step=1.0, e_step=None,
             dropout=True, opt=None, weights=None, 
             num_recycles=None, ramp_recycles=False, 
             num_models=None, sample_models=None, models=None,
             backprop=True, callback=None, save_best=False, verbose=1, 
             design_mode=None, save_filters=None, seq_entropy_threshold=None):

    # update options/settings (if defined)
    self.set_opt(opt, dropout=dropout)
    self.set_weights(weights)

    m = {"soft":[soft,e_soft],"temp":[temp,e_temp],
         "hard":[hard,e_hard],"step":[step,e_step]}
    m = {k:[s,(s if e is None else e)] for k,(s,e) in m.items()}

    if ramp_recycles:
      if num_recycles is None:
        num_recycles = self.opt["num_recycles"]
      m["num_recycles"] = [0,num_recycles]

    for i in range(iters):
      print(f"Designing {i+1}/{iters}")
      for k,(s,e) in m.items():
        if k == "temp":
          self.set_opt({k:(e+(s-e)*(1-(i+1)/iters)**2)})
        else:
          v = (s+(e-s)*((i+1)/iters))
          if k == "step": step = v
          elif k == "num_recycles": num_recycles = round(v)
          else: self.set_opt({k:v})
      
      # decay learning rate based on temperature
      if self.opt["linear_lr_annealing"]:
        # linear annealing
        lr_scale_temp = (m["temp"][1] - m["temp"][0]) * (i+1)/iters + m["temp"][0]
      else:
        lr_scale_temp = self.opt["temp"]
      
      lr_scale = step * ((1 - self.opt["soft"]) + (self.opt["soft"] * lr_scale_temp))
      lr_scale = max(lr_scale, self.opt["min_lr_scale"])
      # ablm_scale value gets calculated here based on the values set in config.yaml
      if sum(self.opt["ablm_scale"]) >=0: 
        if soft>=1: 
          self.opt["_ablm_scale"] = self.opt["ablm_scale"][-2]
          min_ablm_scale = self.opt["_ablm_scale"]
        else:
          self.opt["_ablm_scale"] = self.opt["ablm_scale"][1]
          min_ablm_scale = self.opt["ablm_scale"][0]
        
        self.opt["_ablm_scale"] = max(self.opt["_ablm_scale"] * max(((i+1)/iters), soft), min_ablm_scale)
      
      self.step(lr_scale=lr_scale, num_recycles=num_recycles,
                num_models=num_models, sample_models=sample_models, models=models,
                backprop=backprop, callback=callback, save_best=save_best, verbose=verbose, 
                design_mode=design_mode, save_filters=save_filters, 
                seq_entropy_threshold=seq_entropy_threshold, iter=i)

  def design_logits(self, iters=100, **kwargs):
    ''' optimize logits '''
    self.design(iters, design_mode='logits', **kwargs)

  def design_soft(self, iters=100, temp=1, **kwargs):
    ''' optimize softmax(logits/temp)'''
    self.design(iters, soft=1, temp=temp, design_mode='soft', **kwargs)
  
  def design_hard(self, iters=100, **kwargs):
    ''' optimize argmax(logits) '''
    self.design(iters, soft=1, hard=1, design_mode='hard', **kwargs)

  # ---------------------------------------------------------------------------------
  # experimental
  # ---------------------------------------------------------------------------------
  def design_3stage(self, soft_iters=300, temp_iters=100, hard_iters=10,
                    ramp_recycles=True, **kwargs):
    '''three stage design (logits→soft→hard)'''

    verbose = kwargs.get("verbose",1)

    # stage 1: logits → softmax(logits/1.0)
    if soft_iters > 0:
      if verbose: print("Stage 1: running (logits → soft)")
      self.design_logits(soft_iters, e_soft=1,
        ramp_recycles=ramp_recycles, **kwargs)
      self._tmp["seq_logits"] = self.aux["seq"]["logits"]
      
    # stage 2: softmax(logits/1.0) → softmax(logits/0.01)
    if temp_iters > 0:
      if verbose: print("Stage 2: running (soft → hard)")
      self.design_soft(temp_iters, e_temp=1e-2, **kwargs)
    
    # stage 3:
    if hard_iters > 0:
      if verbose: print("Stage 3: running (hard)")
      kwargs["dropout"] = False
      kwargs["save_best"] = True
      kwargs["num_models"] = len(self._model_names)
      self.design_hard(hard_iters, temp=1e-2, **kwargs)

  def _mutate(self, seq, plddt=None, logits=None, mutation_rate=1):
    '''mutate random position'''
    seq = np.array(seq)
    N,L = seq.shape

    # fix some positions
    i_prob = np.ones(L) if plddt is None else np.maximum(1-plddt,0)
    i_prob[np.isnan(i_prob)] = 0
    if "fix_pos" in self.opt:
      if "pos" in self.opt:
        p = self.opt["pos"][self.opt["fix_pos"]]
        seq[...,p] = self._wt_aatype_sub
      else:
        p = self.opt["fix_pos"]
        seq[...,p] = self._wt_aatype[...,p]
      i_prob[p] = 0
    
    for m in range(mutation_rate):
      # sample position
      # https://www.biorxiv.org/content/10.1101/2021.08.24.457549v1
      i = np.random.choice(np.arange(L),p=i_prob/i_prob.sum())

      # sample amino acid
      logits = np.array(0 if logits is None else logits)
      if logits.ndim == 3: logits = logits[:,i]
      elif logits.ndim == 2: logits = logits[i]
      a_logits = logits - np.eye(self._args["alphabet_size"])[seq[:,i]] * 1e8
      a = categorical(softmax(a_logits))

      # return mutant
      seq[:,i] = a
    
    return seq

  def design_semigreedy(self, iters=100, tries=10, dropout=False, save_filters=None,
                        save_best=True, seq_logits=None, e_tries=None, get_best=False, **kwargs):

    '''semigreedy search'''    
    if e_tries is None: e_tries = tries

    bias = self._inputs["bias"]
    if hasattr(self,'_tmp') and get_best:
      seq = self._tmp["best"]["aux"]["seq"]["pseudo"].argmax(-1)
    else:
      # get starting sequence
      if hasattr(self,"aux"):
        seq = self.aux["seq"]["logits"].argmax(-1)
      else:
        seq = (self._params["seq"] + bias).argmax(-1)

    # bias sampling towards the defined bias
    if seq_logits is None: seq_logits = 0
    
    model_flags = {k:kwargs.pop(k,None) for k in ["num_models","sample_models","models"]}
    verbose = kwargs.pop("verbose",1)
    
    # get current plddt
    aux = self.predict(seq, return_aux=True, verbose=False, **model_flags, **kwargs)
    plddt = self.aux["plddt"]
    plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]

    # optimize!
    if verbose:
      print("Running semigreedy optimization...")
    for i in range(iters):
      print(f"Designing {i+1}/{iters} for semigreedy")
      buff = []
      model_nums = self._get_model_nums(**model_flags)
      num_tries = (tries+(e_tries-tries)*((i+1)/iters))
      for t in range(int(num_tries)):
        mut_seq = self._mutate(seq=seq, plddt=plddt,
                               logits=seq_logits + bias)
        aux = self.predict(seq=mut_seq, return_aux=True, model_nums=model_nums, verbose=False, **kwargs)
        np_seq_repr = np.eye(20)[mut_seq[0]].astype(np.float32)
        _, aux["ablm_ll"] = self.ablm_model.get_ablm_grad(np_seq_repr)
        buff.append({"aux":aux, "seq":np.array(mut_seq)})
      
      # accept best
      losses = [x["aux"]["loss"] - self.opt["ablm_scale"][-1] * x["aux"]["ablm_ll"] for x in buff]

      best = buff[np.argmin(losses)]
      self.aux, seq = best["aux"], jnp.array(best["seq"])
      _, ll = self.ablm_model.get_ablm_grad(np.eye(20)[best["seq"][0]].astype(np.float32))
      self.aux["log"]["ablm_ll"] = ll

      self.set_seq(seq=seq, bias=bias)
      self._save_results(save_best=save_best, verbose=verbose, design_mode='hard', save_filters=save_filters)

      # update plddt
      plddt = best["aux"]["plddt"]
      plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]
      self._k += 1

  def design_pssm_semigreedy(self, soft_iters=300, hard_iters=32, tries=10, e_tries=None,
                             ramp_recycles=True, ramp_models=True, get_best=False, **kwargs):

    verbose = kwargs.get("verbose",1)

    # stage 1: logits → softmax(logits)
    if soft_iters > 0:
      self.design_3stage(soft_iters, 0, 0, ramp_recycles=ramp_recycles, **kwargs)
      self._tmp["seq_logits"] = kwargs["seq_logits"] = self.aux["seq"]["logits"]

    seq_logits = self._tmp["seq_logits"] if soft_iters > 0 else None
    # stage 2: semi_greedy
    if hard_iters > 0:
      kwargs["dropout"] = False
      if ramp_models:
        num_models = len(kwargs.get("models",self._model_names))
        iters = hard_iters
        for m in range(num_models):
          if verbose and m > 0: print(f'Increasing number of models to {m+1}.')

          kwargs["num_models"] = m + 1
          kwargs["save_best"] = (m + 1) == num_models
          self.design_semigreedy(iters, seq_logits=seq_logits, tries=tries, e_tries=e_tries, get_best=get_best, **kwargs)
          if m < 2: iters = iters // 2
      else:
        self.design_semigreedy(hard_iters, seq_logits=seq_logits, tries=tries, e_tries=e_tries, get_best=get_best, **kwargs)

  # ---------------------------------------------------------------------------------
  # experimental optimizers (not extensively evaluated)
  # ---------------------------------------------------------------------------------

  def _design_mcmc(self, steps=1000, half_life=200, T_init=0.01, mutation_rate=1,
                   seq_logits=None, save_best=True, **kwargs):
    '''
    MCMC with simulated annealing
    ----------------------------------------
    steps = number for steps for the MCMC trajectory
    half_life = half-life for the temperature decay during simulated annealing
    T_init = starting temperature for simulated annealing. Temperature is decayed exponentially
    mutation_rate = number of mutations at each MCMC step
    '''

    # code borrowed from: github.com/bwicky/oligomer_hallucination

    # gather settings
    verbose = kwargs.pop("verbose",1)
    model_flags = {k:kwargs.pop(k,None) for k in ["num_models","sample_models","models"]}

    # initialize
    plddt, best_loss, current_loss = None, np.inf, np.inf 
    current_seq = (self._params["seq"] + self._inputs["bias"]).argmax(-1)
    if seq_logits is None: seq_logits = 0

    # run!
    if verbose: print("Running MCMC with simulated annealing...")
    for i in range(steps):

      # update temperature
      T = T_init * (np.exp(np.log(0.5) / half_life) ** i) 

      # mutate sequence
      if i == 0:
        mut_seq = current_seq
      else:
        mut_seq = self._mutate(seq=current_seq, plddt=plddt,
                               logits=seq_logits + self._inputs["bias"],
                               mutation_rate=mutation_rate)

      # get loss
      model_nums = self._get_model_nums(**model_flags)
      aux = self.predict(seq=mut_seq, return_aux=True, verbose=False, model_nums=model_nums, **kwargs)
      loss = aux["log"]["loss"]
  
      # decide
      delta = loss - current_loss
      if i == 0 or delta < 0 or np.random.uniform() < np.exp( -delta / T):

        # accept
        (current_seq,current_loss) = (mut_seq,loss)
        
        plddt = aux["all"]["plddt"].mean(0)
        plddt = plddt[self._target_len:] if self.protocol == "binder" else plddt[:self._len]
        
        if loss < best_loss:
          (best_loss, self._k) = (loss, i)
          self.set_seq(seq=current_seq, bias=self._inputs["bias"])
          self._save_results(save_best=save_best, verbose=verbose)