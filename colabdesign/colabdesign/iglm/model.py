from iglm import IgLM

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CustomIgLM(nn.Module, IgLM):
    """
    A custom class that wraps a frozen IgLM model but uses an external seq_logits tensor
    for the variable (designed) region.
    """

    def __init__(
        self,
        model_name: str = "IgLM",
        chain_token: str = "[HEAVY]",
        iglm_species: str = "[HUMAN]",
        ablm_temp: float = 1.0,
        is_scfv: bool = False,
        vh_first: bool = True,
        vh_len: Optional[int] = None,
        vl_len: Optional[int] = None,
        device: torch.device = None, 
        seed: int = 0,
    ):
        """
        Args:
            model_name: Name of the pretrained IgLM model (e.g. "IgLM" or "IgLM-S").
            seq_length: The length of the variable region to design.
            chain_token: The token representing the antibody chain (e.g., "VHH").
            species_token: The token representing the species (e.g., "human").
            tau: Temperature for the softmax applied to seq_logits.
            device: Torch device to use; if None, the IgLM default (GPU if available, else CPU).
        """
        super().__init__()
        IgLM.__init__(self, model_name=model_name)

        if device is not None:
            self.device = device
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.chain_token = chain_token
        self.species_token = iglm_species
        self.is_scfv = is_scfv
        self.vh_first = vh_first
        self.vh_len = vh_len
        self.vl_len = vl_len
        
        self.tau = ablm_temp

        self.amino_acids = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
        
        aa_ids = []
        for aa in self.amino_acids:
            tid = self.tokenizer.convert_tokens_to_ids(aa)
            if tid == self.tokenizer.unk_token_id:
                raise ValueError(f"Unrecognized amino acid token: {aa}")
            aa_ids.append(tid)
        self.amino_acid_ids = torch.tensor(aa_ids, device=self.device)

        self.chain_id = self.tokenizer.convert_tokens_to_ids(chain_token)
        self.species_id = self.tokenizer.convert_tokens_to_ids(self.species_token)
        self.suffix_id = self.tokenizer.sep_token_id 
        
        # set seed
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)

    def forward(self, seq_logits: torch.Tensor, chain) -> tuple:
        """
        Args:
            seq_logits (torch.Tensor): Input sequence (seq_length, 20).

        Returns:
            ce_loss (nll)
            log_likelihood
            cdr_lls
        """
        soft_probs = F.softmax(seq_logits / self.tau, dim=-1)
        hard_indices = soft_probs.argmax(dim=-1)
        hard_one_hot = F.one_hot(hard_indices, num_classes=soft_probs.size(-1)).float()
        
        # Straight-through estimator
        ste_probs = hard_one_hot + (soft_probs - soft_probs.detach())

        final_probs = ste_probs        
        embed_layer = self.model.get_input_embeddings()
        amino_embeds = embed_layer(self.amino_acid_ids)  # (20, embed_dim)
        var_embeds = final_probs @ amino_embeds             # (L, embed_dim)

        # Construct the full sequence embedding with prefix and suffix
        chain_id = self.tokenizer.convert_tokens_to_ids(chain)
        prefix_ids = torch.tensor([chain_id, self.species_id], device=self.device)
        prefix_embeds = embed_layer(prefix_ids)            # (2, embed_dim)
        
        suffix_ids = torch.tensor([self.suffix_id], device=self.device)
        suffix_embeds = embed_layer(suffix_ids)            # (1, embed_dim)
        
        full_embeds = torch.cat([prefix_embeds, var_embeds, suffix_embeds], dim=0).unsqueeze(0)
        outputs = self.model(inputs_embeds=full_embeds)
        logits = outputs.logits  # shape: (1, total_length, vocab_size)
        
        var_token_ids = self.amino_acid_ids[hard_indices]
        full_target_ids = torch.cat([prefix_ids, var_token_ids, suffix_ids], dim=0).unsqueeze(0)
        
        shift_logits = logits[:, :-1, :]  
        shift_labels = full_target_ids[:, 1:]
        ce_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'  # Return loss for each position
        )
        
        # Reshape the losses back to match the sequence shape (B, S)
        position_losses = ce_loss.reshape(shift_labels.shape)
        position_losses = position_losses[:, 1:-1]
        
        # Calculate the average loss for the entire sequence
        total_loss = position_losses.mean()
        log_likelihood = -total_loss

        return total_loss, log_likelihood.item()
          

    def _get_iglm_grad(self, seq_logits: torch.Tensor, chain='[HEAVY]') -> torch.Tensor:
        ce_loss, ll = self.forward(seq_logits, chain=chain)
        # compute full gradient w.r.t. seq_logits
        full_grad = torch.autograd.grad(ce_loss, seq_logits)[0]
        return full_grad.detach(), ll

    def get_ablm_grad(self, seq) -> np.ndarray:
        # If seq is provided as a dict with logits, extract the tensor.
        if isinstance(seq, dict):
            current_logits = torch.tensor(seq["logits"][0], device=self.device, requires_grad=True)
        else:
            current_logits = torch.tensor(seq, device=self.device, requires_grad=True)
        
        if self.is_scfv:
          if self.vh_first:
            current_logits_h = current_logits[:self.vh_len, :]
            current_logits_l = current_logits[-self.vl_len:, :]
          else:
            current_logits_l = current_logits[:self.vl_len, :]
            current_logits_h = current_logits[-self.vh_len:, :]
            
          iglm_grad_h, ll_h = self._get_iglm_grad(current_logits_h, chain='[HEAVY]')
          iglm_grad_l, ll_l = self._get_iglm_grad(current_logits_l, chain='[LIGHT]')
          
          logits_shape = current_logits.shape[0] - current_logits_h.shape[0] - current_logits_l.shape[0]
          iglm_grad = torch.cat([iglm_grad_h, torch.zeros((logits_shape,20), device='cuda'), iglm_grad_l], dim=0)

          ll = np.sum([ll_l,ll_h])

        else:
            iglm_grad, ll = self._get_iglm_grad(current_logits)

        return iglm_grad.cpu().numpy(), ll
    