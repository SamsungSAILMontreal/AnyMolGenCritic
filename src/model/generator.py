from joblib.parallel import delayed
from networkx.readwrite.gml import Token
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import math
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from data.target_data import PAD_TOKEN, RING_START_TOKEN, RING_END_TOKENS, Data
from time import time
from model.transformers_sota import TransformerConfig, Transformer
from util import property_loss

bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

def shufflerow(tensor, axis):
    row_perm = torch.rand(tensor.shape[:axis+1], device=tensor.device).argsort(axis)  # get permutation indices
    for _ in range(tensor.ndim-axis-1): row_perm.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor.shape[axis+1:]))  # reformat this for the gather operation
    return tensor.gather(axis, row_perm)

class CombinedEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, n_properties_cat):
        super(CombinedEmbedding, self).__init__()
        self.n_properties_cat = n_properties_cat
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size, emb_size) for i in range(self.n_properties_cat)])

    def forward(self, prop):
        out_emb = 0.0
        for i in range(self.n_properties_cat):
            out_emb += self.embeddings[i](prop[:, i]).unsqueeze(1) # [B, 1] -> [B, 1, emb_size]
        return out_emb

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    
class EdgeLogitLayer(nn.Module):
    def __init__(self, vocab, emb_size, hidden_dim, bias=True, act=None):
        super(EdgeLogitLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.scale = hidden_dim ** -0.5
        self.linear0 = nn.Linear(emb_size, self.hidden_dim, bias=bias)
        self.linear1 = nn.Linear(emb_size, self.hidden_dim, bias=bias)

    def forward(self, x, sequences):
        batch_size = x.size(0)
        seq_len = x.size(1)

        out0 = self.linear0(x).view(batch_size, seq_len, self.hidden_dim)
        out1_ = self.linear1(x).view(batch_size, seq_len, self.hidden_dim)

        ring_start_mask = (sequences == self.vocab.get_id(RING_START_TOKEN))
        index_ = ring_start_mask.long().cumsum(dim=1)
        index_ = index_.masked_fill(~ring_start_mask, 0)

        out1 = torch.zeros(batch_size, len(RING_END_TOKENS) + 1, self.hidden_dim, device=out0.device, dtype=out0.dtype)
        out1.scatter_(dim=1, index=index_.unsqueeze(-1).repeat(1, 1, self.hidden_dim), src=out1_)
        out1 = out1[:, 1:]
        out1 = out1.permute(0, 2, 1)
        logits = self.scale * torch.bmm(out0, out1)

        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class BaseGenerator(nn.Module):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        input_dropout,
        dropout,
        disable_treeloc,
        disable_graphmask,
        disable_valencemask,
        disable_counting_ring,
        disable_random_prop_mask,
        enable_absloc,
        MAX_LEN,
        bias, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        rotary, # rotary embedding
        rmsnorm, # RMSNorm instead of LayerNorm
        swiglu, # SwiGLU instead of GELU
        expand_scale,
        special_init,
        gpt,
        vocab,
        n_correct,
        cond_lin,
        cat_var_index,
        cont_var_index,
        ):
        super(BaseGenerator, self).__init__()
        self.nhead = nhead
        self.bias = bias
        self.vocab = vocab
        self.n_correct = n_correct
        self.disable_counting_ring = disable_counting_ring

        if swiglu:
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

        #
        self.token_embedding_layer = TokenEmbedding(len(self.vocab.TOKENS), emb_size)
        if not self.disable_counting_ring:
            self.count_embedding_layer = TokenEmbedding(MAX_LEN, emb_size)

        #
        if enable_absloc:
            assert not rotary
            self.positional_encoding = PositionalEncoding(emb_size)

        self.input_dropout = nn.Dropout(input_dropout)

        #
        if not (enable_absloc or rotary):
            self.linear_loc_embedding_layer = nn.Embedding(MAX_LEN + 1, nhead)
        if not disable_treeloc:
            self.up_loc_embedding_layer = nn.Embedding(MAX_LEN + 1, nhead)
            self.down_loc_embedding_layer = nn.Embedding(MAX_LEN + 1, nhead)

        #
        self.gpt = gpt
        self.rotary = rotary
        self.special_init = special_init
        if self.gpt:
            transf_config = TransformerConfig(block_size=MAX_LEN,
            n_layer = num_layers,
            n_head = nhead,
            n_embd = emb_size,
            dropout = dropout,
            bias = bias, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
            rotary =  rotary, # rotary embedding
            rmsnorm = rmsnorm, # RMSNorm instead of LayerNorm
            swiglu = swiglu, # SwiGLU instead of GELU
            expand_scale = expand_scale
            )
            self.transformer = Transformer(transf_config)
        else:
            encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu", batch_first=False)
            encoder_norm = nn.LayerNorm(emb_size)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, len(self.vocab.TOKENS) - len(RING_END_TOKENS), bias=self.bias)
        self.ring_generator = EdgeLogitLayer(vocab=self.vocab, emb_size=emb_size, hidden_dim=emb_size, bias=self.bias, act=self.act)

        #
        self.disable_treeloc = disable_treeloc
        self.disable_graphmask = disable_graphmask
        self.disable_valencemask = disable_valencemask
        self.enable_absloc = enable_absloc

        # init all weights
        if self.special_init:
            self.apply(self._init_weights)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class CondGenerator(BaseGenerator):
    def __init__(
        self,
        num_layers,
        emb_size,
        nhead,
        dim_feedforward,
        input_dropout,
        dropout,
        disable_treeloc,
        disable_graphmask,
        disable_valencemask,
        disable_counting_ring,
        disable_random_prop_mask,
        enable_absloc,
        lambda_predict_prop,
        MAX_LEN,
        bias, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        rotary, # rotary embedding
        rmsnorm, # RMSNorm instead of LayerNorm
        swiglu, # SwiGLU instead of GELU
        expand_scale,
        special_init,
        gpt,
        n_properties,
        vocab,
        n_correct,
        cond_lin,
        cat_var_index,
        cont_var_index,
        ):
        super(CondGenerator, self).__init__(
            num_layers=num_layers,
            emb_size=emb_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            input_dropout=input_dropout,
            dropout=dropout,
            disable_treeloc=disable_treeloc,
            disable_graphmask=disable_graphmask,
            disable_valencemask=disable_valencemask,
            disable_counting_ring=disable_counting_ring,
            disable_random_prop_mask=disable_random_prop_mask,
            enable_absloc=enable_absloc,
            MAX_LEN=MAX_LEN,
            bias=bias,
            rotary=rotary,
            rmsnorm=rmsnorm,
            swiglu=swiglu,
            expand_scale=expand_scale,
            special_init=special_init,
            gpt=gpt,
            vocab=vocab,
            n_correct=n_correct,
            cond_lin=cond_lin,
            cat_var_index=cat_var_index,
            cont_var_index=cont_var_index,
        )
        self.lambda_predict_prop = lambda_predict_prop
        self.disable_random_prop_mask = disable_random_prop_mask
        self.n_properties = n_properties
        self.cond_lin = cond_lin
        self.cat_var_index = cat_var_index
        self.cont_var_index = cont_var_index
        self.n_properties_cat = len(self.cat_var_index)
        self.n_properties_cont = self.n_properties - self.n_properties_cat
        # continuous
        if self.cond_lin:
            self.cond_embedding_layer_cont = nn.Linear(self.n_properties_cont*2, emb_size)
        else:
            self.cond_embedding_layer_cont = nn.Sequential(
                nn.Linear(self.n_properties_cont*2, emb_size // 2),
                self.act,
                nn.Linear(emb_size // 2, emb_size))
        num_classes = 2 # For now assuming that binary 0/1 (2 is missing value)
        self.cond_embedding_layer_cat = CombinedEmbedding(num_classes+1, emb_size, self.n_properties_cat)

        if self.lambda_predict_prop > 0.0:
            self.predict_prop_layer = nn.Linear(emb_size, self.n_properties, bias=self.bias)

        # init all weights
        if self.special_init:
            self.apply(self._init_weights)
            # apply special scaled init to the residual projections, per GPT-2 paper
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

    def apply_cond_layer(self, prop, mask_cond):
        batch_size = prop.size(0)

        if self.training and not self.disable_random_prop_mask: # random masking of molecule properties
            batch_choices = torch.arange(self.n_properties, device=prop.device).unsqueeze(0).repeat(batch_size,1) < torch.randint(0, self.n_properties+1, (batch_size,1), device=prop.device) # random choose how many properties to keep (equal-prob of each amount of properties)
            batch_choices = shufflerow(batch_choices, 1).to(dtype=prop.dtype) # shuffle [b, n_properties] to randomize which properties are selected
        else: # use all molecule properties
            batch_choices = torch.ones(batch_size, self.n_properties, dtype=prop.dtype, device=prop.device)
        if mask_cond is not None:
            batch_choices[:, mask_cond] = 0

        emb = 0
        if self.n_properties_cont > 0:
            batched_cond_data_cont = torch.cat((prop[:, self.cont_var_index]*batch_choices[:, self.cont_var_index], 1-batch_choices[:, self.cont_var_index]), dim=1) # we concatenate cond-data (after zeroing out the missing values) and the missing-mask
            emb += self.cond_embedding_layer_cont(batched_cond_data_cont).view(batch_size, 1, -1)
        if self.n_properties_cat > 0:
            num_classes = 2 # For now assuming that binary 0/1 (2 is missing value)
            batched_cond_data_cat = (prop[:, self.cat_var_index]*batch_choices[:, self.cat_var_index] + num_classes*(1-batch_choices[:, self.cat_var_index])).to(dtype=torch.int)
            emb += self.cond_embedding_layer_cat(batched_cond_data_cat).view(batch_size, 1, -1)

        return emb

    def forward(self, batched_mol_data, batched_cond_data, mask_cond=None):
        (
            sequences,
            count_sequences,
            graph_mask_sequences,
            valence_mask_sequences,
            linear_loc_squares,
            up_loc_squares,
            down_loc_squares,
        ) = batched_mol_data

        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)

        #
        out = self.token_embedding_layer(sequences)
        if not self.disable_counting_ring:
            out += self.count_embedding_layer(count_sequences)

        # we inject properties as a embedding
        emb_cond = None
        out += self.apply_cond_layer(batched_cond_data, mask_cond)
        out = self.input_dropout(out)

        #
        mask = torch.zeros(batch_size, sequence_len, sequence_len, self.nhead, device=out.device)
        if not (self.enable_absloc or self.rotary):
            mask += self.linear_loc_embedding_layer(linear_loc_squares)

        if not self.disable_treeloc:
            mask += self.up_loc_embedding_layer(up_loc_squares)
            mask += self.down_loc_embedding_layer(down_loc_squares)

        mask = mask.permute(0, 3, 1, 2) # b, n_head, L, L

        #
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len), dtype=sequences.dtype, device=sequences.device)) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))

        #
        if self.gpt:
            out = self.transformer(out, mask, emb_cond)
        else:
            mask = mask.reshape(-1, sequence_len, sequence_len)
            key_padding_mask = sequences == self.vocab.get_id(PAD_TOKEN)
            out = out.transpose(0, 1)
            out = self.transformer(out, mask, key_padding_mask)
            out = out.transpose(0, 1)
        #
        logits0 = self.generator(out)
        logits1 = self.ring_generator(out, sequences)
        logits = torch.cat([logits0, logits1], dim=2)
        if not self.disable_graphmask:
            logits = logits.masked_fill(graph_mask_sequences, float("-inf"))
        if not self.disable_valencemask:
            logits = logits.masked_fill(valence_mask_sequences, float("-inf"))

        if self.lambda_predict_prop > 0.0: # output must be converted to properties
            predicted_prop = self.predict_prop_layer(out)
        else:
            predicted_prop = None

        return logits, predicted_prop

    @torch.no_grad()
    def decode_(self, batched_cond_data, max_len, device, temperature=1.0, guidance=1.0, guidance_rand=False, top_k=0, 
        mask_cond=None, track_property_closeness=False, allow_empty_bond=True):
        num_samples = batched_cond_data.size(0)
        data_list = [Data(MAX_LEN=max_len, vocab=self.vocab, n_correct=self.n_correct, allow_empty_bond=allow_empty_bond, tracker_id=i) for i in range(num_samples)]
        data_list_ended = []
        prop_pred_ended = []
        data_idxs = list(range(num_samples))
        data_idxs_ended = []

        if guidance != 1.0:
            use_guidance = True
        else:
            use_guidance = False
        if guidance_rand:
            guidance = torch.rand(num_samples, 1, dtype=batched_cond_data.dtype, device=device)*1.5 + 0.5 # [0,1] to [0.5, 2.0]
            use_guidance = True

        def _update_data(inp):
            data, id = inp
            data.update(id)
            return data

        for idx in range(max_len):
            if len(data_list) == 0:
                break
            feature_list = [data.featurize() for data in data_list]
            batched_mol_data = Data.collate(feature_list)
            batched_mol_data = [tsr.to(device) for tsr in batched_mol_data]

            #for i in range(len(data_idxs)):
            #    assert data_list[i].tracker_id == data_idxs[i]
            if use_guidance:
                logits_cond, _ = self(batched_mol_data, batched_cond_data[data_idxs], mask_cond=mask_cond)
                logits_uncond, predicted_prop = self(batched_mol_data, batched_cond_data[data_idxs], mask_cond=[True for i in range(self.n_properties)])
                logits_cond = logits_cond[:, -1] / temperature
                logits_uncond = logits_uncond[:, -1] / temperature
                logits = logits_cond
                not_inf = torch.logical_not(torch.isinf(logits_uncond))
                if guidance_rand:
                    guidance_ = guidance[data_idxs]
                else:
                    guidance_ = guidance
                logits[not_inf] = ((1-guidance_)*logits_uncond)[not_inf] + (guidance_*logits_cond)[not_inf]
            else: # no guidance
                logits, _ = self(batched_mol_data, batched_cond_data[data_idxs], mask_cond=mask_cond)
                if track_property_closeness:
                    _, predicted_prop = self(batched_mol_data, batched_cond_data[data_idxs], mask_cond=[True for i in range(self.n_properties)])
                logits = logits[:, -1] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            preds = Categorical(logits=logits).sample()

            data_list = [_update_data(pair) for pair in zip(data_list, preds.tolist())]
            if track_property_closeness:
                prop_pred_ended += [predicted_prop[i, -1, :] for i, data in enumerate(data_list) if data.ended]
            data_list_ended += [data for data in data_list if data.ended]
            data_idxs_ended += [data_idx for data_idx, data in zip(data_idxs, data_list) if data.ended]
            data_idxs = [data_idx for data_idx, data in zip(data_idxs, data_list) if not data.ended]
            data_list = [data for data in data_list if not data.ended]
            if idx == max_len-1:
                for data in data_list:
                    data.error = "incomplete"
                    print(data.error)
            #for i in range(len(data_idxs_ended)):
            #    assert data_list_ended[i].tracker_id == data_idxs_ended[i]

        # Combined finished and unfinished molecules
        data_idxs_all = data_idxs_ended + data_idxs
        data_list_all = data_list_ended + data_list
        ended_all = [True for i in range(len(data_list_ended))] + [False for i in range(len(data_list))]
        if track_property_closeness:
            prop_pred_all = prop_pred_ended + [666*torch.ones_like(prop_pred_ended[0]) for i in range(len(data_list))]

        # Reorder properly the generated data
        ordering = np.argsort(data_idxs_all).tolist()
        data_list_all = [data_list_all[i] for i in ordering]
        ended_all = [ended_all[i] for i in ordering]
        if track_property_closeness:
            prop_pred_all = [prop_pred_all[i] for i in ordering]

        if track_property_closeness:
            prop_pred_all = torch.stack(prop_pred_all, dim=0)
            loss_prop = 0.0
            if self.n_properties_cont > 0:
                loss_prop += 0.5*((prop_pred_all[:, self.cont_var_index] - batched_cond_data[:, self.cont_var_index])**2).sum(1)

            if self.n_properties_cat > 0:
                for i in self.cat_var_index:
                    loss_prop += bce_loss(prop_pred_all[:, i], batched_cond_data[:, i])
        else:
            loss_prop = None

        for i in range(num_samples):
            assert data_list_all[i].tracker_id == i

        return data_list_all, ended_all, loss_prop

    @torch.no_grad()
    def decode(self, batched_cond_data, max_len, device, temperature=1.0, guidance=1.0, guidance_rand=False, top_k=0, 
        mask_cond=None, best_out_of_k=1, predict_prop=False, allow_empty_bond=True, return_loss_prop=False):
        num_samples = batched_cond_data.size(0)
        
        loss_prop = None
        data_list, ended, loss_prop = self.decode_(batched_cond_data=batched_cond_data, 
            max_len=max_len, device=device, temperature=temperature, guidance=guidance, guidance_rand=guidance_rand, top_k=top_k, 
            mask_cond=mask_cond, track_property_closeness=(best_out_of_k > 1 and predict_prop) or return_loss_prop,
            allow_empty_bond=allow_empty_bond)

        # Do a Best-out-of-K based on the predicted properties
        for i in range(1, best_out_of_k):
            if not predict_prop and np.array(ended).int().sum() == len(ended):
                break # if all completed and we have no property-prediction, we end early
            data_list_, ended_, loss_prop_ = self.decode_(batched_cond_data=batched_cond_data, 
                max_len=max_len, device=device, temperature=temperature, guidance=guidance, guidance_rand=guidance_rand, top_k=top_k, 
                mask_cond=mask_cond, track_property_closeness=best_out_of_k > 1 and predict_prop,
                allow_empty_bond=allow_empty_bond)
            # We do a pairwise comparision between old and new sample and chose the best
            if predict_prop:
                for i in range(num_samples):
                    if loss_prop[i] > loss_prop_[i]:
                        data_list[i] = data_list_[i]
                        loss_prop[i] = loss_prop_[i]
            else: # we take the next completed one
                for i in range(num_samples):
                    if not ended[i] and ended_[i]:
                        data_list[i] = data_list_[i]

        return data_list, loss_prop

