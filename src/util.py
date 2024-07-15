from rdkit import Chem, RDLogger
import torch
import numpy as np
import torch.nn.functional as F
from data.target_data import EOS_TOKEN_id
bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

def compute_sequence_accuracy(logits, batched_sequence_data, ignore_index=0):
    batch_size = batched_sequence_data.size(0)
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]
    preds = torch.argmax(logits, dim=-1)

    correct = (preds == targets)
    correct[targets == ignore_index] = True
    elem_acc = correct.float().mean()
    sequence_acc = correct.view(batch_size, -1).all(dim=1).float().mean()

    return elem_acc, sequence_acc

def compute_property_accuracy(batched_sequence_data, prop, pred_prop, cont_var_index, cat_var_index, mean=True):
    batch_size = prop.size(0)
    eos = batched_sequence_data == EOS_TOKEN_id # b x l

    acc_prop = []
    if len(cont_var_index) > 0:
        if mean:
            acc_prop += [0.5*((pred_prop[eos][:, cont_var_index] - prop[:, cont_var_index])**2).mean(dim=0)] # [p_cont]
        else:
            acc_prop += [0.5*((pred_prop[eos][:, cont_var_index] - prop[:, cont_var_index])**2)] # [b, p_cont]
    if len(cat_var_index) > 0:
        preds = (pred_prop[eos][:, cat_var_index] > 0.0).int()
        targets = prop[:, cat_var_index]
        correct = (preds == targets)
        if mean:
            elem_acc = correct.float().mean().view(1) # 1
        else:
            elem_acc = correct.float() # [b, 1]
        acc_prop += [elem_acc]
    if mean:
        acc_prop = torch.cat(acc_prop, dim=0)
    else:
        acc_prop = torch.cat(acc_prop, dim=1)
    ordering = np.argsort(cont_var_index + cat_var_index).tolist()
    if mean:
        acc_prop = acc_prop[ordering]
    else:
        acc_prop = acc_prop[:, ordering]
    return acc_prop

# Make generated molecules have properties close to their conditioned-properties
def property_loss(batched_sequence_data, prop, pred_prop, cont_var_index, cat_var_index, lambda_predict_prop=1.0):
    L = pred_prop.shape[1]

    if len(cont_var_index) > 0:
        loss_prop_cont = lambda_predict_prop*0.5*(((pred_prop[:, :, cont_var_index] - prop[:, cont_var_index].unsqueeze(1))**2).sum(2).mean(dim=1)).mean(dim=0)
    else:
        loss_prop_cont = 0.0
    loss_prop_cat = 0.0
    if len(cat_var_index) > 0:
        for i in cat_var_index:
            loss_prop_cat += lambda_predict_prop*bce_loss(pred_prop[:, :, i], prop[:, i].unsqueeze(1).expand(-1, L))
    return loss_prop_cont, loss_prop_cat

def compute_sequence_cross_entropy(logits, batched_sequence_data, ignore_index=0, prop=None, pred_prop=None, 
    lambda_predict_prop=0.0, cont_var_index=None, cat_var_index=None):
    logits = logits[:, :-1]
    targets = batched_sequence_data[:, 1:]

    if lambda_predict_prop > 0:
        assert pred_prop is not None
        assert cat_var_index is not None
        loss_prop_cont, loss_prop_cat = property_loss(batched_sequence_data=batched_sequence_data, prop=prop, pred_prop=pred_prop, 
            cont_var_index=cont_var_index, cat_var_index=cat_var_index,
            lambda_predict_prop=lambda_predict_prop)
    else:
        loss_prop_cont, loss_prop_cat = 0.0, 0.0
    loss = loss_prop_cont + loss_prop_cat + F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)
    return loss, loss_prop_cont, loss_prop_cat


def compute_entropy(logits, batched_sequence_data, ignore_index=0):
    logits = logits[:, :-1].reshape(-1, logits.size(-1))
    targets = batched_sequence_data[:, 1:].reshape(-1)

    logits = logits[targets != ignore_index]
    probs = torch.softmax(logits, dim=-1)
    probs = probs[~torch.isinf(logits)]
    loss = -(probs * torch.log(probs)).sum() / logits.size(0)
    return loss


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol)
    except:
        return None   


    if len(smiles) == 0:
        return None

    return smiles
