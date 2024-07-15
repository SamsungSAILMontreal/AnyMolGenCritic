import argparse
from data.dataset import get_spanning_tree_from_smiles

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--smile", type=str, default='Clc1ccccc1C2(NC)CCCCC2=O')
    parser.add_argument("--MAX_LEN", type=int, default=250)
    parser.add_argument("--randomize_order", action="store_true")
    hparams = parser.parse_args()

    out = get_spanning_tree_from_smiles([hparams.smile], randomize_order=hparams.randomize_order, MAX_LEN=hparams.MAX_LEN)
    print(out)