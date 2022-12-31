EvalE2EHTR
====

A python package for fast word/character error rate (WER/CER) calculation
* fast (cpp implementation)
* page-level WER, CER, bWER and hWER scores


# Installation
In order to install EvalE2EHTR, follow this recipe:
```bash
conda create -n Eval-E2EHTR python=3.6
conda activate Eval-E2EHTR

pip install numpy
pip install openfst-python
pip install scipy
pip install matplotlib

pip install pybind11
```

# Example
To try this evaluator on [IAMDB's](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) reference-predicted text samples:
```bash
EvalE2EHTR.py -w IAMDB/ > iamdb.out
```

# Contact
Alejandro H. Toselli (ahector@prhlt.upv.es)
