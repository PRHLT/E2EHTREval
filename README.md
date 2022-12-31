EvalE2EHTR
====

A python package for fast word/character error rate (WER/CER) calculation
* fast (cpp implementation)
* page-level WER, CER, bWER and hWER scores


# Installation
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
```bash
# To evaluate the IAMDB reference-predicted text samples:
EvalE2EHTR.py -w IAMDB/ > iamdb.out
```

# Contact
Alejandro H. Toselli (ahector@prhlt.upv.es)
