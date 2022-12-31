EvalE2EHTR
====

The EvalE2EHTR tool implements different metrics employed to evaluate End-to-End HTR approaches, Most of its functionalities have been programed in python, like computation of the NSFD metric and the building of the edit-distance-based cost-matrix with the proposed regularization factor for using with the Hungarian algorithm (HA). Regarding the time-critical HA computation, we employ the implementation provided by the scipy library15 implemented in C and with a python-wrapper, which is based on the one described in [15]. For the also time-critical Levenshtein edit-distance computation, it was used [fasterwer](https://github.com/kahne/fastwer), a library written in C++ and wrapped in python for ease of use. In this library we have also included support for UTF-8 encoding as well as others time-critical functionalities like the implementation of bag-of-words based on hashing for faster computation, and the implementation of the backtrace algorithm to obtain the aligning-path through a minimum edit-distance between reference and hypothesis strings.

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

# Install modified version of the python package **fastwer**
git clone https://github.com/PRHLT/fastwer.git
cd fastwer
python setup.py install
```

# Example
To try this evaluator on [IAMDB's](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) paragraph-level reference & predicted text samples:
```bash
EvalE2EHTR.py -w IAMDB/ > iamdb.out
```

# Contact
Enrique Vidal (evidal@prhlt.upv.es) and Alejandro H. Toselli (ahector@prhlt.upv.es)
