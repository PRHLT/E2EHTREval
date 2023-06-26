EvalE2EHTR
====

The EvalE2EHTR tool implements different metrics employed to evaluate End-to-End HTR approaches, Most of its functionalities have been programed in python, like computation of the NSFD metric and the building of the edit-distance-based cost-matrix with the proposed regularization factor for using with the Hungarian algorithm (HA). Regarding the time-critical HA computation, we employ the implementation provided by the scipy library15 implemented in C and with a python-wrapper, which is based on the one described in [15]. For the also time-critical Levenshtein edit-distance computation, it was used [fasterwer](https://github.com/kahne/fastwer), a library written in C++ and wrapped in python for ease of use. In this library we have also included support for UTF-8 encoding as well as others time-critical functionalities like the implementation of bag-of-words based on hashing for faster computation, and the implementation of the backtrace algorithm to obtain the aligning-path through a minimum edit-distance between reference and hypothesis strings.

A python package for fast word/character error rate (WER/CER) calculation
* fast (cpp implementation)
* page-level WER, CER, bWER and hWER scores


# Installation
Installation through [conda](https://anaconda.org) is high recommended. In order to install EvalE2EHTR, follow this recipe:
```bash
conda create -n Eval-E2EHTR python=3.6
conda activate Eval-E2EHTR

pip install numpy
pip install scipy
pip install pybind11

# Install required version of the modified python package "fastwer"
git clone https://github.com/PRHLT/fastwer.git
cd fastwer
python setup.py install
```

If you find this evaluation tool useful, please cite:
```
@article{VIDAL2023109695,
  title = {End-to-End page-Level assessment of handwritten text recognition},
  journal = {Pattern Recognition},
  volume = {142},
  pages = {109695},
  year = {2023},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2023.109695},
  author = {Enrique Vidal and Alejandro H. Toselli and Antonio Ríos-Vila and Jorge Calvo-Zaragoza}
}
```
Check this paper for more details [Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S003132032300393X).


# Example of use
To try this evaluator on [IAMDB's](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) paragraph-level reference & predicted text samples:
```bash
EvalE2EHTR.py IAMDB/ > iamdb.out
```
If the difference between G-WER\_lev and G-WER\_bow (WER-bWER) is not 0 or very low then there are reading-order mistmatches between reference and hypothesis transcripts.

**-H** option shows also the computation of Hungarian's WER/CER and the Normalized Spearman foot-rule distance (NSFD).
```bash
EvalE2EHTR.py -H IAMDB/ > iamdb.out
```

The meaning with the paper cited above are as follows:

G-WER_lev  (WER)
G-CER_lev  (CER) 

G-WER_bow (bWER)

G-WER_hun (hWER)
G-WER_hlv WER between HA-aligned texts
G-CER_hlv CER between HA-aligned texts (hCER)

G-SPR_hun (NSFD


# Contact
Enrique Vidal (evidal@prhlt.upv.es) and Alejandro H. Toselli (ahector@prhlt.upv.es)
