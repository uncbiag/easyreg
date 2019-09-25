# An Extension Package on Mermaid
The purpose of this package is to provide an easy interface for Mermaid and other popluar registration
package.\
The current support methods include Mermaid-optimization and Mermaid-network. 
We add supports on [AntsPy](https://github.com/ANTsX/ANTsPy), [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg) and Demons(embedded in [SimpleITK](http://www.simpleitk.org/SimpleITK/resources/software.html)), though we recommend the usage of official source.


# Installation
```
conda create -n easyreg python=3.6
source activate easyreg
git clone https://github.com/uncbiag/easyreg.git
cd easyreg

# ################download demo (optional) 
gdown https://drive.google.com/open?id=1RjFV0lht4uQFc2jYmBYxmXtRrdYAzk8S
unzip demo.zip -d . 
# #######################################

git clone https://github.com/uncbiag/mermaid.git
pip install -r requirements.txt
cd mermaid
python setup.py
```
 
# Build Document
Next step, the tutorial and documents can be built from sphinx.

```
cd EASYREG_PATH
cd doc
build html
```


# Paper related
Networks for Joint Affine and Non-parametric Image Registration [[link]](https://arxiv.org/pdf/1903.08811.pdf)\
Region-specific Diffeomorphic Metric Mapping [[link]](https://arxiv.org/pdf/1906.00139.pdf)










    
