# An Extension Package on Mermaid
The purpose of this package is to provide an easy interface for Mermaid and other popluar registration
package.\
The current support methods include Mermaid-optimization and Mermaid-network. 
We add supports on [ANTsPy](https://github.com/ANTsX/ANTsPy), [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg) and Demons(embedded in [SimpleITK](http://www.simpleitk.org/SimpleITK/resources/software.html)), though we recommend the usage of official source.


# Installation
```
conda create -n easyreg python=3.6
source activate easyreg
git clone https://github.com/uncbiag/easyreg.git
cd easyreg
git clone https://github.com/uncbiag/mermaid.git
pip install -r requirements.txt

# ################download demo (optional) 
gdown https://drive.google.com/uc?id=1RI7YevByrLAKy1JTv6KG4RSAnHIC7ybb
unzip demo.zip -d . 
# #######################################

cd mermaid
python setup.py develop
```
 Addtionally, we currently test on ANTsPy 0.1.4 version. Since AntsPy is not fully functioned,
 it will be replaced with custom Ants Package in the next release. The following command installs ANTsPy 0.1.4.
 
 ```

pip install  https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
```


# Build Document
Next step, the tutorial and documents can be built from sphinx.

```
cd EASYREG_REPOSITORY_PATH
cd doc
make html
```


# Paper related
Networks for Joint Affine and Non-parametric Image Registration [[link]](https://arxiv.org/pdf/1903.08811.pdf)\
Region-specific Diffeomorphic Metric Mapping [[link]](https://arxiv.org/pdf/1906.00139.pdf)










    
