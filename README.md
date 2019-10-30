# An Extension Package for Mermaid
The purpose of this package is to provide a simple interface to Mermaid and other popluar registration
packages.\

The currently supported methods include Mermaid-optimization (i.e., optimization-based registration) and Mermaid-network (i.e., deep network-based registration methods using the mermaid deformation models).
We also added support for [ANTsPy](https://github.com/ANTsX/ANTsPy), [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg) and Demons(embedded in [SimpleITK](http://www.simpleitk.org/SimpleITK/resources/software.html)), though we recommend using the official source.


# Installation
```
conda create -n easyreg python=3.6
source activate easyreg
git clone https://github.com/uncbiag/easyreg.git
cd easyreg
git clone https://github.com/uncbiag/mermaid.git
pip install -r requirements.txt

# ################download demo (optional)######
gdown https://drive.google.com/uc?id=1RI7YevByrLAKy1JTv6KG4RSAnHIC7ybb
unzip demo.zip -d . 
# #############################################

cd mermaid
python setup.py develop
```
For third-party toolkits:

**ANTsPy**

We currently test using ANTsPy version 0.1.4. Since AntsPy is not fully functioned,
 it will be replaced with a custom Ants Package in the next release. The following command installs ANTsPy 0.1.4.
 
 ```

pip install  https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl
```

**NiftyReg**

For NiftyReg installation instructions please refer to [NiftyReg Installation](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install)

# Build Documentation
The lastest doc can be found [here](www.cs.unc.edu/~zyshen/package/easyreg/html/index.html)

Source code documentation and tutorials can be built on local using sphinx.

```
cd EASYREG_REPOSITORY_PATH
cd doc
make html
```

Now you are ready to explore various optimization-based as well as learning-based demos provided by EasyReg.

# Related papers

If you find EasyReg is helpful, please cite (see [bibtex](citations.bib)):

Networks for Joint Affine and Non-parametric Image Registration [[link]](https://arxiv.org/pdf/1903.08811.pdf)\
Zhengyang Shen, François-Xavier Vialard, Marc Niethammer. NeurIPS 2019.


Region-specific Diffeomorphic Metric Mapping [[link]](https://arxiv.org/pdf/1906.00139.pdf)\
Zhengyang Shen, Xu Han, Zhenlin Xu, Marc Niethammer. CVPR 2019.









    
