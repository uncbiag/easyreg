# EasyReg
The EasyReg is a image registration package, that support a series of mainstream image registration methods.\
The EasyReg has following properties:
1) A simple interface for learning/optimization based image registration tools, e.g., [Mermaid](https://github.com/uncbiag/mermaid) and other popular registration
packages<sup>*</sup>.
2) A simple interface for fluid-based data augmentation, supporting either random augmentation and data interpolation/extrapolation
3) A simple interface for image segmentation.


\*The currently supported methods include Mermaid-optimization (i.e., optimization-based fluid registration) and Mermaid-network (i.e., deep network-based fluid registration methods using the mermaid deformation models).
We also added support for [ANTsPy](https://github.com/ANTsX/ANTsPy), [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg) and Demons(embedded in [SimpleITK](http://www.simpleitk.org/SimpleITK/resources/software.html)), though we recommend using the official source.

Currently, we support 3d image registration (2d is in progress).

<br/>

## Build Documentation
The latest doc can be found here [https://easyreg-unc-biag.readthedocs.io/en/latest/](https://easyreg-unc-biag.readthedocs.io/en/latest/).

Source code documentation and tutorials can be built on local using sphinx.

```
cd EASYREG_REPOSITORY_PATH
cd doc
make html
```
Now you are ready to explore various optimization-based as well as learning-based demos provided by EasyReg.

<br/><br/>

<br/>

## Registration 
We provide abundant of demos for both learning and optimization methods :)<br/>
Demo list (for more details, please refer to the doc<sup>*</sup>)
1) ANTsPy on OAI (knee MRI of the Osteoarthritis Initiative dataset)
2) NiftyReg on OAI
3) Demons on OAI
4) Optimization-based mermaid registration on OAI ([vSVF](https://arxiv.org/pdf/1903.08811.pdf))
5) Optimization-based mermaid registration on lung pairs<sup>**</sup> (inspiration to expiration) ([RDMM](https://arxiv.org/pdf/1906.00139.pdf))
6) Pretrained learning-based mermaid registration on OAI (vSVF, RDMM)
7) A training demo for joint affine and vSVF registration on sub-OAI dataset (3 pairs)
8) A training demo for Voxelmorph like framework.

\* For 2D demo (vSVF, LDDMM, RDMM) on synthetic data, please refers to [mermaid](https://mermaid.readthedocs.io/en/latest/notes/rdmm_example.html)<br/>
** Thanks Dr. RaúlSan José Estépar for providing the lung data
<br/>
<br/><br/>


For the learning part, the easyreg provides a two-stage learning framework including affine registration and non-parametric registration (map-based). 



An illustration of learning architecture 

<img src="figs/learning_framework.png" alt="learning_framework" width="800"/><br>



Registration results from Region-spec Region-specific Diffeomorphic Metric Mapping (RDMM)

<img src="figs/rdmm_oai_learn_opt.png" alt="rdmm_oai_learn_opt" width="800"/><br>

<br/><br/>


<br/>

## Data Augmentation/Interpolation

We provide an easy interface for fluid-based data augmentation and interpolation/extrapolation.
A typical strategy is as following:
1) Construct a geodesic subspace with given a source image and a set of target images.
2) Sample transformations from the resulting geodesic subspace.
3) Obtain deformed images and segmentations via interpolation.

We provide two demo for data augmentation for affined and non-affined dataset, see.


<br/>

## Segmentation
We provide an interface for image segmentaion task with a u-net structure.
The segmentation framework supports an ensemble method given inverse transformation maps that can be obtained via test phase data augmentation.

An illustration of ensemble method during test.

<img src="figs/test_phase.png" alt="test_phase" width="800"/><br>

<br/><br/>


# Installation
```
conda create -n easyreg python=3.6
conda activate easyreg
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
<br/><br/>




# Related papers

If you find EasyReg is helpful, please cite (see [bibtex](citations.bib)):

Networks for Joint Affine and Non-parametric Image Registration [[link]](https://arxiv.org/pdf/1903.08811.pdf)\
Zhengyang Shen, Xu Han, Zhenlin Xu, Marc Niethammer. CVPR 2019.


Region-specific Diffeomorphic Metric Mapping [[link]](https://arxiv.org/pdf/1906.00139.pdf)\
Zhengyang Shen, François-Xavier Vialard, Marc Niethammer. NeurIPS 2019.




# Our other registration work

See https://github.com/uncbiag/registration for an overview of other registration approaches of our group and a short summary of how the approaches relate.








    
