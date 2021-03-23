Installation Guide
===================

This note briefly describes how to install and use *EasyReg*. Unfortunately, it is currently not available on Conda, but it is fairly simple to instally manually. It is recommended to use virtual environment to avoid conflicts.

Requirements
^^^^^^^^^^^^^^
Our framework needs the following:


  - python == 3.6
  - pytorch >= 1.0

It runs both on Mac OS and Linux, it is not tested in Windows.


Step 0: Installing Anaconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This step mostly skipped, but we rely on Anaconda to install the package, if you are missing Anaconda, their installation manual can be found in this link:
https://docs.anaconda.com/anaconda/install/

Step 1: Creating virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As suggested, the virtual environment can be initiliazed with the following command:

.. code:: shell

    conda create -n easyreg python=3.6
    conda activate easyreg

If at any point, it can be deleted via following:

.. code:: shell

    conda remove --name easyreg --all


Step2: Downloading the files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As our package is not available at PyPI, it is best to refer our repository to download the package files from our repository.

.. code:: shell

    git clone https://github.com/uncbiag/easyreg.git


We further need to download .. _Mermaid: https://mermaid.readthedocs.io/ as our framework heavily depends on the primitives built in Mermaid library.

.. code:: shell

    cd easyreg
    git clone https://github.com/uncbiag/mermaid.git



Step3: Installing dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Other dependencies, such as ITK, can be installed from our requirements.txt.

.. code:: shell

    pip install -r requirements.txt

Step4: Installing Mermaid 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mermaid framework needs to be installed seperately, it can be done with two simple steps:

.. code:: shell

    cd mermaid
    python setup.py develop