Registration models
===================
This section lists the detailed inferface for supported registration methods.

.. contents::

.. _base_model-label:


RegModelBase
^^^^^^^^^

A base class provides shared functions for both optimization-/network- based methods.

.. inheritance-diagram:: easyreg.base_mermaid
.. automodule:: easyreg.base_mermaid
	:members:
	:undoc-members:






.. _base_mermaid-label:

MermaidBase
^^^^^^^^^^^

A base class provides shared functions for both optimization-/network- based methods.

.. inheritance-diagram:: easyreg.base_mermaid
.. automodule:: easyreg.base_mermaid
	:members:
	:undoc-members:



.. _base_toolkit-label:

ToolkitBase
^^^^^^^^^^^^

A base class provides shared functions for toolkit methods (Ants, Niftyreg, Demons).

.. inheritance-diagram:: easyreg.base_toolkit
.. automodule:: easyreg.base_toolkit
	:members:
	:undoc-members:




.. _mermaid_iter-label:


Optimized Mermaid
^^^^^^^^^^^^^^^^^

A class provides an interface for mermaid optimization-based registration.

.. inheritance-diagram:: easyreg.mermaid_iter
.. automodule:: easyreg.mermaid_iter
	:members:
	:undoc-members:



.. _reg_net-label:


Registration Net
^^^^^^^^^^^^^^^^

A base class provides functions for network-based registration.

.. inheritance-diagram:: easyreg.reg_net
.. automodule:: easyreg.reg_net
	:members:
	:undoc-members:


.. _ants_iter-label:


AntsPy
^^^^^^

A class provides an interface for AntsPy registration.

.. inheritance-diagram:: easyreg.ants_iter
.. automodule:: easyreg.ants_iter
	:members:
	:undoc-members:


AntsPy Utils
^^^^^^^^^^^^

Functions for AntsPy registration.

.. inheritance-diagram:: easyreg.ants_utils
.. automodule:: easyreg.ants_utils
	:members:
	:undoc-members:



Demons
^^^^^^

A class provides an interface for Demons registration.

.. inheritance-diagram:: easyreg.demons_iter
.. automodule:: easyreg.demons_iter
	:members:
	:undoc-members:


Demons Utils
^^^^^^^^^^^^

Functions for Demons registration.

.. inheritance-diagram:: easyreg.demons_utils
.. automodule:: easyreg.demons_utils
	:members:
	:undoc-members:


NiftyReg
^^^^^^^^

A class provides an interface for Niftreg registration.

.. inheritance-diagram:: easyreg.nifty_reg_iter
.. automodule:: easyreg.nifty_reg_iter
	:members:
	:undoc-members:


NiftyReg Utils
^^^^^^^^^^^^^^

Functions for Niftreg registration.

.. inheritance-diagram:: easyreg.nifty_reg_utils
.. automodule:: easyreg.nifty_reg_utils
	:members:
	:undoc-members:
