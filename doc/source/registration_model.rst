Registration models
===================
This section lists the detailed inferface for supported registration methods.

.. contents::

.. _base_model-label:


ModelBase
^^^^^^^^^

A base class provides shared functions for both optimization-/network- based methods.

.. inheritance-diagram:: model_pool.base_mermaid
.. automodule:: model_pool.base_mermaid
	:members:
	:undoc-members:






.. _base_mermaid-label:

MermaidBase
^^^^^^^^^^^

A base class provides shared functions for both optimization-/network- based methods.

.. inheritance-diagram:: model_pool.base_mermaid
.. automodule:: model_pool.base_mermaid
	:members:
	:undoc-members:



.. _base_toolkit-label:

ToolkitBase
^^^^^^^^^^^^

A base class provides shared functions for toolkit methods (Ants, Niftyreg, Demons).

.. inheritance-diagram:: model_pool.base_toolkit
.. automodule:: model_pool.base_toolkit
	:members:
	:undoc-members:




.. _mermaid_iter-label:


Optimized Mermaid
^^^^^^^^^^^^^^^^^

A class provides an interface for mermaid optimization-based registration.

.. inheritance-diagram:: model_pool.mermaid_iter
.. automodule:: model_pool.mermaid_iter
	:members:
	:undoc-members:



.. _reg_net-label:


Registration Net
^^^^^^^^^^^^^^^^

A base class provides functions for network-based registration.

.. inheritance-diagram:: model_pool.reg_net
.. automodule:: model_pool.reg_net
	:members:
	:undoc-members:


.. _ants_iter-label:


AntsPy
^^^^^^

A class provides an interface for AntsPy registration.

.. inheritance-diagram:: model_pool.ants_iter
.. automodule:: model_pool.ants_iter
	:members:
	:undoc-members:


AntsPy Utils
^^^^^^^^^^^^

Functions for AntsPy registration.

.. inheritance-diagram:: model_pool.ants_utils
.. automodule:: model_pool.ants_utils
	:members:
	:undoc-members:



Demons
^^^^^^

A class provides an interface for Demons registration.

.. inheritance-diagram:: model_pool.demons_iter
.. automodule:: model_pool.demons_iter
	:members:
	:undoc-members:


Demons Utils
^^^^^^^^^^^^

Functions for Demons registration.

.. inheritance-diagram:: model_pool.demons_utils
.. automodule:: model_pool.demons_utils
	:members:
	:undoc-members:


NiftyReg
^^^^^^^^

A class provides an interface for Niftreg registration.

.. inheritance-diagram:: model_pool.nifty_reg_iter
.. automodule:: model_pool.nifty_reg_iter
	:members:
	:undoc-members:


NiftyReg Utils
^^^^^^^^^^^^^^

Functions for Niftreg registration.

.. inheritance-diagram:: model_pool.nifty_reg_utils
.. automodule:: model_pool.nifty_reg_utils
	:members:
	:undoc-members:
