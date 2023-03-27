![](HPMSer.png)

## HPMSer - Hyper Parameters Search tool

------------

**HPMSer** is a tool for searching optimal hyper-parameters of any function.<br> Assuming there is a function:

`def some_function(a,b,c,d) -> float`

**HPMSer** will search for values of `a,b,c,d` that MAXIMIZE return value of given function.

To start the search process you will need to create object of `HPMSer` class giving to its `__init__`:
- give a `func` (type)
- pass to `func_psdd` parameters space definition (with PSDD - check `pypaq.pms.base.py` for details)
- if some parameters are *known constants*, you may pass their values to `func_const`
- configure `devices`, `n_loops` and optionally other advanced HPMSer parameters

You can check `/examples` for sample run code.<br>There is also a project: https://github.com/piteren/hpmser_rastrigin
that uses **HPMSer**.

------------

**HPMSer** implements:
- smart search with evenly spread out quasi random sampling of space
- parameters space estimation with regression using SVC RBF (Support Vector Regression with Radial Basis Function kernel)
- space sampling based on current space knowledge (estimation)

**HPMSer** features:
- multiprocessing (runs with subprocesses) with CPU & GPU devices with 'devices' param - check `pypaq.mpython.devices` for details
- exceptions handling, keyboard interruption without a crash
- process auto-adjustment
- process saving & resuming
- 3D visualisation of parameters and function values
- TensorBoard logging of process parameters

------------

If you got any questions or need any support, please contact me:  me@piotniewinski.com