![](hpmser.png)

## hpmser - Hyper Parameters Search tool

------------

**hpmser** is a tool for searching optimal hyper-parameters of any function. Assuming there is a function:

`def some_function(a,b,c,d) -> float`

**hpmser** will search for values of `a,b,c,d` that MAXIMIZE return value of given function.

To start the search process you will need to:
- give a `func` (type)
- pass to `func_psdd` parameters space definition (with PSDD - check `pypaq.pms.base.py` for details)
- if some parameters are *known constants*, you may pass their values to `func_const`
- configure `devices`, `n_loops` and optionally other advanced hpmser parameters

You can check `/examples` for sample run code.<br>There is also a project: https://github.com/piteren/hpmser_rastrigin
that uses **hpmser**.

------------

**hpmser** implements mix of:
- optimal space sampling based on current space knowledge (currently obtained results interpolation), with
- random search

**hpmser** features:
- multiprocessing (runs with subprocesses) with CPU & GPU devices with **devices** param - check `pypaq.mpython.devices` for details
- exceptions handling, keyboard interruption without a crash
- process auto-adjustment
- process fine-tuning by user (while running)
- process saving & resuming
- 3D visualisation of parameters and function values
- TensorBoard logging

------------

If you got any questions or need any support, please contact me:  me@piotniewinski.com