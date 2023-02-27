## hpmser - Hyper Parameters Search

------------

Assuming that there is a function:

`    def some_function(a,b,c,d) -> float`

**hpmser()** will search for values of **a,b,c,d** that MAXIMIZE return value of given function.

To start the search process you will need to:
- give a **func** (type)
- pass to **func_psdd** parameters space definition (with PSDD - check `pypaq.pms.base.py` for details)
- if some parameters are *known constants*, you may pass their values to **func_const**
- configure **devices** used and optionally advanced hpmser parameters

You can check `/tests` for run example. There is also a project: https://github.com/piteren/hpmser_rastrigin
that uses **hpmser**.

------------

**hpmser** implements mix of:
- optimal space sampling based on current space knowledge (currently obtained results interpolation), with
- random search

**hpmser** supports:
- multiprocessing (runs with subprocesses) with CPU & GPU devices with **devices** param - check `pypaq.mpython.devices` for details
- exceptions handling, keyboard interruption without a crash
- live process configuration and fine-tuning
- process saving & resuming
- 3D visualisation of parameters and function values
- TensorBoard logging

If you got any questions or need any support, please contact me:  me@piotniewinski.com