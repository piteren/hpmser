## hpmser - Hyper Parameters Searching Function

    searches hyper parameters space to MAXIMIZE the SCORE of given func

        MAXIMIZE the SCORE == find points cluster with:
         - high num of points in a small distance (dst)
         - high max smooth_score
         - high lowest score

         policy of sampling the space is crucial (determines the speed, top result and convergence)
         - fully random sampling is slow, wastes a lot of time and computing power
         - too aggressive sampling may undersample the space and miss the real MAX

    to setup search:
    1. having some func (callable):
        - some parameters need to be optimized
        - some parameters may be fixed / constant
        - if function accepts 'device' or 'devices' it should be type of DevicesPypaq (check pypaq.mpython.devices),
          it will be used by hpmser to put proper device for each function call
        - returns a dict with 'score' or just a value (score)
        There are two parameters of FUNCTION that may be used by hpmser:
            - 'device' (type of DevicesPypaq - check pypaq.mpython.devices) -> hpmser will send device to func
            - 'hpmser_mode' -> will be set to True by hpmser
    2. define PSDD - dictionary with parameters to be optimized and the space to search in (check pypaq.pms.paspa.PaSpa)
    3. import hpmser function into your script, and run it with:
        - func
        - func_psdd << PSDD
        - func_const << dictionary of parameters that have to be given but should be fixed / constant durring optimization
        - .. other params configure hpmser algorithm itself
