
### [PyCon Talk](https://www.youtube.com/watch?v=yrRqNzJTBjk&ab_channel=PyCon2018)
- If you code that is easy to read, works, it might be okay in many scenarios. Do not feel fear or shame to ask do we really need to make this code faster ?
- Profiling
    - Collect data 
    - Analyze 
    - Change/Experiment
    - Repeat 
- CProfile
    - inbuilt profiler into python, shows time spent per function call
    - Command line 
        -`python -m cProfile -o output_file my_script.py`
    - iPython 
        - `%%prun -q -D output_file`
- pstarts gives text based view of output data. 
    ```
        data = pstats.Stats('output_file.cprof')
        data.sort_stats('cummulative').print_stats('output_file', 2)
    ```
- cummultative time - total amount of time under a function. totaltime - time spent inside a function but not on things it calls. 
- `snakeviz` - tool shows function call hierarchy and time spent per function. 
- If you have fast functions(eg percall <= 1e-5) with O(1e5+) calls then focus on minimizing number of calls, you cannot make the execution of the function faster. 
- First way to improve is to use algorithmic ideas to make bottleneck function faster. 
- If you want to go faster python, you will have to do some form of compilation. 
    - `PyPy` alternate implementation of python does compilation of repeated things, makes things run faster. 
    - `Numba` compiles pythjon on the fly. 100% python, numba decorator compiles functions using LLVM. LLVM installation is required. Numpy for math. After first execution of function, instructions are compiled, further executions are blazing fast. Numba will inspect function types and build compiled code. 
    - No python objects will be involved. 
    - `Cython` compile code to c and run quick. 
- [Fun stuff](https://adventofcode.com/)
