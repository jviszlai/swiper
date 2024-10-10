"""TODO: test that all variants of WindowManager work correctly. Specific cases
to test:
    1) All: check that windows are created and processed correctly, e.g. not
       releasing windows until all buffer regions are complete, and not waiting
       excessively long to release a window once it is ready.
    1) SlidingWindowManager: check that each window has at most one buffer
       region and that DAG is tree-like
    2) ParallelWindowManager: Check number of layers of parallel windows in
       simple examples
    3) DynamicWindowManager: TODO
    ...
"""