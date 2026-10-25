# About

klt-cpp is a C++ implementation of the Kanade-Lucas-Tomasi Feature Tracker based on the [C implementation by Stan Birchfield](https://cecas.clemson.edu/~stb/klt/).

## Build

There are no dependencies. To build, run:

```bash
cmake -S . -B build
cmake --build build
```

## Results

The feature extraction and tracking results are identical to the results produced by the original C implementation (`test/orig_klt_values.hpp`). You can verify this by running the tests with

```bash
./build/test/klt-test --output-on-failure
```

## Benchmark

Benchmark performed on the 10 sample images provided in `images/`. `klt-cpp` achieves approximately 1.5x faster tracking.

```
Median time (milliseconds; lower is better)
Workload                       Original C         klt-cpp      Speedup
----------------------------------------------------------------------
Feature selection                   8.062           6.939        1.162x
Pair tracking                      11.083           7.529        1.472x
Full sequence                     124.178          80.162        1.549x
Sequence / transition              13.798           8.907        1.549x
```