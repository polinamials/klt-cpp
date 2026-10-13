# About

klt-cpp is a C++ implementation of the Kanade-Lucas-Tomasi Feature Tracker based on the [C implementation by Stan Birchfield](https://cecas.clemson.edu/~stb/klt/).

# Build

There are no dependencies. To build, run:

```bash
cmake -S . -B build
cmake --build build --parallel
```