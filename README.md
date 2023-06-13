### CGBN fork

This is my fork repo of original [CGBN](https://github.com/nvlabs/CGBN).
For now, it's mainly some QoL improvements.
The original `README.md` is available in this repo as `README_ORIG.md`.

### Differences to the original CGBN

- CMake support, this allows for automatic detection of CUDA architecture on your system.
- Requires C++14
- Tests are now compiled in several compilation units instead of one.
- `cgbn_context_t` and `cgbn_env_t` at different devices (e.g CPU vs GPU) are now instantiated with template parameter instead of macro controlled.
- New method: `all_set_ui32` `all_equals_ui32`.
- Fixed a few minor bugs in the orignal repo.

### Building with CMake

- Make sure OpenMP and GMP are installed in your system.

- `git clone --recurse-submodules <this-repo>`

- Generate a cmake build directory:
`mkdir -p build; cmake -B build -S ./`

- Build 
`cd build && make -j`

- To run tests, run `./tester` under build directory

- To run benchmark, under build directory, run `./xmp_bench` for GPU and `./gmp_bench` for CPU
