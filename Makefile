all: check .cudaArch
	$(eval export CUDA_GENCODE = $(shell cat .cudaArch))
	make -C samples
	make -C unit_tests
	make -C perf_tests

.PHONY: clean download-gtest check
clean:
	make -C samples clean
	make -C unit_tests clean
	make -C perf_tests clean

download-gtest:
	wget 'https://github.com/google/googletest/archive/master.zip' -O googletest-master.zip
	unzip googletest-master.zip 'googletest-master/googletest/*'
	mv googletest-master/googletest gtest
	rmdir googletest-master
	rm -f googletest-master.zip

.cudaArch: cuda-caps.cu
	nvcc cuda-caps.cu -o a.out
	./a.out > .cudaArch
	rm -f a.out

check:
	@if [ -z "$(GTEST_HOME)" -a ! -d "gtest" ]; then echo "Google Test framework required, see documentation"; exit 1; fi
