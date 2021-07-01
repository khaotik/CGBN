ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif
ifdef GTEST_HOME
  GTEST_DIR := $(GTEST_HOME)
endif
ifndef GTEST_HOME
  GTEST_DIR := ../gtest
endif


tester: libgtest.a
	nvcc $(INC) $(LIB) -I$(GTEST_DIR)/include -I../include -std=c++11 -arch=sm_$(CUDA_GENCODE) tester.cu libgtest.a -lgmp -Xcompiler -fopenmp -o tester

.PHONY: clean check
clean:
	rm -f libgtest.a gtest-all.o tester

libgtest.a: check
	g++ -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -pthread -std=c++11 -c $(GTEST_DIR)/src/gtest-all.cc
	ar -rv libgtest.a gtest-all.o
	rm gtest-all.o

check:
	@if [ -z "$(GTEST_HOME)" -a ! -d "../gtest" ]; then echo "Google Test framework required, see XMP documentation"; exit 1; fi

