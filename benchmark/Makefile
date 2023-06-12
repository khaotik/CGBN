.PHONY: clean gmp-run gmp-numactl-run xmp-run

ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

all: gmp_tester xmp_tester

clean:
	rm -f xmp_tester gmp_tester

gmp-run: gmp_tester
	@./gmp_tester

gmp-numactl-run: gmp_tester
	numactl --cpunodebind=0 ./gmp_tester

xmp-run: xmp_tester
	@./xmp_tester

gmp_tester:
	g++ $(INC) $(LIB) gmp_tester.cc -o gmp_tester -lgmp -fopenmp

xmp_tester: 
	nvcc $(INC) $(LIB) -I../include -arch=sm_$(CUDA_GENCODE) xmp_tester.cu -o xmp_tester -lgmp
