ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

all: mo mo_faster

.PHONY: clean
clean:
	rm -f mo mo_faster

mo: modinv_odd.cu
	nvcc $(INC) $(LIB) -I../../include -arch=sm_$(CUDA_GENCODE) $< -o $@ -lgmp

mo_faster: modinv_odd_faster.cu
	nvcc $(INC) $(LIB) -I../../include -arch=sm_$(CUDA_GENCODE) $< -o $@ -lgmp
