ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

TARGET := mr
SRCFILE := miller_rabin.cu

all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)

$(TARGET): $(SRCFILE)
	nvcc $(INC) $(LIB) -I../../include -arch=sm_$(CUDA_GENCODE) $< -o $@ -lgmp
