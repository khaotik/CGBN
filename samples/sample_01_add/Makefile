ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

SRCFILE := add.cu
TARGET := add

all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)

$(TARGET): $(SRCFILE)
	nvcc $(INC) $(LIB) -I../../include -arch=sm_$(CUDA_GENCODE) $(SRCFILE) -o $(TARGET) -lgmp
