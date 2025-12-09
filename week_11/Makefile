NVCC      ?= nvcc
MPICXX    ?= mpicxx
NVSHMEM_HOME ?= $(shell dirname `dirname \`which nvshmem_info 2>/dev/null\``)

INCLUDES  := -I$(NVSHMEM_HOME)/include -Iinclude
LIBS      := -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device
CXXFLAGS  := -O2 -std=c++17
NVCCFLAGS := $(CXXFLAGS) -Xcompiler -fopenmp

SRC_CPP = src/flash_dmoe_host.cpp
SRC_CU  = src/flash_dmoe_kernel.cu src/flash_dmoe_driver.cu

OBJS_CPP = $(SRC_CPP:.cpp=.o)
OBJS_CU  = $(SRC_CU:.cu=.o)

all: flash_dmoe_demo

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -ccbin $(MPICXX) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -ccbin $(MPICXX) -c $< -o $@

flash_dmoe_demo: $(OBJS_CPP) $(OBJS_CU)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OBJS_CPP) $(OBJS_CU) -o $@ $(LIBS)

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) flash_dmoe_demo
