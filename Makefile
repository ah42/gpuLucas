#!/usr/bin/make -f

# PTX code embedded in the application, and will be assembled on-the-fly by the nvidia driver
NVCC_ARCHES += -gencode arch=compute_20,code=compute_20 -gencode arch=compute_13,code=compute_13

# Compiled targets, binary images will be embedded in the application
#NVCC_ARCHES += -gencode arch=compute_20,code=sm_20 -gencode arch=compute_13,code=sm_13

OPT = -O3
CFLAGS = $(OPT) -Wall
NVCC_FLAGS = $(OPT) -use_fast_math $(NVCC_ARCHES) --compiler-options="$(CFLAGS) -fno-strict-aliasing" --ptxas-options=-v --maxrregcount 20

all: gpuLucas

gpuLucas.o: gpuLucas.cu IrrBaseBalanced.cu
	nvcc -c -o $@ gpuLucas.cu $(NVCC_FLAGS)

gpuLucas: gpuLucas.o
	g++ -fPIC $(CFLAGS) -o $@ $^  -Wl,-O1 -Wl,--as-needed -lcudart -lcufft -lqd

clean:
	-rm *.o *~
	-rm gpuLucas
