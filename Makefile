#!/usr/bin/make -f

NVCC_ARCHES += -gencode arch=compute_20,code=sm_20
#NVCC_ARCHES += -gencode arch=compute_20,code=sm_21

CFLAGS = -O3 -Wall
NVCC_FLAGS =  -use_fast_math $(NVCC_ARCHES) --compiler-options="$(CFLAGS) -fno-strict-aliasing" --ptxas-options=-v --maxrregcount 20

all: gpuLucas

gpuLucas.o: gpuLucas.cu IrrBaseBalanced.cu
	nvcc -c -o $@ gpuLucas.cu -O3 $(NVCC_FLAGS)

gpuLucas: gpuLucas.o
	g++ -fPIC $(CFLAGS) -o $@ $^  -Wl,-O1 -Wl,--as-needed -lcudart -lcufft -lqd

clean:
	-rm *.o *~
	-rm gpuLucas
