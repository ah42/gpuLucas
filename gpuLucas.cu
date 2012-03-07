/**
* gpuLucas.cu
*
**************************************************************************** 
* Copyright (c) 2012, Aaron Haviland
* Copyright (c) 2010-2012, Andrew Thall, Alma College
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the names of Andrew Thall or Alma College, nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL ANDREW THALL OR ALMA COLLEGE BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************
*
* A. Thall & A. Hegedus
* Alma College
* 9/1/2010
*
* Implementing the IBDWT method of Crandall in CUDA.
*   This uses a variable base representation and a weighted tranformation
* to reduce the FFT length and eliminate the modular reduction mod M_p.
*
* gpuLucas uses carry-save arithmetic to eliminate carry-adds beyond a single
*   ripple-carry from each digit to the next, following the radix-restoration
*   (dicing) of the convolution products.
*
* The radix-restoration code in IrrBaseBalanced.cu is an ugly kludge,
*   slightly better with templating (thanks, Alex), but it makes up only 1/6th
*   of the runtime, the rest being the weighted transform and componentwise
*   complex multiplication, so might pretty it up, great, but it won't run
*   much faster overall.
*
******************************************************************************
* Tested:  GTX480 and Tesla C2050, Cuda versions 3.2, 4.0, 4.1
* Compiled with Visual Studio 2008, x64.
*   Uses 64-bit (int64_t) and will probably not work in 32-bit x86.
*
* Files:
*    gpuLucas.cu -- main file, including main() and mersenneTest() functions
*    IrrBaseBalanced.cu -- include file (i.e., header, not separate compilation)
*        with the radix-restoration code llintToBalInt<n>() templated routines.
*
* Dependencies:
*    CUFFT
*    QD extended-precision library for dd_real, double-double class
*        (Computed weights for IBDWT for non-power-of-two FFTs
*         suffered catastrophic cancellation in double.)
*         QD at http://crd-legacy.lbl.gov/~dhbailey/mpdist/
*
* Key routines:
*    main() -- sets up the constants for the GPU
*              calls errorTest(testPrime, signalSize), outputs timing and error data
*              calls mersenneTest(testPrime, signalSize) to do full test
*    errorTest(int numIterations, int testPrime, int signalSize)
*    mersenneTest(int testPrime, int signalSize)
*
* Implementing balanced-integers in the irrational base
*
* In bitsPerWord, we use a bit-vector:
*    0 -- low base word
*    1 -- high base word
* Where the positions 0=current, 1=previous, 2=previousprevious, etc.
*    The h_BASE_HI, h_BASE_LO, h_HI_BITS, h_LO_BITS are global constants
* on the host, and BASE_HI, BASE_LO, HI_BITS, LO_BITS, etc., on the device.
* 
* Since minimum word-sizes are (8, 9) in our ranges, never need carry-out bits
* from more than the six preceding terms for a product term, usually, no more than
* two or three with wds of length (18, 19) typical.
*
* NOTE:  must use extended precision to compute A and Ainv for non-power-of-two FFT runlengths
*    We do this on the host using qd library.
*    (Need double-double; gcc's long double is not precise enough.)
*
*  M42643801 took 208299.3 sec/57.86 hours/2.41 days
*    It did 204.72 Lucas iterations per second = 4.88 msec per iteration
*    It used a DWT runlength of 2359296 = 2^21 + 2^18 = 2^18*3^2.
*    and a word-sizes of (18, 19) bits
*    Maximum error reported was 1.8e-1
*  M43112609 to 211447 sec, 58.7 hours, 2.45 days, runlength 2359296
*    M859433 to    112.6 sec, with wd = (17, 18), also with 2 prior words
*   M1257787 to    198.8 sec, with wd = (19, 20), also with 2 prior words, runlength 65536
*                  247.9 sec, on non-overclocked GTX 480
*
*
* Latest build of CUDA and SDK (3.2.12):
*    M859433 to    112.6 sec, with wd = (17, 18), runlength 49152, with 2 prior words
*   M1257787 to    197.2 sec, with wd = (19, 20), also with 2 prior words, runlength 65536 
*                  244.2 sec, on non-overclocked GTX 480
*   M3021377 to   1231.4 sec, with wd = (18, 19), runlength 163840 (2^17+2^15), with 2 prior words
*                                                                 == 2^15*5
*                                                 How about 2^16*3
* 7/2/2011 -- CUDA 4.0
*   M1257787 to    249.2 sec, with wd = (19, 20), two prior words, runlength 65536 on GTX 480
*            to    196.3 sec on Tesla c2050, o/c to 701/1402/1720 Ghz core/proc/mem clocks
*
* 8/1/2011 -- Alex Hegedus and A Thall:
*    Removed CUDPP dependencies
*    Changed to template-based llintToBalInt() for different numbers of carry-bits
*    Rewrote with separate full-test and test-profiling methods
*   M1257787 to    243.5 sec, with wd = (19, 20), two prior words, runlength 65536 on GTX 480
*
* 2/16/2012 -- A Thall
*    CUDA 4.1
*    Can no longer overclock Tesla c2050 with latest NVIDIA GPU control panel
*       No appreciable difference in runtimes between cards...extra processors on 480
*       balance out better floating point performance of Tesla.  480 slightly faster
*       for shorter FFT lengths, tesla for larger
*    Removed broken maxScan code (had been to replace CUDPP dependencies)
*       Replaced with dev-to-host xfer and computation on CPU.
*    MaxScan only used 50 times; if need an actual error-tolerance check for every iteration,
*       write a kernel to simply check each value and do an atomic-set to a signal-flag.
* 2/19/2012 -- A Thall
*    Renamed as gpuLucas
*    Cleanup and documentation of code for release
*    Stripped timing code from mersenneTest()
*
* To do (xxAT: 2/19/2012):
*    1) Timing code is a jumbled up mess.
*    2) For a pipelined and double-checked system, need a lot more automagic routines
*         Also need to be able to save current run after X iterations for rechecking,
*         save-and-restart on a multi-user, massively-multi-GPU system.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <stdint.h>
// most compilers do not have support for __float128 and needs an external library to
// support extended precision
#include <qd/dd_real.h>

// includes, project
#include <cufft.h>
#include "cuda_safecalls.h"

// NOTE: testPrimes below 9689 generate runlengths < 1024, which breaks the code if T_PER_B = 1024
// Create ThreadsPerBlock constant
const int T_PER_B = 512;

// These determine the highest FFT signalSize we will check in findSignalSize()
// where signalSize == 2^MAX_2 * 3^MAX_3 * 5^MAX_5 * 7^MAX_7
#define MAX_2 24 // 16777216
#define MAX_3 15 // 14348907
#define MAX_5 10 // 9765625
#define MAX_7 8  // 5764801

// This determines the maximum allowable roundoff error
#define ERROR_LIMIT 0.40f

// At runtime, set constant and load to GPU for use in IrrBaseBalanced.cu code
int h_LO_BITS;
int h_HI_BITS;
int h_BASE_LO;
int h_BASE_HI;
int h_LO_MODMASK;
int h_HI_MODMASK;

// some global flags
int opt_quiet = 0;
int opt_verbose = 0;
char program_name[] = "gpuLucas";
char program_version[] = "0.9.1";

int resuming = 0;
// checkpoint filename buffers
char checkpoint_file[32];
char checkpoint_backup[32];
// Default checkpoint interval in iterations:
int checkpoint_freq = 10000;

__constant__ int LO_BITS;
__constant__ int HI_BITS;
__constant__ int BASE_LO;
__constant__ int BASE_HI;
__constant__ int LO_MODMASK;
__constant__ int HI_MODMASK;

// Need this include after T_PER_B so can use as shared memory array-length
//   in IrrBaseBalanced.cu routines to avoid dynamic memory alloc on GPU
//   (xxAT sloppy, but okay for now) (means need to recompile for different T_PER_B but
//         have removed NUMBLOCKS dependency, so can do runs of different lengths
// Also needs LO_BITS, etc., constant declarations for templated routines
// This includes all code for parallel carry-add of the balanced-variable base integers
#include "IrrBaseBalanced.cu"

// NOTE:  The largest block size ensures a minimum number of redundant double2llint() functions
//   are called (six extra per block to round product terms and place them in shared memory for
//   "dicing" into individual, variable length words.
//const int NUM_BLOCKS = SIGNAL_SIZE/T_PER_B;  // assume all divisible by T_PER_B

static __host__ void initConstantSymbols(int testPrime, int signalSize) {

	h_LO_BITS = testPrime/signalSize;
	h_HI_BITS = testPrime/signalSize + 1;
	h_BASE_LO = 1 << h_LO_BITS;
	h_BASE_HI = 1 << h_HI_BITS;
	h_LO_MODMASK = h_BASE_LO - 1;
	h_HI_MODMASK = h_BASE_HI - 1;
	cutilSafeCall(cudaMemcpyToSymbol(LO_BITS, &h_LO_BITS, sizeof(int)));
	cutilSafeCall(cudaMemcpyToSymbol(HI_BITS, &h_HI_BITS, sizeof(int)));
	cutilSafeCall(cudaMemcpyToSymbol(BASE_LO, &h_BASE_LO, sizeof(int)));
	cutilSafeCall(cudaMemcpyToSymbol(BASE_HI, &h_BASE_HI, sizeof(int)));
	cutilSafeCall(cudaMemcpyToSymbol(LO_MODMASK, &h_LO_MODMASK, sizeof(int)));
	cutilSafeCall(cudaMemcpyToSymbol(HI_MODMASK, &h_HI_MODMASK, sizeof(int)));
}

// Complex data type
typedef cufftDoubleComplex Complex;
typedef cufftDoubleReal Real;
#define CUFFT_TYPEFORWARD CUFFT_D2Z
#define CUFFT_TYPEINVERSE CUFFT_Z2D
#define CUFFT_EXECFORWARD cufftExecD2Z
#define CUFFT_EXECINVERSE cufftExecZ2D

/**
 * PREDECLARED FUNCTIONS:  these don't really need to be predeclared anymore,
 *   but give an overview of the functions so left it.
 */

static __global__ void ComplexPointwiseSqr(Complex*, int);
static __global__ void loadValue4ToFFTarray(double*, int);
static __global__ void loadIntToDoubleIBDWT(double *dArr, int *iArr, int *iHiArr, double *aArr, int size);

/*
 * In bitsPerWord, we use a bit-vector:
 *    0 -- low base word
 *    1 -- high base word
 * Where the positions 0=current, 1=next, 2=nextnext, etc.
 *    The BASE_HI, BASE_LO, HI_BITS, LO_BITS are global constants.
 */
static __host__ void computeBitsPerWord(int testPrime, int *bitsPerWord, int size);
static __host__ void computeBitsPerWordVectors(unsigned char *bitsPerWord8, int *bitsPerWord, int size);

/**
 * code for convolution error-checking
 */
static __global__ void computeMaxBitVector(float *dev_errArr, int64_t *llint_signal, int len);
static __host__ float findMaxErrorHOST(float *dev_fltArr, float *host_temp, int len);

/**
 * compute A and Ainv in extended precision, cast to doubles
 *   and load them to the host arrays.  We include the FFT 1/N scaling with
 *   host_ainv and pull it out of the pointwiseSqrAndScale code
 */
static __host__ void computeWeightVectors(double *host_A, double *host_Ainv, int testPrime, int size);

/**
 * This completes the invDWT transform by multiplying the signal by a_inv,
 *   and subtracts 2 from signal[0], requiring no carry in current weighted carry-save state
 */
template <int error>
static __global__ void invDWTproductMinus2(int64_t *llintArr, double *signal, double *a_inv, float *errorvals, int size);


/**
 * The sliceAndDice() function pointer is used to call the correct templated
 *   kernel function to do the distribution of convolution product-bits to
 *   higher-order digits.
 * How many previous convolution components will carry into a given digit depends both
 *   on the base being used and on the length of the convolution vector.  Moreso
 *   on the base...because we are using balanced integers, the product terms don't
 *   scale linearly with the length of the product, but by CLT tend toward a zero
 *   mean with a Gaussian distribution as n gets big.  Average case, but still get
 *   outliers and worst cases. 
 * With convolution wordsize typically (18, 19) bits, two preceding terms are typically needed.
 *   For shorter wordsizes, a product may need product bits from up to six lower-order words.
 * Use llintToIrrBal<2,3,4,5,6>, as appropriate.  And yes, we can have pointers
 *   to global kernels.  (Works fine, just address.)
 */
void (*sliceAndDice)(int *iArr, int *hiArr, int64_t *lliArr, unsigned char *bperW8arr, const int size);

/**
 * For n = 2 to 6. This uses templated kernel functions for the different lengths,
 *   as defined in IrrBaseBalanced.cu file.  (Thanks, Alex.)
 *   These seem to be good divisions for the sliceAndDice but might need to be adjusted.
 * Auto-selected signalSize will almost always choose cases 17 through 19.
 */
  
void setSliceAndDice(int testPrime, int signalSize) {

	int ratio = testPrime / signalSize;

	if (ratio >= 21) {
		fprintf(stderr, "testPrime / signalSize out of range: %d\n",
				(int)(testPrime / signalSize));
		exit(-1);
	}

	if (ratio >= 18)
		sliceAndDice = llintToIrrBal<2>;
	else if (ratio >= 15)
		sliceAndDice = llintToIrrBal<3>;
	else if (ratio >= 12)
		sliceAndDice = llintToIrrBal<4>;
	else if (ratio >= 9)
		sliceAndDice = llintToIrrBal<5>;
	else
		sliceAndDice = llintToIrrBal<6>;
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

/**
 * errorTrial() outputs timing and error information and returns
 *    the average time per Lucas-Lehmer iteration based on timing
 *    of convolution-multiply and rebalancing functions
 */
float errorTrial(unsigned int testIterations, unsigned int testPrime, unsigned int signalSize);
unsigned int findSignalSize(unsigned int testPrime);
void mersenneTest(unsigned int testPrime, unsigned int signalSize, Real *d_signal, unsigned int iter);

void writeCheckpoint(Real *signal, unsigned int testPrime, unsigned int signalSize, unsigned int iter);
Real *readCheckpoint(unsigned int testPrime, unsigned int *signalSize, unsigned int *resume_iter);

/**
 * print_help()
 */
void print_help() {
	fprintf(stderr, "%s (v%s):\n\n", program_name, program_version);
	fprintf(stderr, "\t-c\tspecify checkpoint frequency (default 10000 iterations\n");
	fprintf(stderr, "\t-d\tspecify CUDA device to use (default 0)\n");
	fprintf(stderr, "\t-v\tbe more verbose\n");
	fprintf(stderr, "\t-q\tbe less verbose\n");
	fprintf(stderr, "\t-f\tspecify signalLength (must be divisible by %d)\n", T_PER_B);
	fprintf(stderr, "\t-n\tspecify number to be tested\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	unsigned int signalSize = 0, testPrime = 0;
	int c;
	int use_device = 0;

	// getopt: parse command line options
	while (( c = getopt(argc, argv, "hvqf:n:d:c:")) != -1) {
		switch (c) {
			case 'h':
				print_help();
				return(1);
			case 'v':
				opt_verbose = 1;
				opt_quiet = 0;
				break;
			case 'q':
				opt_verbose = 0;
				opt_quiet = 1;
				break;
			case 'f':
				signalSize = atoi(optarg);
				if (signalSize%T_PER_B) {
					fprintf(stderr, "Option -f: must be divisible by %d\n", T_PER_B);
					return(-1);
				}
				break;
			case 'n':
				testPrime = atoi(optarg);
				break;
			case 'd':
				use_device = atoi(optarg);
				break;
			case 'c':
				checkpoint_freq = atoi(optarg);
				break;
			case '?':
				if (optopt == 'f' || optopt == 'n') {
					print_help();
					fprintf(stderr, "Option -%c requires an argument\n", optopt);
					return(-1);
				} else if (isprint (optopt)) {
					print_help();
					fprintf(stderr, "Unknown option `-%c'.\n", optopt);
					return(-1);
				}	
				break;
			default:
				break;
		}
	}

	for (int index = optind; index < argc; index++) {
		if (testPrime == 0 && (strlen(argv[index]) == strspn(argv[index],"0123456789"))) {
				testPrime = atoi(argv[index]);
		} else {
				print_help();
				printf ("Non-option argument %s\n", argv[index]);
				return(-1);
		}
	}
	if ( testPrime == 0) {
			print_help();
			fprintf(stderr, "testPrime not specified, aborting\n");
			abort();
	}

	sprintf(checkpoint_file, "%d" ".chk", testPrime);
	sprintf(checkpoint_backup, "%d" ".chk.bak", testPrime);

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
	}
	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		fprintf(stderr, "There is no device supporting CUDA\n");
		return(-1);
	} else
		fprintf(stderr, "Found %d CUDA Capable device(s)\n", deviceCount);

	int dev;
	cudaDeviceProp deviceProp;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaGetDeviceProperties(&deviceProp, dev);
		fprintf(stderr, "\tDevice %d: %s\n", dev, deviceProp.name);
	}

	// Chosen device = use_device
	cudaGetDeviceProperties(&deviceProp, use_device);
	fprintf(stderr, "Using device %d: %s\n\n", use_device, deviceProp.name);
	cudaSetDevice(use_device);

	unsigned int resume_iter;
	Real *resume_signal;
	if ((resume_signal = readCheckpoint(testPrime, &signalSize, &resume_iter)) != NULL)
		resuming = 1;

	if (signalSize == 0)  {
		signalSize = findSignalSize(testPrime);
		printf("Optimal signalSize detected: %d\n\n", signalSize);
	} else
		printf("Using specified FFT runlength %d\n\n", signalSize);

	// BEGIN by initializing constant memory on device
	initConstantSymbols(testPrime, signalSize);

	// Based on the problem size, and runlength, set the number of carry digits
	//   and assign the global slice-and-dice function from the templated
	//   llintToBalInt<n>() function
	setSliceAndDice(testPrime, signalSize);


	if ((int)sizeof(int64_t) != 8) {
		printf("size of int64_t = %d (if not 8, you're in trouble)\n", (int) sizeof(int64_t));
		exit(-1);
	}

	printf("Testing M%d, using an irrational base with wordlengths (%d, %d),\n"
		   "\tusing an FFT runlength of 2^%f = %d\n",
		   testPrime, h_LO_BITS, h_HI_BITS, log(1.0*signalSize)/log(2.0), signalSize);
	if (!opt_quiet)
		printf("\n\tNUM_BLOCKS = %d, T_PER_B = %d\n", signalSize/T_PER_B, T_PER_B);

	// START timer now
	cudaEvent_t errorTrial_start, errorTrial_stop, mersenneTest_start, mersenneTest_stop;
	cutilSafeCall(cudaEventCreate(&errorTrial_start));
	cutilSafeCall(cudaEventCreate(&errorTrial_stop));
	cutilSafeCall(cudaEventRecord(errorTrial_start, 0)); 

	
	// errorTrial() called to give an estimate of convolution sizes and errors,
	//  as well as FFT timings and rebalancing time.
	// return value is average time per Lucas-Lehmer iteration based on
	//   GPU timings
	int testIterations, trialFraction;
	if (!resuming) {
		trialFraction = 10000;
		testIterations = testPrime / trialFraction;
		// Make sure we run at least a couple iterations through errorTrial in case testPrime is too small
		testIterations = max(100, testIterations);
		// But not so many it takes forever on larger testPrimes
		testIterations = min(2500, testIterations);
	} else {
		// Only need a limited run on resuming, to establish timing
		testIterations = 100;
		trialFraction = testPrime / testIterations;
	}
	if (!opt_quiet)
		printf("\nRunning %d iterations in an error trial test before %s full test...\n",
				testIterations, resuming == 1 ? "resuming" : "beginning");
	float elapsedMsecDEV = errorTrial(testIterations, testPrime, signalSize);

	// stop the timer
	cutilSafeCall(cudaEventRecord(errorTrial_stop, 0));
	cutilSafeCall(cudaEventSynchronize(errorTrial_stop));

	//get the the total elapsed time in ms. negative value returned on abort condition
	float elapsedMsec;
	cutilSafeCall(cudaEventElapsedTime(&elapsedMsec, errorTrial_start, errorTrial_stop));

	cutilSafeCall(cudaEventDestroy(errorTrial_start));
	cutilSafeCall(cudaEventDestroy(errorTrial_stop));

	if (elapsedMsecDEV < 0.0f) {
			printf ("Encountered an error in the errorTrial test. Aborting\n");
			cutilDeviceReset();
			exit(EXIT_FAILURE);
	} else
		printf("\nError trial completed successfully.\n");

	if (!opt_quiet) {
		printf("\nTiming:  To test M%d"
				"\n  elapsed time :      %10.f msec = %.1f sec"
				"\n  dev. elapsed time:  %10.f msec = %.1f sec"
				"\n  est. total time:    %10.f msec = %.1f sec\n",
				testPrime,
				elapsedMsec, elapsedMsec/1000,
				elapsedMsecDEV*trialFraction, elapsedMsecDEV*trialFraction/1000,
				elapsedMsecDEV*testPrime, elapsedMsecDEV*testPrime/1000);

		printf("\nBeginning full test of M%d\n\n", testPrime);
	} else
		printf("\n  est. total time:\t%10.f msec = %.1f sec\n\n",
				elapsedMsecDEV*testPrime, elapsedMsecDEV*testPrime/1000);

	if (resuming)
		printf("\nResuming full test of M%d at iteration %d (%2.1f%%)\n\n", testPrime, resume_iter, 100.0f * (float)resume_iter / (float)testPrime);
	cutilSafeCall(cudaEventCreate(&mersenneTest_start));
	cutilSafeCall(cudaEventCreate(&mersenneTest_stop));
	cutilSafeCall(cudaEventRecord(mersenneTest_start, 0));


	// Define this outside the test so we have somewhere to copy resume data to, and free up host memory
	Real *d_signal;
	cutilSafeCall(cudaMalloc((void**)&d_signal, sizeof(Real) * signalSize));

	if (resuming) {
		cutilSafeCall(cudaMemcpy(d_signal, resume_signal, sizeof(Real) * signalSize, cudaMemcpyHostToDevice));
		free(resume_signal);
	}

	mersenneTest(testPrime, signalSize, d_signal, resume_iter);

	cutilSafeCall(cudaFree(d_signal));

	//get the the total elapsed time in ms
	cutilSafeCall(cudaEventRecord(mersenneTest_stop, 0));
	cutilSafeCall(cudaEventSynchronize(mersenneTest_stop));

    cutilSafeCall(cudaEventElapsedTime(&elapsedMsec, mersenneTest_start, mersenneTest_stop));
	
	cutilSafeCall(cudaEventDestroy(mersenneTest_start));
	cutilSafeCall(cudaEventDestroy(mersenneTest_stop));

	// Remove checkpoint files	
	(void) unlink (checkpoint_file);
	(void) unlink (checkpoint_backup);

	printf("\nTotal elapsed time:\t%10.f msec = %.1f sec\n",
		   elapsedMsec, elapsedMsec/1000);

	cutilDeviceReset();
	exit(EXIT_SUCCESS);
}

/**
 * HERE BEGINS THE HOST AND KERNEL CODE TO SUPPORT THE APPLICATION
 *   NOTE:  some changed, moved to IrrBaseBalanced11.cu
 */

// Complex pointwise multiplication
static __global__ void ComplexPointwiseSqr(Complex* cval, int size) {
	Complex c, temp;
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < size) {
		temp = cval[tid];
		c.y = 2.0*temp.x*temp.y;
		//c.x = (temp.x + temp.y)*(temp.x - temp.y);  xxAT ??
		c.x = temp.x*temp.x - temp.y*temp.y;
		cval[tid] = c;
	}
} 

/**
 * compute A and Ainv in extended precision, cast to doubles
 *   and load them to the host arrays
 * Uses dd_real 128-bit double-doubles to avoid catastropic cancellation errors
 *   for non-power-of-two FFT lengths
 */
static __host__ void computeWeightVectors(double *host_A, double *host_Ainv, int testPrime, int size) {

	dd_real dd_A, dd_Ainv;
	dd_real dd_N = dd_real(size);
	dd_real dd_2 = dd_real(2.0);

	for (int ddex = 0; ddex < size; ddex++) {
		dd_real dd_expo = dd_real(ddex)*dd_real(testPrime)/dd_N;
		dd_A = pow(dd_2, ceil(dd_expo) - dd_expo);
		dd_Ainv = (1.0 / dd_A) / dd_N;
		host_A[ddex] = to_double(dd_A);
		host_Ainv[ddex] = to_double(dd_Ainv);
	}
}

static __host__ void computeBitsPerWord(int testPrime, int *bitsPerWord, int size) {

	double PoverN = testPrime/(double)size;
	for (int j = 1; j <= size; j++) {	
		bitsPerWord[j - 1] = (int) (ceil(PoverN*j) - ceil(PoverN*(j - 1)));
	}
}

/**
 * do modular wrap-around to get successive words from element [size - 1]
 * Works backwards to get preceeding bits
 */
static __host__ void computeBitsPerWordVectors(unsigned char *bitsPerWord8, int *bitsPerWord, int size) {

	for (int i = 0; i < size; i++) {
		bitsPerWord8[i] = 0;

		for (int bit = 0; bit < 8; bit++) {
			short bitval;
			if (i - bit < 0)
				bitval = (bitsPerWord[size + i - bit] == h_LO_BITS ? 0 : 1);
			else
				bitval = (bitsPerWord[       i - bit] == h_LO_BITS ? 0 : 1);
			bitsPerWord8[i] |= bitval << bit;
		}
	}	
}

// load values of int array into double array for FFT.  Low-order 2 bytes go in lowest numbered
//     position in dArr
static __global__ void loadValue4ToFFTarray(double *dArr, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid == 0)
		dArr[tid] = 4.0;
	else
		dArr[tid] = 0.0;
}


// This includes pseudobalance by adding hi order terms from last rebalancing.
static __global__ void loadIntToDoubleIBDWT(double *dArr, int *iArr, int *iHiArr, double *aArr, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	int ival = iArr[tid];
	ival += (tid == 0 ? iHiArr[size - 1] : iHiArr[tid - 1]);

	dArr[tid] = ival*aArr[tid];
}

/**
 * We assume the a_inv also includes the 1/SIGNAL_SIZE scaling needed by the DFT
 * We also do the subtract 2 from the Lucas-square, requiring no carry in the
 *   current balanced carry-save signal.
 */
// Error version assigns the round-off error back to errorvals[tid]
template <int error_flag>
static __global__ void invDWTproductMinus2(int64_t *llintArr, double *signal, double *a_inv, float *errorvals, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	double sig;
	if (tid == 0)
		sig = signal[tid]*a_inv[tid] - 2.0;
	else
		sig = signal[tid]*a_inv[tid];

	llintArr[tid] = double2ll(sig, cudaRoundNearest);
	if (error_flag) {
		errorvals[tid] = (float) fabs(sig - llrint(sig));
	}
}

/**
 * uses Xfer to host and then sequential max check on array from errorVector computed above
 *   called seldom (currently, every 1/50 of the total iterations), so no effect on runtime.
 */
static __host__ float findMaxErrorHOST(float *dev_fltArr, float *host_temp, int len) {

	cudaMemcpy(host_temp, dev_fltArr, sizeof(float)*len, cudaMemcpyDeviceToHost);
	float maxVal = 0.0f;
	for (int i = 0; i < len; i++)
		if (host_temp[i] > maxVal)
			maxVal = host_temp[i];
	return maxVal;
}

/**
 *computeMaxVector()
 *function returns list of number of significant bits of a list of int64_ts
 *AS IS, list can only be as long as however many strings you can launch, now 67,107,840 on 2.0 gpus
 */
static __global__ void computeMaxBitVector(float *dev_errArr, int64_t *llint_signal, int len){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < len){
		if (llint_signal[tid] >= 0){
			dev_errArr[tid] = (float) __clzll(llint_signal[tid]);
		}
		else{
			dev_errArr[tid] = (float) __clzll(llint_signal[tid]*-1);
		}
	}
}

/**
 * findSignalSize()
 * Determines the best signalSize to use for a given testPrime
 * Choice based on runtime and error
 */
unsigned int findSignalSize(unsigned int testPrime) {
	int optimal_length = 0;
	float bestTime = 99999;
	uint64_t signalSize; // need to be bigger than necessary so some of the FFTlen combinations don't overflow
	// Only use lengths that are between 1/15th and 1/20th the testPrime
	// and round it up to the nearest T_PER_B size, since we're only testing multiples of T_PER_B
	int max_nx = ((testPrime / 17 / T_PER_B) + 1) * T_PER_B;
	int min_nx = ((testPrime / 20 / T_PER_B) + 1) * T_PER_B;

	if (!opt_quiet)
		printf("Testing FFT lengths between %d and %d\n\n", min_nx, max_nx);

	int retry = 0;
restart_findSignalSize:
	cudaEvent_t start_findSignalSize, stop_findSignalSize;
	cutilSafeCall(cudaEventCreate(&start_findSignalSize));
	cutilSafeCall(cudaEventCreate(&stop_findSignalSize));
	float elapsedTime, maxerr;

	// Need to run enough iterations to build-up the error, if there is any
	// This seems to be around 40-45 iterations in practice.
	int testIterations = 50;

	for (int two = 0; two <= MAX_2; two++) {
		for (int three = 0; three <= MAX_3; three++) {
			for (int five = 0; five <= MAX_5; five++) {
				for (int seven = 0; seven <= MAX_7; seven++) {
					signalSize = (powl(2,two) * powl(3,three) * powl(5,five) * powl(7,seven));
					if ((signalSize < (unsigned int)max_nx) & (signalSize > (unsigned int)min_nx) & (signalSize % T_PER_B == 0)) {
						maxerr = 0;
						int numBlocks = signalSize/T_PER_B;
						setSliceAndDice(testPrime, signalSize);
						initConstantSymbols(testPrime, signalSize);
 
 						// Store computed bit values and bases for precomputation of masks for the 
						int *h_bitsPerWord = (int *) malloc(sizeof(int)*signalSize);
						unsigned char *h_bitsPerWord8 = (unsigned char *) malloc(sizeof(unsigned char)*signalSize);
						
						// Allocate device memory for signal
						int *i_signalOUT;
						Real *d_signal;
						Complex *z_signal;
						int i_sizeOUT = sizeof(int)*signalSize;
						int d_size = sizeof(Real)*signalSize;
						int z_size = sizeof(Complex)*(signalSize/2 + 1);
						int bpw_size = sizeof(unsigned char)*signalSize;
						int llintSignalSize = sizeof(int64_t)*signalSize;
						Real *dev_A, *dev_Ainv;
						unsigned char *bitsPerWord8;
						int64_t *llint_signal;
						cutilSafeCall(cudaMalloc((void**)&i_signalOUT, i_sizeOUT));
						cutilSafeCall(cudaMalloc((void**)&d_signal, d_size));
						cutilSafeCall(cudaMalloc((void**)&z_signal, z_size));
						cutilSafeCall(cudaMalloc((void**)&dev_A, d_size));
						cutilSafeCall(cudaMalloc((void**)&dev_Ainv, d_size));
						cutilSafeCall(cudaMalloc((void**)&bitsPerWord8, bpw_size));
						cutilSafeCall(cudaMalloc((void**)&llint_signal, llintSignalSize));
						
						cufftHandle plan1, plan2;
						cufftSafeCall(cufftPlan1d(&plan1, signalSize, CUFFT_TYPEFORWARD, 1));
						cufftSafeCall(cufftPlan1d(&plan2, signalSize, CUFFT_TYPEINVERSE, 1));
						
						// Variables for the GPK carry-adder
						// Array for high-bit carry out
						int *i_hiBitArr;
						cutilSafeCall(cudaMalloc((void**)&i_hiBitArr, sizeof(int)*signalSize));
						
						//make host and device arrays for error computation
						float *dev_errArr;
						cudaMalloc((void**) &dev_errArr, signalSize*sizeof(float));
						float *host_errArr = (float *) malloc(signalSize*sizeof(float));
						
						// Compute word-sizes to use when dicing products to sum to int array
						computeBitsPerWord(testPrime, h_bitsPerWord, signalSize);
						computeBitsPerWordVectors(h_bitsPerWord8, h_bitsPerWord, signalSize);
						cutilSafeCall(cudaMemcpy(bitsPerWord8, h_bitsPerWord8, bpw_size, cudaMemcpyHostToDevice));

						double *h_A = (double *) malloc(signalSize*sizeof(double));
						double *h_Ainv = (double *) malloc(signalSize*sizeof(double));
						
						// compute weights in extended precision, essential for non-power-of-two signal_size
						computeWeightVectors(h_A, h_Ainv, testPrime, signalSize);
						cutilSafeCall(cudaMemcpy(dev_A, h_A, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
						cutilSafeCall(cudaMemcpy(dev_Ainv, h_Ainv, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
						
						// load the int array to the doubles for FFT
						// This is already balanced, and already multiplied by a_0 = 1 for DWT
						loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal, signalSize);
						cutilCheckMsg("Kernel execution failed [ loadValue4ToFFTarray ]");
						
						int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;
					
						int iter = 2;
						// start timer
						cutilSafeCall(cudaEventRecord(start_findSignalSize, 0));
						for (; iter < testIterations; iter++) {
							// Transform signal
							cufftSafeCall(CUFFT_EXECFORWARD(plan1, (Real *)d_signal, (Complex *)z_signal));
							cutilCheckMsg("Kernel execution failed [ CUFFT_EXECFORWARD ]");

							// Multiply the coefficients componentwise
							ComplexPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);
							cutilCheckMsg("Kernel execution failed [ ComplexPointwiseSqr ]");
							
							// Transform signal back
							cufftSafeCall(CUFFT_EXECINVERSE(plan2, (Complex *)z_signal, (Real *)d_signal));
							cutilCheckMsg("Kernel execution failed [ CUFFT_EXECINVERSE ]");

							// Calculate error
							invDWTproductMinus2<1><<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, dev_errArr, signalSize);
							cutilCheckMsg("Kernel execution failed [ ivnDWTproductMinus2ERROR ]");
							
							float this_err = findMaxErrorHOST(dev_errArr, host_errArr, signalSize);
							maxerr = std::max(this_err, maxerr);
							
							if (maxerr >= ERROR_LIMIT) // no need to continue this test
								break;

							sliceAndDice<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, llint_signal, bitsPerWord8, signalSize);
							cutilCheckMsg("Kernel execution failed [ sliceAndDice ]");
							
							loadIntToDoubleIBDWT<<<numBlocks, T_PER_B>>>(d_signal, i_signalOUT, i_hiBitArr, dev_A, signalSize);
							cutilCheckMsg("Kernel execution failed [ loadIntToDoubleIBDWT ]");
						}

						cutilSafeCall(cudaEventRecord(stop_findSignalSize, 0));
						cutilSafeCall(cudaEventSynchronize(stop_findSignalSize));
						cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start_findSignalSize, stop_findSignalSize));
						
						//Destroy CUFFT context
						cufftSafeCall(cufftDestroy(plan1));
						cufftSafeCall(cufftDestroy(plan2));
						
						// cleanup memory
						free(h_bitsPerWord);
						free(h_bitsPerWord8);
						free(h_A);
						free(h_Ainv);
						free(host_errArr);
						
						cutilSafeCall(cudaFree(i_signalOUT));
						cutilSafeCall(cudaFree(d_signal));
						cutilSafeCall(cudaFree(z_signal));
						cutilSafeCall(cudaFree(i_hiBitArr));
						cutilSafeCall(cudaFree(dev_A));
						cutilSafeCall(cudaFree(dev_Ainv));
						cutilSafeCall(cudaFree(bitsPerWord8));
						cutilSafeCall(cudaFree(llint_signal));
						cutilSafeCall(cudaFree(dev_errArr));

						if (!opt_quiet)
							printf("Testing signalSize %d, time: %3.2fms, error: %1.4f", (int)signalSize, elapsedTime/iter, maxerr);

						if (maxerr < ERROR_LIMIT) {
							if (elapsedTime < bestTime) {
								// we have a new best signalSize
								optimal_length = signalSize;
								bestTime = elapsedTime;
							}
						} else if (!opt_quiet) 
								printf(" (skipping: error too high)");
						if (!opt_quiet) {
							printf("\n");
							fflush(stdout);
						}
					}
				}
			}
		}
	}

	// If we haven't found a usable length, increase the search range slightly.
	// Should only happen on smaller testPrimes, so won't increase run-time on larger testPrimes
	if (optimal_length == 0) {
		if (!opt_quiet)
			printf("Could not find a signalSize; Increasing search range\n");
		min_nx = max_nx;
		max_nx = max_nx + 4*T_PER_B;
		retry++;
		if (retry > 100) { // Should this provide a wide enough range before bailing out?
			fprintf(stderr, "Could not find a suitable signalSize, exiting\n");
			exit (-1);
		}
		goto restart_findSignalSize;
	}
		
	cutilSafeCall(cudaEventDestroy(start_findSignalSize));
	cutilSafeCall(cudaEventDestroy(stop_findSignalSize));
	
	return optimal_length;
}

/**
* errorTrial()
*/
float errorTrial(unsigned int testIterations, unsigned int testPrime, unsigned int signalSize) {

	// We assume throughout that signalSize is divisible by T_PER_B
	const int numBlocks = signalSize/T_PER_B;

	// Run at least 50 testIterations. We'll encounter a floating point error later on if we don't
	// main() calls us with at least 100 iterations already, so this shouldn't happen
	if (testIterations<50)
		testIterations = 50;

	// Allocate host memory to return signal as necessary
	int *h_signalOUT = (int *) malloc(sizeof(int)*signalSize);
 
	// Store computed bit values and bases for precomputation of 
	//    masks for the 
	int *h_bases = (int *) malloc(sizeof(int)*signalSize);
	int *h_bitsPerWord = (int *) malloc(sizeof(int)*signalSize);
	unsigned char *h_bitsPerWord8 = (unsigned char *) malloc(sizeof(unsigned char)*signalSize);

	// Allocate device memory for signal
	int *i_signalOUT;
	Real *d_signal;
	Complex *z_signal;
	int i_sizeOUT = sizeof(int)*signalSize;
	int d_size = sizeof(Real)*signalSize;
	int z_size = sizeof(Complex)*(signalSize/2 + 1);
	int bpw_size = sizeof(unsigned char)*signalSize;

	int llintSignalSize = sizeof(int64_t)*signalSize;

	Real *dev_A, *dev_Ainv;
	unsigned char *bitsPerWord8;
	int64_t *llint_signal;
	cutilSafeCall(cudaMalloc((void**)&i_signalOUT, i_sizeOUT));
	cutilSafeCall(cudaMalloc((void**)&d_signal, d_size));
	cutilSafeCall(cudaMalloc((void**)&z_signal, z_size));

	cutilSafeCall(cudaMalloc((void**)&dev_A, d_size));
	cutilSafeCall(cudaMalloc((void**)&dev_Ainv, d_size));
	cutilSafeCall(cudaMalloc((void**)&bitsPerWord8, bpw_size));
	cutilSafeCall(cudaMalloc((void**)&llint_signal, llintSignalSize));

	// allocate device memory for DWT weights and base values
	// CUFFT plan
	cufftHandle plan1, plan2;
	cufftSafeCall(cufftPlan1d(&plan1, signalSize, CUFFT_TYPEFORWARD, 1));
	cufftSafeCall(cufftPlan1d(&plan2, signalSize, CUFFT_TYPEINVERSE, 1));

	// Variables for the GPK carry-adder
	// Array for high-bit carry out
	int *i_hiBitArr;
	cutilSafeCall(cudaMalloc((void**)&i_hiBitArr, sizeof(int)*signalSize));

	// CUDPP plan for parallel-scan int GPK adds
	
	//make host and device arrays for error computation
	float *dev_errArr;
	cudaMalloc((void**) &dev_errArr, signalSize*sizeof(float));
	float *host_errArr = (float *) malloc(signalSize*sizeof(float));

	// Compute word-sizes to use when dicing products to sum to int array
	computeBitsPerWord(testPrime, h_bitsPerWord, signalSize);
	computeBitsPerWordVectors(h_bitsPerWord8, h_bitsPerWord, signalSize);
	cutilSafeCall(cudaMemcpy(bitsPerWord8, h_bitsPerWord8, bpw_size, cudaMemcpyHostToDevice));

	if (opt_verbose) {
		for (int i = 0; i < 20; i++) {
			printf("word[%d] numbits = %d\n", i, h_bitsPerWord[i]);
			printf("numbits of this and following 7 are: ");
			for (int bit = 1; bit < 256; bit *= 2)
				printf("%d ", bit & h_bitsPerWord8[i] ? h_HI_BITS : h_LO_BITS);
			printf("\n");
		}
		for (unsigned int i = signalSize - 8; i < signalSize; i++) {
			printf("word[%d] numbits = %d\n", i, h_bitsPerWord[i]);
			printf("numbits of this and following 7 are: ");
			for (int bit = 1; bit < 256; bit *= 2)
				printf("%d ", bit & h_bitsPerWord8[i] ? h_HI_BITS : h_LO_BITS);
			printf("\n");
		}
	}
	
	double *h_A = (double *) malloc(signalSize*sizeof(double));
	double *h_Ainv = (double *) malloc(signalSize*sizeof(double));

	// compute weights in extended precision, essential for non-power-of-two signal_size
	computeWeightVectors(h_A, h_Ainv, testPrime, signalSize);
	cutilSafeCall(cudaMemcpy(dev_A, h_A, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dev_Ainv, h_Ainv, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
	if (opt_verbose) {
		printf("weight vector looks like:\n");
		for (int i = 0; i < 20; i++) 
			printf("a[%d] = %f\n", i, h_A[i]);
		for (int i = 0; i < 20; i++) 
			printf("ainv[%d] = %f\n", i, h_Ainv[i]);
	}

	// load the int array to the doubles for FFT
	// This is already balanced, and already multiplied by a_0 = 1 for DWT
	loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal, signalSize);
	cutilCheckMsg("Kernel execution failed [ loadValue4ToFFTarray ]");

	float totalTime = 0;
	// Loop M-2 times
    for (unsigned int iter = 2; iter < testIterations; iter++) {
		if (iter % (testIterations/50) == 0) {

			cudaEvent_t start, stop;
			cutilSafeCall(cudaEventCreate(&start));
			cutilSafeCall(cudaEventCreate(&stop));
			cutilSafeCall(cudaEventRecord(start, 0));

			// Transform signal
			cufftSafeCall(CUFFT_EXECFORWARD(plan1, (Real *)d_signal, (Complex *)z_signal));
			cutilCheckMsg("Kernel execution failed [ CUFFT_EXECFORWARD ]");
			// Multiply the coefficients componentwise
			int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

			ComplexPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);

			cutilCheckMsg("Kernel execution failed [ ComplexPointwiseSqr ]");

			// Transform signal back
			cufftSafeCall(CUFFT_EXECINVERSE(plan2, (Complex *)z_signal, (Real *)d_signal));
			cutilCheckMsg("Kernel execution failed [ CUFFT_EXECINVERSE ]");

			cutilSafeCall(cudaEventRecord(stop, 0));
			cutilSafeCall(cudaEventSynchronize(stop));
			float elapsedTime;
			cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
			if (opt_verbose)
					printf("Time for FFT, squaring, INV FFT:  %3.3f ms\n", elapsedTime);
			totalTime += elapsedTime;
			cutilSafeCall(cudaEventDestroy(start));
			cutilSafeCall(cudaEventDestroy(stop));

			// ERROR TESTS
			invDWTproductMinus2<1><<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, dev_errArr, signalSize);
			cutilCheckMsg("Kernel execution failed [ ivnDWTproductMinus2ERROR ]");

			float maxerr = findMaxErrorHOST(dev_errArr, host_errArr, signalSize);
			if (maxerr >= 0.5f) {
					if (opt_verbose)
						printf("max abs error = %f, is too high. Exiting\n", maxerr);
					totalTime = -1;
					break;
			} else if (opt_verbose)
					printf("\n[%d/50]: iteration %d: max abs error = %f", iter/(testPrime/50), iter, maxerr);

			computeMaxBitVector<<<numBlocks, T_PER_B>>>(dev_errArr, llint_signal, signalSize);
			cutilCheckMsg("Kernel execution failed [ computeMaxBitVector ]");
			float maxBitVector = findMaxErrorHOST(dev_errArr, host_errArr, signalSize);
			if (opt_verbose) {
					printf("\n[%d/50]: iteration %d: max Bit Vector = %f", iter/(testPrime/50), iter, maxBitVector);
					fflush(stdout);
			}

			// Time rebalancing
			cutilSafeCall(cudaEventCreate(&start));
			cutilSafeCall(cudaEventCreate(&stop));
			cutilSafeCall(cudaEventRecord(start, 0));
			sliceAndDice<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, llint_signal, bitsPerWord8, signalSize);
			cutilCheckMsg("Kernel execution failed [ sliceAndDice ]");
			cutilSafeCall(cudaEventRecord(stop, 0));
			cutilSafeCall(cudaEventSynchronize(stop));
			cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
			if (opt_verbose)
					printf("\nTime to rebalance llint:  %3.3f ms\n", elapsedTime);
			totalTime += elapsedTime;
			cutilSafeCall(cudaEventDestroy(start));
			cutilSafeCall(cudaEventDestroy(stop));
		}
		else {
			// Transform signal
			cufftSafeCall(CUFFT_EXECFORWARD(plan1, (Real *)d_signal, (Complex *)z_signal));
			cutilCheckMsg("Kernel execution failed [ CUFFT_EXECFORWARD ]");
			// Multiply the coefficients componentwise
			int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

			ComplexPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);
			cutilCheckMsg("Kernel execution failed [ ComplexPointwiseSqr ]");

			// Transform signal back
			cufftSafeCall(CUFFT_EXECINVERSE(plan2, (Complex *)z_signal, (Real *)d_signal));
			cutilCheckMsg("Kernel execution failed [ CUFFT_EXECINVERSE ]");

			invDWTproductMinus2<0><<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, dev_errArr, signalSize);
			cutilCheckMsg("Kernel execution failed [ invDWTproductMinus2 ]");

			sliceAndDice<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, llint_signal, bitsPerWord8, signalSize);
			cutilCheckMsg("Kernel execution failed [ sliceAndDice ]");
		}

		loadIntToDoubleIBDWT<<<numBlocks, T_PER_B>>>(d_signal, i_signalOUT, i_hiBitArr, dev_A, signalSize);
		cutilCheckMsg("Kernel execution failed [ loadIntToDoubleIBDWT ]");
	}
	
	// DONE!  Final copy out from GPU, since not done by default as for CPU stages
	// DO GOOD REBALANCE HERE
	cudaEvent_t start, stop;
	cutilSafeCall(cudaEventCreate(&start));
	cutilSafeCall(cudaEventCreate(&stop));
	cutilSafeCall(cudaEventRecord(start, 0));

	addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, signalSize);
	rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signalOUT, bitsPerWord8, signalSize);
	cutilSafeCall(cudaMemcpy(h_signalOUT, i_signalOUT, i_sizeOUT, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaEventRecord(stop, 0));
	cutilSafeCall(cudaEventSynchronize(stop));
	float elapsedTime;
	cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
	if (opt_verbose)
			printf("\nTime to rebalance and write-back:  %3.1f ms\n", elapsedTime);
	cutilSafeCall(cudaEventDestroy(start));
	cutilSafeCall(cudaEventDestroy(stop));

	//Destroy CUFFT context
	cufftSafeCall(cufftDestroy(plan1));
	cufftSafeCall(cufftDestroy(plan2));

	// cleanup memory
	free(h_signalOUT);
	free(h_bases);
	free(h_bitsPerWord);
	free(h_bitsPerWord8);
	free(h_A);
	free(h_Ainv);
	free(host_errArr);

	cutilSafeCall(cudaFree(i_signalOUT));
	cutilSafeCall(cudaFree(d_signal));
	cutilSafeCall(cudaFree(z_signal));

	cutilSafeCall(cudaFree(i_hiBitArr));
	cutilSafeCall(cudaFree(dev_A));
	cutilSafeCall(cudaFree(dev_Ainv));
	cutilSafeCall(cudaFree(bitsPerWord8));
	cutilSafeCall(cudaFree(llint_signal));

	cutilSafeCall(cudaFree(dev_errArr));
	
	return totalTime/50;
}

/**
* print_residue() -- output the Lucas-Lehmer residue for non-prime exponents
*   needed for result submission to GIMPS, or verifying results with other clients
*/
void print_residue(int testPrime, int *h_signalOUT, int signalSize) {
	static uint64_t *hex = NULL;
	static uint64_t prior_hex = 0;

	int64_t k, j=0, i, word, k1;
	double lo = floor((exp(floor((double)testPrime/signalSize)*log(2.0)))+0.5);
	double hi = lo+lo;
	uint64_t b = testPrime % signalSize; 
	uint64_t c = signalSize - b; 
	int totalbits = 64;
	
	int sudden_death = 0; 
	int64_t NminusOne = signalSize - 1; 

	while (1) {
			k = j;
			if (h_signalOUT[k] < 0.0) {
					k1 = (j + 1) % signalSize;
					--h_signalOUT[k1];
					if (j == 0 || (j != NminusOne && ((((b*j) % signalSize) >= c) || j == 0)))
							h_signalOUT[k] += hi;
					else
							h_signalOUT[k] += lo;
			} else if (sudden_death)
					break;
			if (++j == signalSize) {
					sudden_death = 1;
					j = 0;
			}
	}

	if (hex != NULL && totalbits/8 + 1 > prior_hex) {
			free(hex);
			hex = NULL;
			prior_hex = totalbits/8 + 1;
	}

	if (hex == NULL && (hex = (uint64_t *)calloc(totalbits/8 + 1, sizeof(uint64_t))) == NULL) {
			printf("Cannot get memory for residue bits; calloc()\n");
			exit(-1);
	}
	
	j = 0;
	i = 0;
	do {
			k = (int64_t)(ceil((double)testPrime*(j + 1)/signalSize) - ceil((double)testPrime*j/signalSize));
			if (k > totalbits)
					k = totalbits;
			totalbits -= k;
			word = (int64_t)h_signalOUT[j + ((j & 0) >> 0)];
			for (j++; k > 0; k--, i++) {
					if (i % 8 == 0)
							hex[i/8] = 0L;
					hex[i/8] |= ((word & 0x1) << (i % 8));
					word >>= 1;
			}
	} while(totalbits > 0);
	
	printf("0x");
	
	for (j = (i - 1)/8; j >= 0; j--) {
			printf("%02lx", hex[j]);
	}
	return;
}

/**
* mersenneTest() -- full test of 2^testPrime - 1, including max error term every 1/50th
*   time through loop
*/
void mersenneTest(unsigned int testPrime, unsigned int signalSize, Real *d_signal, unsigned int iter) {

	// We assume throughout that signalSize is divisible by T_PER_B
	const int numBlocks = signalSize/T_PER_B;
	const int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;
	
	// Allocate host memory to return signal as necessary
	int *h_signalOUT = (int *) malloc(sizeof(int)*signalSize);
	// For Checkpointing
	Real *cp_signalOUT = (Real *) malloc(sizeof(Real)*signalSize);

	// Store computed bit values and bases for precomputation of 
	//    masks for the 
	int *h_bases = (int *) malloc(sizeof(int)*signalSize);
	int *h_bitsPerWord = (int *) malloc(sizeof(int)*signalSize);
	unsigned char *h_bitsPerWord8 = (unsigned char *) malloc(sizeof(unsigned char)*signalSize);

	// Allocate device memory for signal
	int *i_signalOUT;
//	Real *d_signal;
	Complex *z_signal;
	int i_sizeOUT = sizeof(int)*signalSize;
	int d_size = sizeof(Real)*signalSize;
	int z_size = sizeof(Complex)*(signalSize/2 + 1);
	int bpw_size = sizeof(unsigned char)*signalSize;

	int llintSignalSize = sizeof(int64_t)*signalSize;

	Real *dev_A, *dev_Ainv;
	unsigned char *bitsPerWord8;
	int64_t *llint_signal;
	cutilSafeCall(cudaMalloc((void**)&i_signalOUT, i_sizeOUT));
//	cutilSafeCall(cudaMalloc((void**)&d_signal, d_size));
	cutilSafeCall(cudaMalloc((void**)&z_signal, z_size));

	cutilSafeCall(cudaMalloc((void**)&dev_A, d_size));
	cutilSafeCall(cudaMalloc((void**)&dev_Ainv, d_size));
	cutilSafeCall(cudaMalloc((void**)&bitsPerWord8, bpw_size));
	cutilSafeCall(cudaMalloc((void**)&llint_signal, llintSignalSize));

	// allocate device memory for DWT weights and base values
	// CUFFT plan
	cufftHandle plan1, plan2;
	cufftSafeCall(cufftPlan1d(&plan1, signalSize, CUFFT_TYPEFORWARD, 1));
	cufftSafeCall(cufftPlan1d(&plan2, signalSize, CUFFT_TYPEINVERSE, 1));

	// Array for high-bit carry out
	int *i_hiBitArr;
	cutilSafeCall(cudaMalloc((void**)&i_hiBitArr, sizeof(int)*signalSize));

	// Error-checking device and host arrays
	float *dev_errArr; 
	cudaMalloc((void**) &dev_errArr, signalSize*sizeof(float));
	float *host_errArr = (float *) malloc(signalSize*sizeof(float));

	// Compute word-sizes to use when dicing products to sum to int array
	computeBitsPerWord(testPrime, h_bitsPerWord, signalSize);
	computeBitsPerWordVectors(h_bitsPerWord8, h_bitsPerWord, signalSize);
	cutilSafeCall(cudaMemcpy(bitsPerWord8, h_bitsPerWord8, bpw_size, cudaMemcpyHostToDevice));

	// compute weights in extended precision, essential for non-power-of-two signal_size,
	//   and then load to device
	double *h_A = (double *) malloc(signalSize*sizeof(double));
	double *h_Ainv = (double *) malloc(signalSize*sizeof(double));
	computeWeightVectors(h_A, h_Ainv, testPrime, signalSize);
	cutilSafeCall(cudaMemcpy(dev_A, h_A, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dev_Ainv, h_Ainv, sizeof(double)*signalSize, cudaMemcpyHostToDevice));

	if (!resuming) {
		iter = 2;
		
		// load the int array to the doubles for FFT
		// This is already balanced, and already multiplied by a_0 = 1 for DWT
		loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal, signalSize);
		cutilCheckMsg("Kernel execution failed [ loadValue4ToFFTarray ]");
	}

	float maxerr = 0.0f;
	// Loop M-2 times
	for (; iter < testPrime; iter++) {
		// Transform signal
		cufftSafeCall(CUFFT_EXECFORWARD(plan1, (Real *)d_signal, (Complex *)z_signal));
		cutilCheckMsg("Kernel execution failed [ CUFFT_EXECFORWARD ]");

		// Multiply the coefficients componentwise
		ComplexPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);
		cutilCheckMsg("Kernel execution failed [ ComplexPointwiseSqr ]");

		// Transform signal back
		cufftSafeCall(CUFFT_EXECINVERSE(plan2, (Complex *)z_signal, (Real *)d_signal));
		cutilCheckMsg("Kernel execution failed [ CUFFT_EXECINVERSE ]");

		// Every so often, do some error checking. Do this at every checkpoint as well
		if ((iter % (testPrime/1000) == 0) | (iter % checkpoint_freq == 1)) {
			invDWTproductMinus2<1><<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, dev_errArr, signalSize);
			cutilCheckMsg("Kernel execution failed [ invDWTproductMinus2ERROR ]");

			maxerr = findMaxErrorHOST(dev_errArr, host_errArr, signalSize);
			// FIXME: -Aaron: we need to add a way to stop/restart if the error is too high.
		} else {
			invDWTproductMinus2<0><<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, dev_errArr, signalSize);
			cutilCheckMsg("Kernel execution failed [ invDWTproductMinus2 ]");
		}

		// REBALANCE llint TIMING
		sliceAndDice<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, llint_signal, bitsPerWord8, signalSize);
		cutilCheckMsg("Kernel execution failed [ sliceAndDice ]");

		loadIntToDoubleIBDWT<<<numBlocks, T_PER_B>>>(d_signal, i_signalOUT, i_hiBitArr, dev_A, signalSize);
		cutilCheckMsg("Kernel execution failed [ loadIntToDoubleIBDWT ]");

		// Checkpoint every checkpoint_freq iterations
		// To match CUDALucas output: CL counts iterations differently and thus displays residue at what we consider to be iteration+1
		if (iter % checkpoint_freq == 1) {
			cutilDeviceSynchronize();
			cutilSafeCall(cudaMemcpy(cp_signalOUT, d_signal, sizeof(Real) * signalSize, cudaMemcpyDeviceToHost));
			writeCheckpoint(cp_signalOUT, testPrime, signalSize, iter + 1);

			if (!opt_quiet) {
				// Display current iteration's residue and error
				addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, signalSize);
				rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signalOUT, bitsPerWord8, signalSize);
				cutilSafeCall(cudaMemcpy(h_signalOUT, i_signalOUT, i_sizeOUT, cudaMemcpyDeviceToHost));
				// Lie about the current iteration to match other programs
				printf("[%4.1f%%] Iteration %d: max err = %1.4f, ", 100.0f * (float)iter / (float)testPrime, iter - 1, maxerr);
				print_residue(testPrime, h_signalOUT, signalSize);
				printf("\n");
			}
		}
	}

	// DONE!  Final copy out from GPU, since not done by default as for CPU stages
	// DO GOOD REBALANCE HERE
	addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, signalSize);
	cutilCheckMsg("Kernel execution failed [ addPseudoBalancd ]");
	
	rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signalOUT, bitsPerWord8, signalSize);
	cutilCheckMsg("Kernel execution failed [ rebalanceIrrIntSEQGPU]");

	cutilSafeCall(cudaMemcpy(h_signalOUT, i_signalOUT, i_sizeOUT, cudaMemcpyDeviceToHost));

	bool nonZeros = false;
	for (unsigned int i = 0; i < signalSize; i++) {
		if (h_signalOUT[i] > 0) {
			nonZeros = true;
			break;
		}
	}
	if (nonZeros) {
		if (testPrime < 50000 & opt_verbose) {
			for (unsigned int i = 0; i < signalSize; i++) {
			//	unsigned char ch = h_signal[i] & 0xFF;

				printf("%05x", h_signalOUT[i]);
				if (i % 2 == 3)
					printf(" ");
				if (i % 20 == 39)
					printf("\n");
			}
			printf("\n");
		}
		if (opt_verbose)
			printf("\nM_%d tests as non-prime.\n\n", testPrime);

		printf("\nM( %d )C, ", testPrime);
		print_residue(testPrime, h_signalOUT, signalSize);
	} else {
		printf("\n\n\aPRIME FOUND: M_%d tests as prime.", testPrime);
		printf("\nM( %d )P", testPrime);

	}
	printf(", n = %d, %s v%s\n", signalSize, program_name, program_version);


	//Destroy CUFFT context
	cufftSafeCall(cufftDestroy(plan1));
	cufftSafeCall(cufftDestroy(plan2));

	// cleanup memory
	free(h_signalOUT);
	free(cp_signalOUT);
	free(h_bases);
	free(h_bitsPerWord);
	free(h_bitsPerWord8);
	free(h_A);
	free(h_Ainv);
	free(host_errArr);

	cutilSafeCall(cudaFree(i_signalOUT));
//	cutilSafeCall(cudaFree(d_signal));
	cutilSafeCall(cudaFree(z_signal));

	cutilSafeCall(cudaFree(i_hiBitArr));
	cutilSafeCall(cudaFree(dev_A));
	cutilSafeCall(cudaFree(dev_Ainv));
	cutilSafeCall(cudaFree(bitsPerWord8));
	cutilSafeCall(cudaFree(llint_signal));

	cutilSafeCall(cudaFree(dev_errArr));
}

Real *readCheckpoint(unsigned int testPrime, unsigned int *signalSize, unsigned int *resume_iter) {
	FILE *fPtr;
	unsigned int testPrime_r, signalSize_r, resumeiter_r; 
	Real *signal;

	char program_name_r[16];
	char program_vers_r[16];

	if (opt_verbose)
		fprintf(stderr, "Attempting to resume from checkpoint file: %s ...", checkpoint_file);
	fPtr = fopen (checkpoint_file, "rb");
	if (!fPtr) {
		if (opt_verbose)
			fprintf(stderr, "failed.\nAttempting to resume from backup checkpoint file: %s ...", checkpoint_backup);
		// First checkpoint failed. Try backup
		fPtr = fopen (checkpoint_backup, "rb");
		if (!fPtr) {
			// Both failed, give up
			if (opt_verbose)
				fprintf(stderr, "failed.\n\nUnable to load a checkpoint file, starting from the beginning\n");
			return NULL;
		}
	}
	
	if (opt_verbose)
		fprintf(stderr,"good!.\n");

	if (fread(&program_name_r, 1, sizeof(program_name_r), fPtr) != sizeof(program_name_r) ||
		fread(&program_vers_r, 1, sizeof(program_vers_r), fPtr) != sizeof(program_vers_r)) {
		fclose (fPtr);
		return NULL;
	}

	if  (strcmp(program_name_r, program_name) != 0) {
		fprintf(stderr, "Checkpoint was created with a different application, not using checkpoint\n\n");
		return NULL;
	}

	if (strcmp(program_vers_r, program_version) != 0) {
		fprintf(stderr, "Checkpoint was created with a different version of %s, attempting to continue\n\n", program_name);
	}

	// check parameters
	if (fread (&testPrime_r,  1, sizeof (testPrime_r),  fPtr) != sizeof (testPrime_r)  ||
		fread (&signalSize_r, 1, sizeof (signalSize_r), fPtr) != sizeof (signalSize_r) ||
		fread (&resumeiter_r, 1, sizeof (resumeiter_r), fPtr) != sizeof (resumeiter_r)) {
			fprintf (stderr, "\nThe checkpoint doesn't match current test.  Current test will be restarted\n");
			fclose (fPtr);
			return NULL;
	}
	
	if (testPrime != testPrime_r) { 
		fprintf (stderr, "\nThe checkpoint doesn't match current test.  Current test will be restarted\n");
		fclose (fPtr);
		return NULL;
	}
	
	// check for successful read of z, delayed until here since zSize can vary
	signal = (Real *) malloc (sizeof (Real) * (signalSize_r));
	if (fread (signal, 1, sizeof (Real) * (signalSize_r), fPtr) != (sizeof (Real) * (signalSize_r))) {
		fprintf (stderr, "\nThe checkpoint doesn't match current test.  Current test will be restarted\n");
		fclose (fPtr);
		free (signal);
		return NULL;
	}

	// We have a good checkpoint. Return it
	*resume_iter = resumeiter_r;
	*signalSize = signalSize_r;
	return signal;
}

void writeCheckpoint (Real *signal, unsigned int testPrime, unsigned int signalSize, unsigned int iter) {
//	return;
	FILE *fPtr;

	(void) unlink (checkpoint_backup);
	(void) rename (checkpoint_file, checkpoint_backup);
	
	fPtr = fopen (checkpoint_file, "wb");
	if (!fPtr)
		return;

	char buf[16];

	memset(buf, 0, sizeof(buf));
	sprintf(buf, "%s", program_name);
	fwrite (&buf            , 1, sizeof (buf)              , fPtr);
	memset(buf, 0, sizeof(buf));
	sprintf(buf, "%s", program_version);
	fwrite (&buf            , 1, sizeof (buf)              , fPtr);
	fwrite (&testPrime      , 1, sizeof (testPrime)        , fPtr);
	fwrite (&signalSize     , 1, sizeof (signalSize)       , fPtr);
	fwrite (&iter           , 1, sizeof (iter)             , fPtr);
	fwrite (signal          , 1, sizeof (Real) * signalSize, fPtr);
	fclose (fPtr);
}
