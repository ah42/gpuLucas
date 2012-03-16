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
#include <signal.h>

// most compilers do not have support for __float128 and needs an external library to
// support extended precision
#include <qd/dd_real.h>

// includes, project
#include <cufft.h>
#include "cuda_safecalls.h"

// NOTE: testPrimes below 9689 generate runlengths < 1024, which breaks the code if T_PER_B = 1024
// Create ThreadsPerBlock constant
const int T_PER_B = 256;

// These determines how many elements we compute per thread
// It changes the block dimensions from T_PER_B to T_PER_B/UNROLL_KERNEL
#define UNROLL_KERNEL 2

// These determine the highest FFT signalSize we will check in findSignalSize()
// where signalSize == 2^MAX_2 * 3^MAX_3 * 5^MAX_5 * 7^MAX_7
// signalSize also must be divisible by T_PER_B, so every transform will have a power-of-2 component
#define MAX_2 24 // 16777216
#define MAX_3 5  // 243 * T_PER_B
#define MAX_5 3  // 125 * T_PER_B
#define MAX_7 2  //  49 * T_PER_B

// This determines the maximum allowable roundoff error
#define ERROR_LIMIT 0.30f

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
char program_version[] = "0.9.3";
int iter = 0;
int signalSize = 0, testPrime = 0;
int numBlocks = 0, numFFTblocks = 0;
int in_place = 0;

volatile int do_gracefulExit = 0;

int resuming = 0;
// checkpoint filename buffers
char checkpoint_file[32];
char checkpoint_backup[32];
// Default checkpoint interval in iterations:
// 50,000 or higher eliminates most of the overhead cost in rebalanceIrrIntSEQGPU
int checkpoint_freq = 50000;

__constant__ int LO_BITS;
__constant__ int HI_BITS;
__constant__ int BASE_LO;
__constant__ int BASE_HI;
__constant__ int LO_MODMASK;
__constant__ int HI_MODMASK;
__constant__ int DEV_SIGNALSIZE;

// Need this include after T_PER_B so can use as shared memory array-length
//   in IrrBaseBalanced.cu routines to avoid dynamic memory alloc on GPU
//   (xxAT sloppy, but okay for now) (means need to recompile for different T_PER_B but
//         have removed NUMBLOCKS dependency, so can do runs of different lengths
// Also needs LO_BITS, etc., constant declarations for templated routines
// This includes all code for parallel carry-add of the balanced-variable base integers
#include "IrrBaseBalanced.cu"

static __host__ void initConstantSymbols() {

	h_LO_BITS = testPrime/signalSize;
	h_HI_BITS = testPrime/signalSize + 1;
	h_BASE_LO = 1 << h_LO_BITS;
	h_BASE_HI = 1 << h_HI_BITS;
	h_LO_MODMASK = h_BASE_LO - 1;
	h_HI_MODMASK = h_BASE_HI - 1;
	cutilSafeCall(cudaMemcpyToSymbol(DEV_SIGNALSIZE, &signalSize, sizeof(int)));
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

// Global pointers
static int     *i_signal, *h_signalOUT, *h_bitsPerWord;
static int8_t  *i_hiBitArr;
static Real    *d_signal, *dev_A, *dev_Ainv, *cp_signal;
static Complex *z_signal;
static uint8_t *bitsPerWord8, *h_bitsPerWord8;
static int64_t *llint_signal;
static double  *h_A, *h_Ainv;
static float   *dev_errArr, *host_errArr;

static cufftHandle plan1, plan2;

/**
 * PREDECLARED FUNCTIONS:  these don't really need to be predeclared anymore,
 *   but give an overview of the functions so left it.
 */

static __global__ void ComplexPointwiseSqr(Complex* z_signal, const int size);
static __global__ void loadValue4ToFFTarray(double *d_signal);
static __global__ void loadIntToDoubleIBDWT(double *d_signal, const int *i_signal, const int8_t *i_hiBitArr, const double *dev_A);

/*
 * In bitsPerWord, we use a bit-vector:
 *    0 -- low base word
 *    1 -- high base word
 * Where the positions 0=current, 1=next, 2=nextnext, etc.
 *    The BASE_HI, BASE_LO, HI_BITS, LO_BITS are global constants.
 */
static __host__ void computeBitsPerWord(int *bitsPerWord);
static __host__ void computeBitsPerWordVectors(uint8_t *bitsPerWord8, int *bitsPerWord);

/**
 * code for convolution error-checking
 */
static __global__ void computeMaxBitVector(float *dev_errArr, int64_t *llint_signal);
static __host__ float findMaxErrorHOST(float *dev_errArr, float *host_errArr);

/**
 * compute A and Ainv in extended precision, cast to doubles
 *   and load them to the host arrays.  We include the FFT 1/N scaling with
 *   host_ainv and pull it out of the pointwiseSqrAndScale code
 */
static __host__ void computeWeightVectors(double *host_A, double *host_Ainv);

/**
 * This completes the invDWT transform by multiplying the signal by a_inv,
 *   and subtracts 2 from signal[0], requiring no carry in current weighted carry-save state
 */
template <int error>
static __global__ void invDWTproductMinus2(int64_t *llint_signal, const double *d_signal, const double *dev_Ainv, float *dev_errArr);


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
void (*sliceAndDice)(int *i_signal, int8_t *i_hiBitArr, const int64_t *llint_signal, const uint8_t *bitsPerWord8);

/**
 * For n = 2 to 6. This uses templated kernel functions for the different lengths,
 *   as defined in IrrBaseBalanced.cu file.  (Thanks, Alex.)
 *   These seem to be good divisions for the sliceAndDice but might need to be adjusted.
 * Auto-selected signalSize will almost always choose cases 17 through 19.
 */
  
void setSliceAndDice() {

	int ratio = testPrime / signalSize;

	if (ratio >= 21) {
		fprintf(stderr, "testPrime (%d)/ signalSize (%d) out of range: %d\n",
				testPrime, signalSize, (int)(testPrime / signalSize));
		exit(-1);
	}

	if (ratio >= 18)
		sliceAndDice = llintToIrrBal<2>;
	else if (ratio >= 16)
		sliceAndDice = llintToIrrBal<3>;
	else if (ratio >= 14)
		sliceAndDice = llintToIrrBal<4>;
	else if (ratio >= 12)
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
static __host__ float errorTrial(int testIterations);
static __host__ int findSignalSize();
static __host__ void mersenneTest(Real *d_signal);

static __host__ void writeCheckpoint(Real *signal);
static __host__ Real *readCheckpoint(int *signalSize, int *resume_iter);

static __host__ void printFriendlyTime(char *buf, int time);

static __host__ int mallocArrays();
static __host__ void freeArrays();

static __host__ void gracefulExit(int sig) {
	printf("[%4.1f] Iter %d: Received signal %d, performing graceful exit\n", 
			100 * (float)iter/(float)testPrime, iter - 1, sig);
	do_gracefulExit = 1;
}

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
			if (atol(argv[index]) > 2147483647) {
				fprintf(stderr, "testPrime too large, aborting\n");
				exit(-1);
			}
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
			exit(-1);
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
		if (!opt_quiet) {
			fprintf(stderr, "\tDevice %d: %s\n", dev, deviceProp.name);
			fprintf(stderr, "\t\tCompute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
			fprintf(stderr, "\t\tTotal Global RAM:   %dMiB\n", (int)deviceProp.totalGlobalMem/1024/1024);
		}
	}

	// Chosen device = use_device
	cudaGetDeviceProperties(&deviceProp, use_device);

	// Make sure this is a device that we can use
	if ((deviceProp.major < 2) & (deviceProp.minor < 3)) {
		fprintf(stderr, "\nError: unable to use device %d.\n\n"
				"A card with a minimum compute-capability of 1.3 is required for double precision computations\n",
				use_device);
		exit(-1);
	}

	fprintf(stderr, "Using device %d: %s\n\n", use_device, deviceProp.name);
	cudaSetDevice(use_device);

	int resume_iter = 0;

	if ((cp_signal = readCheckpoint(&signalSize, &resume_iter)) != NULL)
		resuming = 1;

	if (signalSize == 0)  {
		signalSize = findSignalSize();
		printf("Optimal signalSize detected: %d\n\n", signalSize);
	} else
		printf("Using specified FFT runlength %d\n\n", signalSize);

	// BEGIN by initializing constant memory on device
	initConstantSymbols();

	// Based on the problem size, and runlength, set the number of carry digits
	//   and assign the global slice-and-dice function from the templated
	//   llintToBalInt<n>() function
	setSliceAndDice();


	if ((int)sizeof(int64_t) != 8) {
		printf("size of int64_t = %d (if not 8, you're in trouble)\n", (int) sizeof(int64_t));
		exit(-1);
	}

	if (!opt_quiet) {
		printf("Testing M%d, using an irrational base with wordlengths (%d, %d),\n"
				"\tusing an FFT runlength of 2^%f = %d\n",
				testPrime, h_LO_BITS, h_HI_BITS, log(1.0*signalSize)/log(2.0), signalSize);
		printf("\n\tNUM_BLOCKS = %d, T_PER_B = %d\n", signalSize/T_PER_B, T_PER_B);
	}

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

	// START timer now
	cudaEvent_t start, stop;
	cutilSafeCall(cudaEventCreate(&start));
	cutilSafeCall(cudaEventCreate(&stop));
	cutilSafeCall(cudaEventRecord(start, 0));

	// errorTrial() called to give an estimate of convolution sizes and errors,
	// as well as FFT timings and rebalancing time.
	// Return value is average time per Lucas-Lehmer iteration based on GPU timings
	float elapsedMsecDEV = errorTrial(testIterations);

	// stop the timer
	cutilSafeCall(cudaEventRecord(stop, 0));
	cutilSafeCall(cudaEventSynchronize(stop));

	//get the the total elapsed time in ms. negative value returned on abort condition
	float elapsedMsec;
	cutilSafeCall(cudaEventElapsedTime(&elapsedMsec, start, stop));

	if (elapsedMsecDEV < 0.0f) {
			printf ("Encountered an error in the errorTrial test. Aborting\n");
			cutilDeviceReset();
			exit(EXIT_FAILURE);
	} else if (!opt_quiet)
		printf("\nError trial completed successfully.\n");

	if (!opt_quiet) {
		char buf[64];
		printFriendlyTime(buf, (elapsedMsecDEV*testPrime)/1000);
		
		printf("\nTiming:  To test M%d"
				"\n  elapsed time :      %10d msec = %.1f sec"
				"\n  dev. elapsed time:  %10d msec = %d sec"
				"\n  est. total time:    %15s",
				testPrime,
				(int)elapsedMsec, elapsedMsec/1000,
				(int)(elapsedMsecDEV*trialFraction), (int)(elapsedMsecDEV*trialFraction/1000),
				buf);
	
		time_t eta_time = (elapsedMsecDEV*(testPrime-resume_iter))/1000.0 + time(NULL);    // eta relative to 'now'
		strftime(buf, 64, "%A %c", localtime(&eta_time));
		printf(" = %s\n", buf);
	}


	if (resuming) {
		printf("\nResuming full test of M%d at iteration %d (%2.1f%%)\n\n", testPrime, resume_iter, 100.0f * (float)resume_iter / (float)testPrime);
		iter = resume_iter;
	} else if (!opt_quiet) {
		printf("\nBeginning full test of M%d\n\n", testPrime);
	}

	// prepare graceful-exit signal handler
	struct sigaction act;
	act.sa_handler = gracefulExit;
	sigemptyset(&act.sa_mask);
	act.sa_flags = 0;
	sigaction(SIGINT, &act, 0);
	sigaction(SIGQUIT, &act, 0);
	sigaction(SIGTERM, &act, 0);
	sigaction(SIGTSTP, &act, 0);

	cutilSafeCall(cudaEventRecord(start, 0));
	// If we're not resuming, this isn't malloc'd yet.
	if (cp_signal == NULL)
		cp_signal = (Real *) malloc(sizeof(Real)*signalSize);

	mersenneTest(cp_signal);

	// This isn't free'd by freeArrays();
	if (cp_signal != NULL)
		free(cp_signal);

	//get the the total elapsed time in ms
	cutilSafeCall(cudaEventRecord(stop, 0));
	cutilSafeCall(cudaEventSynchronize(stop));

    cutilSafeCall(cudaEventElapsedTime(&elapsedMsec, start, stop));
	
	cutilSafeCall(cudaEventDestroy(start));
	cutilSafeCall(cudaEventDestroy(stop));

	if (!do_gracefulExit) {
		// Remove checkpoint files unless we're exiting gracefully
		(void) unlink (checkpoint_file);
		(void) unlink (checkpoint_backup);
	}

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
static __global__ void ComplexPointwiseSqr(Complex* z_signal, const int size) {
	const int tid = UNROLL_KERNEL * blockIdx.x*blockDim.x + threadIdx.x;
	
	Complex temp[UNROLL_KERNEL];
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		temp[i] = z_signal[tid + i*blockDim.x];
	}
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		if (tid + i*blockDim.x < size) {
			// We are too memory-bound for any specific multiplication method to make much difference
			// Use cuCmul from cuComplex.h for simplicity's sake
			z_signal[tid + i*blockDim.x] = cuCmul(temp[i], temp[i]);
		}
	}
} 

/**
 * compute A and Ainv in extended precision, cast to doubles
 *   and load them to the host arrays
 * Uses dd_real 128-bit double-doubles to avoid catastropic cancellation errors
 *   for non-power-of-two FFT lengths
 */
static __host__ void computeWeightVectors(double *host_A, double *host_Ainv) {

	dd_real dd_A, dd_Ainv;
	dd_real dd_N = dd_real(signalSize);
	dd_real dd_2 = dd_real(2.0);

	for (int ddex = 0; ddex < signalSize; ddex++) {
		dd_real dd_expo = dd_real(ddex)*dd_real((int)testPrime)/dd_N;
		dd_A = pow(dd_2, ceil(dd_expo) - dd_expo);
		dd_Ainv = (1.0 / dd_A) / dd_N;
		host_A[ddex] = to_double(dd_A);
		host_Ainv[ddex] = to_double(dd_Ainv);
	}
}

static __host__ void computeBitsPerWord(int *bitsPerWord) {

	double PoverN = testPrime/(double)signalSize;
	for (int j = 1; j <= signalSize; j++) {	
		bitsPerWord[j - 1] = (int) (ceil(PoverN*j) - ceil(PoverN*(j - 1)));
	}
}

/**
 * do modular wrap-around to get successive words from element [size - 1]
 * Works backwards to get preceeding bits
 */
static __host__ void computeBitsPerWordVectors(uint8_t *bitsPerWord8, int *bitsPerWord) {

	for (int i = 0; i < signalSize; i++) {
		bitsPerWord8[i] = 0;

		for (int bit = 0; bit < 8; bit++) {
			short bitval;
			if (i - bit < 0)
				bitval = (bitsPerWord[signalSize + i - bit] == h_LO_BITS ? 0 : 1);
			else
				bitval = (bitsPerWord[             i - bit] == h_LO_BITS ? 0 : 1);
			bitsPerWord8[i] |= bitval << bit;
		}
	}	
}

// load values of int array into double array for FFT.  Low-order 2 bytes go in lowest numbered
//     position in dArr
static __global__ void loadValue4ToFFTarray(double *d_signal) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid == 0)
		d_signal[tid] = 4.0;
	else
		d_signal[tid] = 0.0;
}


// This includes pseudobalance by adding hi order terms from last rebalancing.
static __global__ void loadIntToDoubleIBDWT(double *d_signal, const int *i_signal, const int8_t *i_hiBitArr, const double *dev_A) {
	const int tid = UNROLL_KERNEL * blockIdx.x*blockDim.x + threadIdx.x;
	
	int ival[UNROLL_KERNEL*2];
	double dval[UNROLL_KERNEL];
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		ival[i] = i_hiBitArr[tid  + i*blockDim.x -1];
	}
	
	if (tid == 0)
		ival[0] = i_hiBitArr[DEV_SIGNALSIZE - 1];
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		ival[i+UNROLL_KERNEL] = i_signal[tid + i*blockDim.x];
		dval[i] = dev_A[tid + i*blockDim.x];
	}

#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		ival[i] += ival[i+UNROLL_KERNEL];
	}
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		dval[i] *= (double)ival[i];
	}
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		if (tid + i*blockDim.x < DEV_SIGNALSIZE) {
			d_signal[tid + i*blockDim.x] = dval[i];
		}
	}
}

/**
 * We assume the a_inv also includes the 1/SIGNAL_SIZE scaling needed by the DFT
 * We also do the subtract 2 from the Lucas-square, requiring no carry in the
 *   current balanced carry-save signal.
 */
// Error version assigns the round-off error back to errorvals[tid]
template <int error_flag>
static __global__ void invDWTproductMinus2(int64_t *llint_signal, const double *d_signal, const double *dev_Ainv, float *dev_errArr) {
	const int tid = UNROLL_KERNEL * blockIdx.x * blockDim.x + threadIdx.x;
	
	double2 sig[UNROLL_KERNEL];
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		sig[i].x = d_signal[tid + i*blockDim.x];
		sig[i].y = dev_Ainv[tid + i*blockDim.x];
	}
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		sig[i].x *= sig[i].y;
	}
	
	if (tid == 0)
		sig[0].x -= 2.0;
	
#pragma unroll
	for (int i=0; i<UNROLL_KERNEL; i++) {
		llint_signal[tid + i*blockDim.x] = __double2ll_rn(sig[i].x);
	}
	
	if (error_flag) {
#pragma unroll
		for (int i=0; i<UNROLL_KERNEL; i++) {
			dev_errArr[tid + i*blockDim.x] = (float) fabs(sig[i].x - llrint(sig[i].x));
		}
	}
}

/**
 * uses Xfer to host and then sequential max check on array from errorVector computed above
 *   called seldom (currently, every 1/50 of the total iterations), so no effect on runtime.
 */
static __host__ float findMaxErrorHOST(float *dev_errArr, float *host_errArr) {

	cutilSafeCall(cudaMemcpy(host_errArr, dev_errArr, sizeof(float)*signalSize, cudaMemcpyDeviceToHost));
	float maxVal = 0.0f;
	for (int i = 0; i < signalSize; i++)
		if (host_errArr[i] > maxVal)
			maxVal = host_errArr[i];
	return maxVal;
}

/**
 *computeMaxVector()
 *function returns list of number of significant bits of a list of int64_ts
 *AS IS, list can only be as long as however many strings you can launch, now 67,107,840 on 2.0 gpus
 */
static __global__ void computeMaxBitVector(float *dev_errArr, int64_t *llint_signal){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < DEV_SIGNALSIZE) {
		if (llint_signal[tid] >= 0){
			dev_errArr[tid] = (float) __clzll(llint_signal[tid]);
		}
		else{
			dev_errArr[tid] = (float) __clzll(llint_signal[tid]*-1);
		}
	}
}


/**
 * mersenneIter()
 * Run a single iteration of the test
 * Called from mersenneTest, errorTrial, and findSignalSize
 */

template <int check_error, int timing>
static __host__ float mersenneIter() {
	cudaEvent_t start, stop;
	float elapsedTime, totalTime=0;

	if (timing) {
		cutilSafeCall(cudaEventCreate(&start));
		cutilSafeCall(cudaEventCreate(&stop));
		cutilSafeCall(cudaEventRecord(start, 0));
	}

	// Transform signal
	if (unlikely(in_place)) {
		cufftSafeCall(CUFFT_EXECFORWARD(plan1, d_signal, (cufftDoubleComplex *) d_signal));
		cutilCheckMsg("Kernel execution failed [ CUFFT_EXECFORWARD ]");
		
		// Multiply the coefficients componentwise
		ComplexPointwiseSqr<<<numFFTblocks, T_PER_B/UNROLL_KERNEL>>>((cufftDoubleComplex *) d_signal, signalSize/2 + 1);
		cutilCheckMsg("Kernel execution failed [ ComplexPointwiseSqr ]");
		
		// Transform signal back
		cufftSafeCall(CUFFT_EXECINVERSE(plan2, (cufftDoubleComplex *) d_signal, d_signal));
		cutilCheckMsg("Kernel execution failed [ CUFFT_EXECINVERSE ]");
	} else {
		cufftSafeCall(CUFFT_EXECFORWARD(plan1, d_signal, z_signal));
		cutilCheckMsg("Kernel execution failed [ CUFFT_EXECFORWARD ]");

		// Multiply the coefficients componentwise
		ComplexPointwiseSqr<<<numFFTblocks, T_PER_B/UNROLL_KERNEL>>>(z_signal, signalSize/2 + 1);    
		cutilCheckMsg("Kernel execution failed [ ComplexPointwiseSqr ]");

		// Transform signal back
		cufftSafeCall(CUFFT_EXECINVERSE(plan2, z_signal, d_signal)); 
		cutilCheckMsg("Kernel execution failed [ CUFFT_EXECINVERSE ]");
	}
	
	if (timing) {
		cutilSafeCall(cudaEventRecord(stop, 0));
		cutilSafeCall(cudaEventSynchronize(stop));
		cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
		if (opt_verbose)
			printf("Time for FFT, squaring, INV FFT:  %3.3f ms\n", elapsedTime);
		totalTime += elapsedTime;
	}

	// Every so often, we do some error checking.
	if (check_error) {
		invDWTproductMinus2<1><<<numBlocks, T_PER_B/UNROLL_KERNEL>>>(llint_signal, d_signal, dev_Ainv, dev_errArr);
		// do error calculation in the function that calls us, since we don't need to touch dev_errArr any more in this function
		// error = findMaxErrorHOST(dev_errArr, host_errArr, signalSize);
	} else {
		invDWTproductMinus2<0><<<numBlocks, T_PER_B/UNROLL_KERNEL>>>(llint_signal, d_signal, dev_Ainv, dev_errArr);
	}

	cutilCheckMsg("Kernel execution failed [ invDWTproductMinus2 ]");


	if (timing) {
		float maxerr = findMaxErrorHOST(dev_errArr, host_errArr);
		if (maxerr >= ERROR_LIMIT) {
			if (opt_verbose)
				printf("max abs error = %f, is too high. Exiting\n", maxerr);
			return -1;
		} else if (opt_verbose)
			printf("\n[%d/50]: iteration %d: max abs error = %f", iter/(testPrime/50), iter, maxerr);
		
		computeMaxBitVector<<<numBlocks, T_PER_B>>>(dev_errArr, llint_signal);
		cutilCheckMsg("Kernel execution failed [ computeMaxBitVector ]");
		float maxBitVector = findMaxErrorHOST(dev_errArr, host_errArr);
		if (opt_verbose) {
			printf("\n[%d/50]: iteration %d: max Bit Vector = %f", iter/(testPrime/50), iter, maxBitVector);
			fflush(stdout);
		}

		// Time rebalancing
		cutilSafeCall(cudaEventRecord(start, 0));
	}

	// REBALANCE llint TIMING
	sliceAndDice<<<numBlocks, T_PER_B>>>(i_signal, i_hiBitArr, llint_signal, bitsPerWord8);
	cutilCheckMsg("Kernel execution failed [ sliceAndDice ]");

	loadIntToDoubleIBDWT<<<numBlocks, T_PER_B/UNROLL_KERNEL>>>(d_signal, i_signal, i_hiBitArr, dev_A);
	cutilCheckMsg("Kernel execution failed [ loadIntToDoubleIBDWT ]");

	if (timing) {
		cutilSafeCall(cudaEventRecord(stop, 0));
		cutilSafeCall(cudaEventSynchronize(stop));
		cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
		if (opt_verbose)
			printf("\nTime to rebalance llint:  %3.3f ms\n", elapsedTime);
		totalTime += elapsedTime;
		cutilSafeCall(cudaEventDestroy(start));
		cutilSafeCall(cudaEventDestroy(stop));
		return totalTime;
	} else
		return 0;
}

/**
 * mallocArrays()
 * cudaMalloc the GPU arrays
 */
static __host__ int mallocArrays() {
	// Allocate device memory for signal
	int i_size = sizeof(int)*signalSize;
	int d_size = sizeof(Real)*signalSize;
	int z_size = sizeof(Complex)*(signalSize/2 + 1);
	int bpw_size = sizeof(uint8_t)*signalSize;
	int llintSignalSize = sizeof(int64_t)*signalSize;

	// Check for available device RAM before allocating this signalSize
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
	in_place = 0;
	// we need roughly 50 times the signalSize in bytes, plus the fft.
	// Memory needs:
	// cudaMallocs: roughly 50 times the signalSize + 896 bytes, minimum 2048KiB
	// cufft plans: roughly 32 times the signalSize, minimum 1024KiB
	unsigned int estimated_size = max(50 * signalSize/1024 + 896, 2048) + max(32 * signalSize/1024, 1024);
	if ((unsigned int)free_mem/1024 < estimated_size) {
		// We don't have enough available memory, can we do slower in-place transorms?
		estimated_size = max(42 * signalSize/1024 + 896, 2048) + max(32 * signalSize/1024, 1024);
		if ((unsigned int)free_mem/1024 < estimated_size) {
			printf("Not enough available device memory (Needed: %dKiB, have: %dKiB)\n", estimated_size, (int)free_mem/1024);
			fflush(stdout);
			return -1;
		} else {
			printf("Using in-place transform ");
			in_place = 1;
		}
	}

	cutilSafeCall(cudaMalloc((void**)&i_signal, i_size));
	cutilSafeCall(cudaMalloc((void**)&d_signal, d_size));
	if (in_place == 0)
		cutilSafeCall(cudaMalloc((void**)&z_signal, z_size));
	cutilSafeCall(cudaMalloc((void**)&dev_A, d_size));
	cutilSafeCall(cudaMalloc((void**)&dev_Ainv, d_size));
	cutilSafeCall(cudaMalloc((void**)&bitsPerWord8, bpw_size));
	cutilSafeCall(cudaMalloc((void**)&llint_signal, llintSignalSize));
	h_signalOUT = (int *) malloc(i_size);

	// Array for high-bit carry out
	cutilSafeCall(cudaMalloc((void**)&i_hiBitArr, sizeof(int8_t)*signalSize));

	//make host and device arrays for error computation
	cudaMalloc((void**) &dev_errArr, signalSize*sizeof(float));
	host_errArr = (float *) malloc(signalSize*sizeof(float));

	// CUFFT plan
	cufftSafeCall(cufftPlan1d(&plan1, signalSize, CUFFT_TYPEFORWARD, 1));
	cufftSafeCall(cufftPlan1d(&plan2, signalSize, CUFFT_TYPEINVERSE, 1));

	// Store computed bit values and bases for precomputation of masks
	h_bitsPerWord = (int *) malloc(i_size);
	h_bitsPerWord8 = (uint8_t *) malloc(sizeof(uint8_t)*signalSize);

	// Make sure all the cuda Arrays are cleared before we start working on them
	cutilSafeCall(cudaMemset(i_signal, 0, i_size));
	cutilSafeCall(cudaMemset(d_signal, 0, d_size));
	if (in_place == 0)
		cutilSafeCall(cudaMemset(z_signal, 0, z_size));
	cutilSafeCall(cudaMemset(dev_A, 0, d_size));
	cutilSafeCall(cudaMemset(dev_Ainv, 0, d_size));
	cutilSafeCall(cudaMemset(bitsPerWord8, 0, bpw_size));
	cutilSafeCall(cudaMemset(llint_signal, 0, llintSignalSize));
	cutilSafeCall(cudaMemset(i_hiBitArr, 0, sizeof(int8_t)*signalSize));
	cutilSafeCall(cudaMemset(dev_errArr, 0, signalSize*sizeof(float)));

	// Compute word-sizes to use when dicing products to sum to int array
	computeBitsPerWord(h_bitsPerWord);
	computeBitsPerWordVectors(h_bitsPerWord8, h_bitsPerWord);
	cutilSafeCall(cudaMemcpy(bitsPerWord8, h_bitsPerWord8, bpw_size, cudaMemcpyHostToDevice));

	h_A = (double *) malloc(signalSize*sizeof(double));
	h_Ainv = (double *) malloc(signalSize*sizeof(double));
	
	// compute weights in extended precision, essential for non-power-of-two signal_size
	computeWeightVectors(h_A, h_Ainv);
	cutilSafeCall(cudaMemcpy(dev_A, h_A, d_size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(dev_Ainv, h_Ainv, d_size, cudaMemcpyHostToDevice));

	return 0;
}

static __host__ void freeArrays() {
	//Destroy CUFFT context
	cufftSafeCall(cufftDestroy(plan2));
	cufftSafeCall(cufftDestroy(plan1));

	// cleanup memory
	free(h_signalOUT);
	free(h_bitsPerWord);
	free(h_bitsPerWord8);
	free(h_A);
	free(h_Ainv);
	free(host_errArr);
	
	cutilSafeCall(cudaFree(i_signal));
	cutilSafeCall(cudaFree(d_signal));
	if (in_place == 0)
		cutilSafeCall(cudaFree(z_signal));

	cutilSafeCall(cudaFree(dev_A));
	cutilSafeCall(cudaFree(dev_Ainv));
	cutilSafeCall(cudaFree(bitsPerWord8));
	cutilSafeCall(cudaFree(llint_signal));
	
	cutilSafeCall(cudaFree(i_hiBitArr));

	cutilSafeCall(cudaFree(dev_errArr));
}

/**
 * findSignalSize()
 * Determines the best signalSize to use for a given testPrime
 * Choice based on runtime and error
 */
static __host__ int findSignalSize() {
	int optimal_length = 0;
	float bestTime = 99999; // dummy value
	uint64_t signalSize64; // need to be bigger than necessary so some of the FFTlen combinations don't overflow

	// Start with testing lengths between a minimum = 1/20th of the testPrime, rounded to the nearest T_PER_B multiple
	// and a maximum = next higher power-of-two, but no higher than 1/14th of the testPrime
	// to avoid wasting our time testing un-necessarily large sizes
	unsigned int min_nx = ((testPrime / 20 / T_PER_B) + 1) * T_PER_B;

	unsigned int max_nx = 1;
	while (max_nx < (testPrime >> 4)) // shift until max_nx is a power of two large enough to cover 1/16th the testPrime
		max_nx <<= 1;
	max_nx = min(max_nx, ((testPrime / 14 / T_PER_B) + 1) * T_PER_B); // but just in case it's too big, cap it at 1/14th the testPrime

	if (!opt_quiet)
		printf("Testing FFT lengths between %d and %d\n\n", min_nx, max_nx);

	int retry = 0;
	cudaEvent_t start_findSignalSize, stop_findSignalSize;
	cutilSafeCall(cudaEventCreate(&start_findSignalSize));
	cutilSafeCall(cudaEventCreate(&stop_findSignalSize));

restart_findSignalSize:
	float elapsedTime, maxerr=0;

	// Need to run enough iterations to build-up the error, if there is any
	// This seems to be around 40-45 iterations in practice.
	int testIterations = 50;

	for (int two = 0; two <= MAX_2; two++) {
		for (int three = 0; three <= MAX_3; three++) {
			for (int five = 0; five <= MAX_5; five++) {
				for (int seven = 0; seven <= MAX_7; seven++) {
					signalSize64 = (powl(2,two) * powl(3,three) * powl(5,five) * powl(7,seven));
					if (
							(signalSize64 <= (unsigned int)max_nx) &
							(signalSize64 >= (unsigned int)min_nx) &
							(signalSize64 % T_PER_B == 0) &
							(3*three + 5*five + 7*seven < 15) // This test should eliminate most of the error-prone compound lengths
							) {
						maxerr = 0;
						signalSize = signalSize64;
						numBlocks = signalSize/T_PER_B;
						numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;
						setSliceAndDice();
						initConstantSymbols();

						if (!opt_quiet) {
							printf("Testing signalSize %9d... ", signalSize);
							fflush(stdout);
						}
 
						maxerr = mallocArrays();
						if (maxerr != 0)
							break;

						// load the int array to the doubles for FFT
						// This is already balanced, and already multiplied by a_0 = 1 for DWT
						loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal);
						cutilCheckMsg("Kernel execution failed [ loadValue4ToFFTarray ]");
					
						int iter = 0;
						// start timer
						cutilSafeCall(cudaEventRecord(start_findSignalSize, 0));
						for (; likely(iter < testIterations); iter++) {
							mersenneIter<1, 0>();

							float this_err = findMaxErrorHOST(dev_errArr, host_errArr);
							maxerr = std::max(this_err, maxerr);

							if (maxerr >= ERROR_LIMIT) // no need to continue this test
								break;
						}

						cutilSafeCall(cudaEventRecord(stop_findSignalSize, 0));
						cutilSafeCall(cudaEventSynchronize(stop_findSignalSize));
						cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start_findSignalSize, stop_findSignalSize));
					
						freeArrays();

						if (!opt_quiet)
							printf("time: %5.3fms, error: %6.5f", (float)elapsedTime/(float)iter, maxerr);

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
		min_nx = max_nx;
		max_nx = max_nx + 8*T_PER_B;
		retry++;
		if ((retry > 10) | (maxerr < 0)) { // Should this provide a wide enough range before bailing out?
			fprintf(stderr, "Could not find a suitable signalSize, exiting\n");
			exit (-1);
		}

		if (!opt_quiet)
			printf("Could not find a signalSize; Increasing search range\n");

		goto restart_findSignalSize;
	}
		
	cutilSafeCall(cudaEventDestroy(start_findSignalSize));
	cutilSafeCall(cudaEventDestroy(stop_findSignalSize));
	
	return optimal_length;
}

/**
* errorTrial()
*/
static __host__ float errorTrial(int testIterations) {
	// Run at least 100 testIterations. We'll encounter a floating point error later on if we don't
	// main() calls us with at least 100 iterations already, so this shouldn't happen
	if (testIterations<100)
		testIterations = 100;

	mallocArrays();

	if (opt_verbose) {
		for (int i = 0; i < 20; i++) {
			printf("word[%d] numbits = %d\n", i, h_bitsPerWord[i]);
			printf("numbits of this and following 7 are: ");
			for (int bit = 1; bit < 256; bit *= 2)
				printf("%d ", bit & h_bitsPerWord8[i] ? h_HI_BITS : h_LO_BITS);
			printf("\n");
		}

		for (int i = signalSize - 8; i < signalSize; i++) {
			printf("word[%d] numbits = %d\n", i, h_bitsPerWord[i]);
			printf("numbits of this and following 7 are: ");
			for (int bit = 1; bit < 256; bit *= 2)
				printf("%d ", bit & h_bitsPerWord8[i] ? h_HI_BITS : h_LO_BITS);
			printf("\n");
		}

		printf("Weight vector looks like:\n");
		for (int i = 0; i < 20; i++) 
			printf("a[%d] = %f\n", i, h_A[i]);
		for (int i = 0; i < 20; i++) 
			printf("ainv[%d] = %f\n", i, h_Ainv[i]);
	}

	numBlocks = signalSize/T_PER_B;
	numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

	// load the int array to the doubles for FFT
	// This is already balanced, and already multiplied by a_0 = 1 for DWT
	loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal);
	cutilCheckMsg("Kernel execution failed [ loadValue4ToFFTarray ]");

	float totalTime = 0;
	// Loop testIterations times
	for (iter = 0; likely(iter < testIterations); iter++) {
		if (iter % (testIterations/100) == 0) {
			float perIterTime = mersenneIter<1, 1>();

			// If mersenneIter returned an error value, pass it along
			if (unlikely(perIterTime < 0)) {
				totalTime = -1;
				break;
			}

			totalTime += perIterTime;
		}
		else {
			mersenneIter<0, 0>();
		}
	}
	
	// Average time per iteration
	totalTime /= 100;

	// DONE!  Final copy out from GPU, since not done by default as for CPU stages
	// DO GOOD REBALANCE HERE
	cudaEvent_t start, stop;
	cutilSafeCall(cudaEventCreate(&start));
	cutilSafeCall(cudaEventCreate(&stop));
	cutilSafeCall(cudaEventRecord(start, 0));

	addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signal, i_hiBitArr);
	cutilCheckMsg("Kernel execution failed [ addPseudoBalanced ]");
	rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signal, bitsPerWord8);
	cutilCheckMsg("Kernel execution failed [ rebalanceIrrIntSEQGPU ]");
	cutilSafeCall(cudaMemcpy(h_signalOUT, i_signal, sizeof(int)*signalSize, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaEventRecord(stop, 0));
	cutilSafeCall(cudaEventSynchronize(stop));
	float elapsedTime;
	cutilSafeCall(cudaEventElapsedTime(&elapsedTime, start, stop));
	if (opt_verbose)
		printf("\nTime to rebalance and write-back:  %3.1f ms\n", elapsedTime);
	// Add this to the total time, for checkpointing
	if (!opt_quiet)
		totalTime += elapsedTime/checkpoint_freq;
	cutilSafeCall(cudaEventDestroy(start));
	cutilSafeCall(cudaEventDestroy(stop));

	freeArrays();

	return totalTime;
}

/**
* print_residue() -- output the Lucas-Lehmer residue for non-prime exponents
*   needed for result submission to GIMPS, or verifying results with other clients
*/
static __host__ void print_residue(int *h_signalOUT) {
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
static __host__ void mersenneTest(Real *cp_signal) {
	mallocArrays();

	if (resuming) {
		cutilSafeCall(cudaMemcpy(d_signal, cp_signal, sizeof(Real) * signalSize, cudaMemcpyHostToDevice));
	}    

	numBlocks = signalSize/T_PER_B;
	numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

	if (!resuming) {
		iter = 2;
		
		// load the int array to the doubles for FFT
		// This is already balanced, and already multiplied by a_0 = 1 for DWT
		loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal);
		cutilCheckMsg("Kernel execution failed [ loadValue4ToFFTarray ]");
	}

	float maxerr = 0.0f;


	int last_cp_iter = iter;

	// Timers for displaying timing at checkpoint, and ETA to completion
	// Only used if not quiet
	cudaEvent_t cp_start, cp_stop;

	cutilSafeCall(cudaEventCreate(&cp_start));
	cutilSafeCall(cudaEventCreate(&cp_stop));
	cutilSafeCall(cudaEventRecord(cp_start, 0));
	cutilSafeCall(cudaEventSynchronize(cp_start));

	sigset_t x;
	sigemptyset (&x);
	sigaddset(&x, SIGINT);
	sigaddset(&x, SIGQUIT);
	sigaddset(&x, SIGTERM);
	sigaddset(&x, SIGTSTP);

	// Loop M-2 times
	for (; likely(iter < testPrime); iter++) {
		// Every so often, do some error checking. Do this at every checkpoint as well
		if (unlikely((iter % (testPrime/1000) == 0) | (iter % checkpoint_freq == 1))) {
			mersenneIter<1, 0>();
			maxerr = findMaxErrorHOST(dev_errArr, host_errArr);
			// FIXME: -Aaron: we should add a way to stop/restart if the error is too high.
			if (unlikely(maxerr >= ERROR_LIMIT)) {
				printf("\n\n\nError limit exceeded: %f >= %f\n\n\n", maxerr, ERROR_LIMIT);
			}
		} else {
			mersenneIter<0, 0>();
		}

		// Checkpoint every checkpoint_freq iterations
		// To match CUDALucas output: CL counts iterations differently and thus displays residue at what we consider to be iteration+1
		if (unlikely(iter % checkpoint_freq == 1 | do_gracefulExit == 1)) {
			// Whatever we do, do not allow an interrupt while we're writing a checkpoint for risk of corrupting the checkpoint files
			sigprocmask(SIG_BLOCK, &x, NULL);

			// This sync shouldn't be necessary, but it's best to be safe when writing a checkpoint
			// Timing cost of memcpy and writing the checkpoint is negligible < .01ms
			cutilDeviceSynchronize();
			cutilSafeCall(cudaMemcpy(cp_signal, d_signal, sizeof(Real) * signalSize, cudaMemcpyDeviceToHost));
			writeCheckpoint(cp_signal);
			sigprocmask(SIG_UNBLOCK, &x, NULL);

			if (unlikely(do_gracefulExit)) {
				// no need to do the rest of this function, if we're exiting early gracefully
				freeArrays();
				return;
			}    


			if (!opt_quiet) {
				cutilSafeCall(cudaEventRecord(cp_stop, 0));
				cutilSafeCall(cudaEventSynchronize(cp_stop));

				// Get the the total elapsed time in ms since last checkpoint
				float elapsedMsec;
				cutilSafeCall(cudaEventElapsedTime(&elapsedMsec, cp_start, cp_stop));
				float iter_time = elapsedMsec / (iter - last_cp_iter);

				// calculate ETA in seconds
				int eta_diff = iter_time * (float) (testPrime - iter) / 1000.0f;

				// Reset timer
				last_cp_iter = iter;
				cutilSafeCall(cudaEventRecord(cp_start, 0)); 
				cutilSafeCall(cudaEventSynchronize(cp_start));

				if (opt_verbose) {
					// Display current iteration's residue and error if verbose
					addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signal, i_hiBitArr);
					cutilCheckMsg("Kernel execution failed [ addPseudoBalanced ]");
					
					// FIXME: Timing cost of the rebalance is expensive > .5 seconds per call for 26xxxxxx exponent
					rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signal, bitsPerWord8);
					cutilCheckMsg("Kernel execution failed [ rebalanceIrrIntSEQGPU ]");
					
					cutilSafeCall(cudaMemcpy(h_signalOUT, i_signal, sizeof(int)*signalSize, cudaMemcpyDeviceToHost));
				}

				// buf needs to fit formatted eta_time
				char buf[64];
				// Every 20 lines, print out extra information. Standard console is 24 lines
				if (iter % (checkpoint_freq * 20) == 1) {
					time_t eta_time = eta_diff + time(NULL);	// eta relative to 'now'
					strftime(buf, 64, "%A %c", localtime(&eta_time));
					printf("%s v%s: testing %d (n = %d)\nEstimated completion: %s\n", program_name, program_version, testPrime, signalSize, buf);
				}

				printFriendlyTime(buf, eta_diff);

				if (opt_verbose) {
					// Lie about the current iteration to match other programs
					printf("[%4.1f%%] Iter %9d: %6.3f ms/iter, ETA %s, ", 100.0f * (float)iter / (float)testPrime, iter - 1, iter_time, buf);
					print_residue(h_signalOUT);
					printf("\n");
				} else {
					printf("[%4.1f%%] Iter %9d: %6.3f ms/iter, ETA %s\n", 100.0f * (float)iter / (float)testPrime, iter - 1, iter_time, buf);
				}

				fflush(stdout);
			}
		}
	}

	cutilSafeCall(cudaEventDestroy(cp_stop));
	cutilSafeCall(cudaEventDestroy(cp_start));

	
	if (do_gracefulExit) {
		// no need to do the rest of this function, if we're exiting early gracefully
		freeArrays();
		return;
	}

	// DONE!  Final copy out from GPU, since not done by default as for CPU stages
	// DO GOOD REBALANCE HERE
	addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signal, i_hiBitArr);
	cutilCheckMsg("Kernel execution failed [ addPseudoBalancd ]");
	
	rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signal, bitsPerWord8);
	cutilCheckMsg("Kernel execution failed [ rebalanceIrrIntSEQGPU]");

	cutilSafeCall(cudaMemcpy(h_signalOUT, i_signal, sizeof(int)*signalSize, cudaMemcpyDeviceToHost));

	bool nonZeros = false;
	for (int i = 0; i < signalSize; i++) {
		if (h_signalOUT[i] > 0) {
			nonZeros = true;
			break;
		}
	}
	if (nonZeros) {
		if (testPrime < 50000 & opt_verbose) {
			for (int i = 0; i < signalSize; i++) {
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
		print_residue(h_signalOUT);
	} else {
		printf("\n\n\aPRIME FOUND: M_%d tests as prime.", testPrime);
		printf("\nM( %d )P", testPrime);

	}
	printf(", n = %d, %s v%s\n", signalSize, program_name, program_version);

	freeArrays();
	return;
}

static __host__ void printFriendlyTime(char *buf, int time) {
	int length = 0;
	if (time >= 86400) {
		// print days
		length = sprintf(buf, "%dd ", time/86400);
	}    
	
	if (time >= 3600) {
		// print hours
		length += sprintf(buf + length, "%02d:", (time%86400)/3600);
	}    
	
	// print minutes:seconds
	sprintf(buf + length, "%02d:%02d", (time%3600)/60, time%60);
}

static __host__ Real *readCheckpoint(int *signalSize, int *resume_iter) {
	FILE *fPtr;
	int testPrime_r, resumeiter_r, signalSize_r; 
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
				fprintf(stderr, "failed.\n\nUnable to find a valid checkpoint file, starting from the beginning\n");
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
		fprintf(stderr, "Checkpoint file was created with a different application, not using checkpoint\n\n");
		return NULL;
	}

	if (strcmp(program_vers_r, program_version) != 0) {
		fprintf(stderr, "Checkpoint file was created with a different version of %s, attempting to continue\n\n", program_name);
	}

	// check parameters
	if (fread (&testPrime_r,  1, sizeof (testPrime_r),  fPtr) != sizeof (testPrime_r)  ||
		fread (&signalSize_r, 1, sizeof (signalSize_r), fPtr) != sizeof (signalSize_r) ||
		fread (&resumeiter_r, 1, sizeof (resumeiter_r), fPtr) != sizeof (resumeiter_r)) {
			fprintf (stderr, "\nThe checkpoint file doesn't match the current test.  Exiting...\n");
			fclose (fPtr);
			exit(-1);
	}
	
	if (testPrime != testPrime_r) { 
		fprintf (stderr, "\nThe checkpoint file doesn't match the current test.  Exiting...\n");
		fclose (fPtr);
		exit(-1);
	}
	
	// check for successful read of z, delayed until here since zSize can vary
	signal = (Real *) malloc (sizeof (Real) * (signalSize_r));
	if (fread (signal, 1, sizeof (Real) * (signalSize_r), fPtr) != (sizeof (Real) * (signalSize_r))) {
		fprintf (stderr, "\nThe checkpoint file doesn't match the current test.  Exiting...\n");
		fclose (fPtr);
		free (signal);
		exit(-1);
	}

	// We have a good checkpoint. Return it
	*resume_iter = resumeiter_r + 1;
	*signalSize = signalSize_r;
	return signal;
}

static __host__ void writeCheckpoint (Real *signal) {
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
	sync();
	fclose (fPtr);
}
