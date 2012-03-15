/**
* IrrBaseBalanced.cu -- globals and kernel code to do operations on balanced radix numbers
*   This uses the Crandall irrational base method
*
* A. Thall & A. Hegedus
* Project:  gpuLucas
* 11/6/2010
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
* MODIFICATIONS:
*  8/1/2011:
*     xxAH:  llintToIrrBal() now template function
*     xxAT:  Removed all dependencies on CUDPP and fast carry addition
*  2/19/2012:  xxAT release version
*/

#ifndef D_IRRBASEBALANCED
#define D_IRRBASEBALANCED

/**
 * Distribute product-accumulated bits to subsequent digits of variable base product
 * @template-param number - number of subsequent digits to distribute product bits
 */
template <int number>
static __global__ void llintToIrrBal(int *i_signalOUT, int8_t *i_hiBitArr, int64_t *llint_signal, uint8_t *bitsPerWord8, const int signalSize) {
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const int tba = threadIdx.x; // thread block address for digits index

	// Use int for each element, the radix place and its n preceeding
	__shared__ int64_t digits[T_PER_B + number];
	__shared__ int8_t signs[T_PER_B + number];
	
	// first n threads of block initialize leading digits.
	//   Be carefule to wrap-around from end of array if tid < n
	//   otherwise, load end of previous block at [tid - n]
	if (tba < number) {
		int offset = tid;
		if (tid < number)
			offset += signalSize;
		signs[tba] = llint_signal[offset - number] < 0 ? -1 : 1;
		digits[tba]= llint_signal[offset - number] * signs[tba];
	}

	signs[tba + number] = llint_signal[tid] < 0 ? -1 : 1;
	digits[tba + number] = llint_signal[tid] * signs[tba + number]; 
	uint8_t bperW8 = bitsPerWord8[tid];

	// get info for this digit
	int isHi = bperW8 & 1;
	int BITS = LO_BITS + isHi;
	int myBase = BASE_LO << isHi;
	int myMask = myBase - 1;
	int baseOver2 = myBase >> 1;
	int8_t hival = 0;

	__syncthreads();

	// Walk backwards through the cached long longs, pulling off
	//   higher and higher order bits, all of length myMask for the
	//   current digit.
	// shiftBits is amount to shift word (tid - N) before pulling off
	//   higher order bits with myMask for current digit

	int sum = signs[tba + number]*(digits[tba + number] & myMask);
  	int shiftBits = 0;

	// compiler will unroll this loop by templated number
#pragma unroll
	for (int i = 1; i <= number; i++) {
		shiftBits += LO_BITS + ((bperW8 >> i) & 1);
		sum += signs[tba + number - i]*((digits[tba + number - i] >> shiftBits) &  myMask);
	}

	// do pseudo-rebalance, storing borrow or carry to hiArr[tid]
	if (sum < -baseOver2)
		hival = -((-sum + baseOver2) >> BITS); //  /myBase);
	else if (sum >= baseOver2)
		hival = (sum + baseOver2) >> BITS; // /myBase;

	i_signalOUT[tid] = sum - (hival << BITS);
	i_hiBitArr[tid] = hival;
}

/**
 * do a single carry of the high-order carry of the previous digit to the
 *    current digit.  Don't rebalance if exceeds max or min on balanced
 *    representation.
 */
static __global__ void addPseudoBalanced(int *i_signalOUT, int8_t *i_hiBitArr, int signalSize) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid == 0) 
		i_signalOUT[tid] += i_hiBitArr[signalSize - 1];
	else
		i_signalOUT[tid] += i_hiBitArr[tid - 1];
}

/**
 * Final rebalance of irrational base representation, by one-time-only sequential
 *   add-with-carry with rebalancedIrrIntSEQGPU<<<1, 1>>> call.  Could as easily be
 *   done CPU-side.
 */
// FIXME: -aaron: This is very slow, and costs .5 seconds on each call (every checkpoint)
static __global__ void rebalanceIrrIntSEQGPU(int *i_signalOUT, uint8_t *bitsPerWord8, int signalSize) {

	int carryOut = 0;
	int tBase, tBaseOver2;
	int BASE_HIOVER2 = BASE_HI >> 1;
	int BASE_LOOVER2 = BASE_LO >> 1;
#pragma unroll 128
	for (int i = 0; i < signalSize; i++) {

		if (bitsPerWord8[i] & 1) {
			tBase = BASE_HI;
			tBaseOver2 = BASE_HIOVER2;
		}
		else {
			tBase = BASE_LO;
			tBaseOver2 = BASE_LOOVER2;
		}
		int b = i_signalOUT[i];

		int total = b + carryOut;

		if (total >= tBaseOver2) {
			i_signalOUT[i] = total - tBase;
			carryOut = 1;
		}
		else if (total < -tBaseOver2) {
			i_signalOUT[i] = total + tBase;
			carryOut = -1;
		}
		else {
			i_signalOUT[i] = total;
			carryOut = 0;
		}
	}
	if (carryOut != 0)
		i_signalOUT[0] += carryOut;
}

#endif // #ifndef D_IRRBASEBALANCED
