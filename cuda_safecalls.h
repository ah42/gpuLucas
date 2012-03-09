/* This contains selected functions from CUDA SDK 4.0.17 */

#include <stdio.h>

inline cudaError cutilDeviceSynchronize() {
#if CUDART_VERSION >= 4000
        return cudaDeviceSynchronize();
#else
        return cudaThreadSynchronize();
#endif
}

inline cudaError cutilDeviceReset() {
#if CUDART_VERSION >= 4000
        return cudaDeviceReset();
#else
        return cudaThreadExit();
#endif
}

// Assist the compiler with branch prediction
#define likely(x) __builtin_expect((x),1)
#define unlikely(x) __builtin_expect((x),0)

// We define these calls here, so the user doesn't need to include __FILE__ and __LINE__
// The advantage is the developers gets to use the inline function so they can debug
#define cutilSafeCallNoSync(err)     __cudaSafeCallNoSync(err, __FILE__, __LINE__)
#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cutilSafeThreadSync()        __cudaSafeThreadSync(__FILE__, __LINE__)
#define cufftSafeCall(err)           __cufftSafeCall     (err, __FILE__, __LINE__)
#define cutilCheckError(err)         __cutilCheckError   (err, __FILE__, __LINE__)
#define cutilCheckMsg(msg)           __cutilGetLastError (msg, __FILE__, __LINE__)

// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
// when the user double clicks on the error line in the Output pane. Like any compile error.

inline void __cudaSafeCallNoSync( cudaError err, const char *file, const int line ) {
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCallNoSync() Runtime API error %d : %s.\n",
				file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

inline void __cudaSafeCall( cudaError err, const char *file, const int line ) {
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
				file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

inline void __cudaSafeThreadSync( const char *file, const int line ) {
	cudaError err = cutilDeviceSynchronize();
	if ( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaDeviceSynchronize() Runtime API error %d: %s.\n",
				file, line, (int)err, cudaGetErrorString( err ) );
		exit(-1);
	}
}

inline void __cutilCheckError( bool err, const char *file, const int line ) {
    if( true != err) {
        fprintf(stderr, "%s(%i) : CUDA error.\n",
        		file, line);
        exit(-1);
    }
}

inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line ) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) ); 
        exit(-1);
    }   
}

#ifdef _CUFFT_H_
inline void __cufftSafeCall( cufftResult err, const char *file, const int line ) {
	if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "%s(%i) : cufftSafeCall() CUFFT error %d: ",
				file, line, (int)err);
		switch (err) {
			case CUFFT_INVALID_PLAN:   fprintf(stderr, "CUFFT_INVALID_PLAN\n"); break;
			case CUFFT_ALLOC_FAILED:   fprintf(stderr, "CUFFT_ALLOC_FAILED\n"); break;
			case CUFFT_INVALID_TYPE:   fprintf(stderr, "CUFFT_INVALID_TYPE\n"); break;
			case CUFFT_INVALID_VALUE:  fprintf(stderr, "CUFFT_INVALID_VALUE\n"); break;
			case CUFFT_INTERNAL_ERROR: fprintf(stderr, "CUFFT_INTERNAL_ERROR\n"); break;
			case CUFFT_EXEC_FAILED:    fprintf(stderr, "CUFFT_EXEC_FAILED\n"); break;
			case CUFFT_SETUP_FAILED:   fprintf(stderr, "CUFFT_SETUP_FAILED\n"); break;
			case CUFFT_INVALID_SIZE:   fprintf(stderr, "CUFFT_INVALID_SIZE\n"); break;
			case CUFFT_UNALIGNED_DATA: fprintf(stderr, "CUFFT_UNALIGNED_DATA\n"); break;
			default: fprintf(stderr, "CUFFT Unknown error code\n"); 
		}
		exit(-1);
	}
}
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    { { 0x10,  8 },
      { 0x11,  8 },
      { 0x12,  8 },
      { 0x13,  8 },
      { 0x20, 32 },
      { 0x21, 48 },
      {   -1, -1 }
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }
    printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
    return -1;
}
// end of GPU Architecture definitions


// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId() {
    int current_device   = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, max_perf_device  = 0;
    int device_count     = 0, best_SM_arch     = 0;
    cudaDeviceProp deviceProp;

    cudaGetDeviceCount( &device_count );
    // Find the best major SM Architecture GPU device
    while ( current_device < device_count ) { 
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major > 0 && deviceProp.major < 9999) {
            best_SM_arch = best_SM_arch >  deviceProp.major ? best_SM_arch : deviceProp.major;
        }   
        current_device++;
    }   

    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count ) { 
        cudaGetDeviceProperties( &deviceProp, current_device );
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
            sm_per_multiproc = 1;
        } else {
            sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
        }   

        int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
        if( compute_perf  > max_compute_perf ) { 
            // If we find GPU with SM major > 2, search only these
            if ( best_SM_arch > 2 ) { 
                // If our device==dest_SM_arch, choose this, or else pass
                if (deviceProp.major == best_SM_arch) { 
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }   
            } else {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }   
        }   
        ++current_device;
    }   
    return max_perf_device;
}

