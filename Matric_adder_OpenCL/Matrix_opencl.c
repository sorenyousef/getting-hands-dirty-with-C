/*
================================================================================
MATRIX ADDITION ON GPU USING OPENCL (PURE C)
================================================================================
PURPOSE:
    This program demonstrates parallel matrix addition using GPU computing.
    It adds two 4x4 matrices (A + B = C) using the GPU for acceleration.
WHAT IS GPU COMPUTING?
    - CPU: Good at sequential tasks (one thing at a time, very fast)
    - GPU: Good at parallel tasks (thousands of things simultaneously)
    - For matrix addition, each element can be computed independently
    - GPU can calculate all 16 elements at the same time!
COMPILATION:
    Mac: clang -framework OpenCL matrix_add_opencl.c -o matrix_add -Wno-deprecated-declarations
    Linux: gcc matrix_add_opencl.c -o matrix_add -lOpenCL
   
RUN:
    ./matrix_add

================================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Include OpenCL library for GPU programming
#ifdef __APPLE__
#include <OpenCL/opencl.h>  // macOS uses this path
#else
#include <CL/cl.h>          // Linux/Windows uses this path
#endif
/*
================================================================================
GPU KERNEL CODE (The code that runs on the GPU)
================================================================================
This is written in OpenCL C language (similar to regular C).
The GPU will execute this function in parallel across many threads.
IMPORTANT CONCEPT:
    - Imagine 16 workers (threads) working simultaneously
    - Each worker is responsible for adding ONE pair of numbers
    - Worker 0 adds A[0] + B[0], Worker 1 adds A[1] + B[1], etc.
    - All 16 additions happen at the same time!
KEYWORDS EXPLAINED:
    __kernel  = This function will run on the GPU (like __global__ in CUDA)
    __global  = This pointer points to GPU memory (not CPU memory)
    get_global_id(0) = Returns which thread number this is (0, 1, 2, ... 15)
*/
const char* gpuKernelSourceCode =
"__kernel void addTwoMatricesOnGPU(__global const float* firstMatrix,\n"
"                                  __global const float* secondMatrix,\n"
"                                  __global float* resultMatrix,\n"
"                                  const int totalNumberOfElements)\n"
"{\n"
"    // STEP 1: Figure out which element this thread should process\n"
"    // get_global_id(0) returns: 0, 1, 2, 3, ... up to totalNumberOfElements\n"
"    int myElementIndex = get_global_id(0);\n"
"    \n"
"    // STEP 2: Safety check - make sure we don't access memory we shouldn't\n"
"    // Sometimes we launch more threads than needed for technical reasons\n"
"    if (myElementIndex < totalNumberOfElements) {\n"
"        // STEP 3: Perform the actual addition for this one element\n"
"        // This is the actual work: C[i] = A[i] + B[i]\n"
"        resultMatrix[myElementIndex] = firstMatrix[myElementIndex] + secondMatrix[myElementIndex];\n"
"    }\n"
"}\n";
/*
================================================================================
HELPER FUNCTION: Check for OpenCL Errors
================================================================================
OpenCL functions return error codes. This function checks if an error occurred
and prints a helpful message if something went wrong.
*/
void checkForOpenCLErrors(cl_int errorCode, const char* operationDescription) {
    if (errorCode != CL_SUCCESS) {
        printf("❌ ERROR: %s (OpenCL Error Code: %d)\n", operationDescription, errorCode);
        printf("   The program will now exit.\n");
        exit(1);
    }
}
/*
================================================================================
HELPER FUNCTION: Print a Matrix
================================================================================
Displays a matrix in a readable format on the screen.
*/
void printMatrix(const char* matrixName, float* matrix, int rows, int columns) {
    printf("%s:\n", matrixName);
    for (int row = 0; row < rows; row++) {
        printf("  ");
        for (int col = 0; col < columns; col++) {
            // Calculate the 1D array index from 2D position (row, col)
            int index = row * columns + col;
            printf("%6.2f ", matrix[index]);
        }
        printf("\n");
    }
    printf("\n");
}
/*
================================================================================
MAIN FUNCTION
================================================================================
*/
int main() {
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║     MATRIX ADDITION ON GPU USING OPENCL (PURE C)              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
   
    /* ========================================================================
       STEP 1: DEFINE MATRIX DIMENSIONS AND CALCULATE MEMORY REQUIREMENTS
       ======================================================================== */
   
    printf("📊 STEP 1: Setting up matrix dimensions\n");
    printf("----------------------------------------\n");
   
    // Our matrices will be 4x4 (4 rows, 4 columns)
    const int numberOfRows = 4;
    const int numberOfColumns = 4;
    const int totalNumberOfElements = numberOfRows * numberOfColumns;  // 4 * 4 = 16
   
    // Calculate how much memory we need in bytes
    // sizeof(float) = 4 bytes, so 16 elements * 4 bytes = 64 bytes per matrix
    const size_t memorySizeInBytes = totalNumberOfElements * sizeof(float);
   
    printf("   Matrix size: %d x %d = %d elements\n",
           numberOfRows, numberOfColumns, totalNumberOfElements);
    printf("   Memory needed per matrix: %zu bytes\n\n", memorySizeInBytes);
   
    /* ========================================================================
       STEP 2: CREATE AND INITIALIZE MATRICES ON CPU (HOST MEMORY)
       ======================================================================== */
   
    printf("💾 STEP 2: Creating matrices in CPU memory\n");
    printf("----------------------------------------\n");
   
    // Allocate memory on the CPU (regular RAM) for our three matrices
    float* firstMatrix_CPU = (float*)malloc(memorySizeInBytes);
    float* secondMatrix_CPU = (float*)malloc(memorySizeInBytes);
    float* resultMatrix_CPU = (float*)malloc(memorySizeInBytes);
   
    // Check if memory allocation was successful
    if (!firstMatrix_CPU || !secondMatrix_CPU || !resultMatrix_CPU) {
        printf("❌ ERROR: Failed to allocate CPU memory\n");
        return -1;
    }
   
    // Initialize the first matrix with values 1, 2, 3, 4, 5, ... 16
    float initialValuesForFirstMatrix[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
   
    // Initialize the second matrix with a pattern
    float initialValuesForSecondMatrix[16] = {
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
        4, 4, 4, 4
    };
   
    // Copy the initial values into our matrices
    memcpy(firstMatrix_CPU, initialValuesForFirstMatrix, memorySizeInBytes);
    memcpy(secondMatrix_CPU, initialValuesForSecondMatrix, memorySizeInBytes);
   
    printf("   ✓ Allocated and initialized matrices in CPU memory\n\n");
   
    /* ========================================================================
       STEP 3: INITIALIZE OPENCL - FIND AND CONNECT TO THE GPU
       ======================================================================== */
   
    printf("🎮 STEP 3: Connecting to the GPU\n");
    printf("----------------------------------------\n");
   
    cl_int openclErrorCode;  // Variable to store error codes from OpenCL functions
   
    // STEP 3A: Get the OpenCL Platform
    // A platform represents the OpenCL implementation (Apple, NVIDIA, AMD, etc.)
    cl_platform_id computePlatform;
    openclErrorCode = clGetPlatformIDs(1, &computePlatform, NULL);
    checkForOpenCLErrors(openclErrorCode, "Failed to find OpenCL platform");
   
    // Get the platform name so we know what we're using
    char platformName[128];
    clGetPlatformInfo(computePlatform, CL_PLATFORM_NAME, 128, platformName, NULL);
    printf("   Platform found: %s\n", platformName);
   
    // STEP 3B: Get the GPU Device
    // The device is the actual hardware (the GPU chip)
    cl_device_id gpuDevice;
    openclErrorCode = clGetDeviceIDs(computePlatform, CL_DEVICE_TYPE_GPU, 1, &gpuDevice, NULL);
    checkForOpenCLErrors(openclErrorCode, "Failed to find GPU device");
   
    // Get the GPU name
    char gpuDeviceName[128];
    clGetDeviceInfo(gpuDevice, CL_DEVICE_NAME, 128, gpuDeviceName, NULL);
    printf("   GPU device found: %s\n", gpuDeviceName);
   
    // STEP 3C: Create a Context
    // A context is like "opening a connection" to the GPU
    // It allows us to communicate with the GPU and manage its resources
    cl_context gpuContext = clCreateContext(NULL, 1, &gpuDevice, NULL, NULL, &openclErrorCode);
    checkForOpenCLErrors(openclErrorCode, "Failed to create GPU context");
    printf("   ✓ Created connection to GPU\n");
   
    // STEP 3D: Create a Command Queue
    // The command queue is how we send instructions to the GPU
    // Think of it like a "to-do list" for the GPU
    cl_command_queue gpuCommandQueue = clCreateCommandQueue(gpuContext, gpuDevice, 0,
&openclErrorCode);
    checkForOpenCLErrors(openclErrorCode, "Failed to create command queue");
    printf("   ✓ Created command queue for sending work to GPU\n\n");
   
    /* ========================================================================
       STEP 4: ALLOCATE MEMORY ON THE GPU (DEVICE MEMORY)
       ======================================================================== */
   
    printf("🔧 STEP 4: Allocating memory on GPU\n");
    printf("----------------------------------------\n");
   
    // The GPU has its own separate memory (VRAM - Video RAM)
    // We need to allocate space in GPU memory for our matrices
   
    // Create buffer for first matrix (read-only, GPU will only read from it)
    cl_mem firstMatrix_GPU = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY,
                                            memorySizeInBytes, NULL, &openclErrorCode);
    checkForOpenCLErrors(openclErrorCode, "Failed to allocate GPU memory for first matrix");
   
    // Create buffer for second matrix (read-only)
    cl_mem secondMatrix_GPU = clCreateBuffer(gpuContext, CL_MEM_READ_ONLY,
                                             memorySizeInBytes, NULL, &openclErrorCode);
    checkForOpenCLErrors(openclErrorCode, "Failed to allocate GPU memory for second matrix");
    
    // Create buffer for result matrix (write-only, GPU will write results here)
    cl_mem resultMatrix_GPU = clCreateBuffer(gpuContext, CL_MEM_WRITE_ONLY,
                                             memorySizeInBytes, NULL, &openclErrorCode);
    checkForOpenCLErrors(openclErrorCode, "Failed to allocate GPU memory for result matrix");
   
    printf("   ✓ Allocated %zu bytes for each matrix on GPU\n", memorySizeInBytes);
    printf("   ✓ Total GPU memory used: %zu bytes\n\n", memorySizeInBytes * 3);
   
    /* ========================================================================
       STEP 5: COPY DATA FROM CPU TO GPU
       ======================================================================== */
   
    printf("📤 STEP 5: Copying data from CPU to GPU\n");
    printf("----------------------------------------\n");
   
    // The GPU can't access CPU memory directly, so we must copy data
    // This is like uploading files to a cloud server before processing them
   
    // Copy first matrix from CPU memory to GPU memory
    openclErrorCode = clEnqueueWriteBuffer(gpuCommandQueue, firstMatrix_GPU, CL_TRUE, 0,
                                          memorySizeInBytes, firstMatrix_CPU, 0, NULL, NULL);
    checkForOpenCLErrors(openclErrorCode, "Failed to copy first matrix to GPU");
    printf("   ✓ Copied first matrix to GPU\n");
   
    // Copy second matrix from CPU memory to GPU memory
    openclErrorCode = clEnqueueWriteBuffer(gpuCommandQueue, secondMatrix_GPU, CL_TRUE, 0,
                                          memorySizeInBytes, secondMatrix_CPU, 0, NULL, NULL);
    checkForOpenCLErrors(openclErrorCode, "Failed to copy second matrix to GPU");
    printf("   ✓ Copied second matrix to GPU\n\n");
   
    /* ========================================================================
       STEP 6: COMPILE THE GPU KERNEL CODE
       ======================================================================== */
   
    printf("⚙️  STEP 6: Compiling GPU kernel code\n");
    printf("----------------------------------------\n");
   
    // Create a program object from our kernel source code
    cl_program gpuProgram = clCreateProgramWithSource(gpuContext, 1,
                                                     &gpuKernelSourceCode, NULL,
&openclErrorCode);
    checkForOpenCLErrors(openclErrorCode, "Failed to create program from source");
   
    // Compile the kernel code for the specific GPU we're using
    // This is like compiling C code with gcc, but for GPU architecture
    openclErrorCode = clBuildProgram(gpuProgram, 1, &gpuDevice, NULL, NULL, NULL);
   
    // If compilation fails, print the error log to help debug
    if (openclErrorCode != CL_SUCCESS) {
        char buildLog[4096];
        clGetProgramBuildInfo(gpuProgram, gpuDevice, CL_PROGRAM_BUILD_LOG,
                             sizeof(buildLog), buildLog, NULL);
        printf("❌ Kernel compilation failed:\n%s\n", buildLog);
        return -1;
    }
    printf("   ✓ GPU kernel compiled successfully\n");
   
    // Extract the specific kernel function we want to run
    cl_kernel matrixAdditionKernel = clCreateKernel(gpuProgram, "addTwoMatricesOnGPU",
&openclErrorCode);
    checkForOpenCLErrors(openclErrorCode, "Failed to create kernel");
    printf("   ✓ Kernel function 'addTwoMatricesOnGPU' ready to execute\n\n");
   
    /* ========================================================================
       STEP 7: SET KERNEL ARGUMENTS (PASS PARAMETERS TO GPU FUNCTION)
       ======================================================================== */
   
    printf("📋 STEP 7: Setting kernel parameters\n");
    printf("----------------------------------------\n");
   
    // The kernel function needs to know where the data is located
    // We pass the GPU memory buffers as arguments (like function parameters)
   
    // Argument 0: First matrix (input)
    openclErrorCode = clSetKernelArg(matrixAdditionKernel, 0, sizeof(cl_mem), &firstMatrix_GPU);
    checkForOpenCLErrors(openclErrorCode, "Failed to set kernel argument 0 (first matrix)");
   
    // Argument 1: Second matrix (input)
    openclErrorCode = clSetKernelArg(matrixAdditionKernel, 1, sizeof(cl_mem),
&secondMatrix_GPU);
    checkForOpenCLErrors(openclErrorCode, "Failed to set kernel argument 1 (second matrix)");
   
    // Argument 2: Result matrix (output)
    openclErrorCode = clSetKernelArg(matrixAdditionKernel, 2, sizeof(cl_mem),
&resultMatrix_GPU);
    checkForOpenCLErrors(openclErrorCode, "Failed to set kernel argument 2 (result matrix)");
   
    // Argument 3: Total number of elements (so kernel knows when to stop)
    openclErrorCode = clSetKernelArg(matrixAdditionKernel, 3, sizeof(int),
&totalNumberOfElements);
    checkForOpenCLErrors(openclErrorCode, "Failed to set kernel argument 3 (element count)");
   
    printf("   ✓ All kernel arguments configured\n\n");
   
    /* ========================================================================
       STEP 8: CONFIGURE AND LAUNCH THE GPU KERNEL
       ======================================================================== */
   
    printf("🚀 STEP 8: Launching GPU kernel\n");
    printf("----------------------------------------\n");
   
    // STEP 8A: Determine optimal work group size
   
    // Global work size = Total number of threads to launch
    // We need one thread per matrix element
    size_t totalThreadsToLaunch = totalNumberOfElements;
   
    // Work group size = How many threads work together in a group
    // This is similar to CUDA's "block size"
    // Get the maximum work group size this GPU supports
    size_t maximumWorkGroupSize;
    clGetDeviceInfo(gpuDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                   sizeof(size_t), &maximumWorkGroupSize, NULL);
   
    // Choose a safe work group size
    // Start with 16 threads per group (a good default for small problems)
    size_t threadsPerWorkGroup = 16;
   
    // Make sure we don't exceed GPU's maximum
    if (threadsPerWorkGroup > maximumWorkGroupSize) {
        threadsPerWorkGroup = maximumWorkGroupSize;
    }
   
    // Make sure we don't have more threads per group than total threads
    if (threadsPerWorkGroup > totalThreadsToLaunch) {
        threadsPerWorkGroup = totalThreadsToLaunch;
    }
   
    // STEP 8B: Ensure global size is a multiple of work group size
    // OpenCL requires this for efficiency reasons
    size_t remainder = totalThreadsToLaunch % threadsPerWorkGroup;
    if (remainder != 0) {
        // Round up to the nearest multiple
        totalThreadsToLaunch += (threadsPerWorkGroup - remainder);
    }
   
    printf("   GPU Execution Configuration:\n");
    printf("   ├─ Total threads (global work size): %zu\n", totalThreadsToLaunch);
    printf("   ├─ Threads per work group (local work size): %zu\n", threadsPerWorkGroup);
    printf("   ├─ Number of work groups: %zu\n", totalThreadsToLaunch / threadsPerWorkGroup);
    printf("   └─ Maximum work group size on this GPU: %zu\n\n", maximumWorkGroupSize);
   
    // STEP 8C: Launch the kernel!
    // This tells the GPU to execute our kernel function in parallel
    printf("   ⏳ Executing matrix addition on GPU...\n");
    openclErrorCode = clEnqueueNDRangeKernel(gpuCommandQueue, matrixAdditionKernel, 1, NULL,
                                            &totalThreadsToLaunch, &threadsPerWorkGroup,
                                            0, NULL, NULL);
    checkForOpenCLErrors(openclErrorCode, "Failed to execute kernel on GPU");
    
    // Wait for the GPU to finish all work
    // This is important - we can't read results until GPU is done!
    clFinish(gpuCommandQueue);
    printf("   ✓ GPU computation completed!\n\n");
   
    /* ========================================================================
       STEP 9: COPY RESULTS BACK FROM GPU TO CPU
       ======================================================================== */
   
    printf("📥 STEP 9: Copying results from GPU to CPU\n");
    printf("----------------------------------------\n");
   
    // Now that the GPU is done, we need to get the results back
    // Copy the result matrix from GPU memory back to CPU memory
    openclErrorCode = clEnqueueReadBuffer(gpuCommandQueue, resultMatrix_GPU, CL_TRUE, 0,
                                         memorySizeInBytes, resultMatrix_CPU, 0, NULL, NULL);
    checkForOpenCLErrors(openclErrorCode, "Failed to copy result from GPU to CPU");
    printf("   ✓ Results copied back to CPU memory\n\n");
   
    /* ========================================================================
       STEP 10: DISPLAY THE RESULTS
       ======================================================================== */
   
    printf("📊 STEP 10: Results\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
   
    // Print all three matrices so we can verify the addition worked correctly
    printMatrix("Matrix A (First Input)", firstMatrix_CPU, numberOfRows, numberOfColumns);
    printMatrix("Matrix B (Second Input)", secondMatrix_CPU, numberOfRows, numberOfColumns);
    printMatrix("Matrix C = A + B (Result from GPU)", resultMatrix_CPU, numberOfRows,
numberOfColumns);
   
    // Verify a few results manually
    printf("🔍 Verification:\n");
    printf("   C[0,0] = A[0,0] + B[0,0] = %.2f + %.2f = %.2f ✓\n",
           firstMatrix_CPU[0], secondMatrix_CPU[0], resultMatrix_CPU[0]);
    printf("   C[1,1] = A[1,1] + B[1,1] = %.2f + %.2f = %.2f ✓\n",
           firstMatrix_CPU[5], secondMatrix_CPU[5], resultMatrix_CPU[5]);
    printf("   C[3,3] = A[3,3] + B[3,3] = %.2f + %.2f = %.2f ✓\n\n",
           firstMatrix_CPU[15], secondMatrix_CPU[15], resultMatrix_CPU[15]);
   
    /* ========================================================================
       STEP 11: CLEAN UP - FREE ALL ALLOCATED MEMORY
       ======================================================================== */
   
    printf("🧹 STEP 11: Cleaning up resources\n");
    printf("----------------------------------------\n");
   
    // Release GPU resources (very important to avoid memory leaks!)
    clReleaseMemObject(firstMatrix_GPU);
    clReleaseMemObject(secondMatrix_GPU);
    clReleaseMemObject(resultMatrix_GPU);
    clReleaseKernel(matrixAdditionKernel);
    clReleaseProgram(gpuProgram);
    clReleaseCommandQueue(gpuCommandQueue);
    clReleaseContext(gpuContext);
    printf("   ✓ Released all GPU resources\n");
   
    // Free CPU memory
    free(firstMatrix_CPU);
    free(secondMatrix_CPU);
    free(resultMatrix_CPU);
    printf("   ✓ Freed all CPU memory\n\n");
   
    /* ========================================================================
       PROGRAM COMPLETE
       ======================================================================== */
   
    printf("════════════════════════════════════════════════════════════════\n");
    printf("✅ SUCCESS! Matrix addition completed on GPU\n");
    printf("════════════════════════════════════════════════════════════════\n\n");
   
    printf("📚 Key Takeaways:\n");
    printf("   • GPUs can perform many calculations simultaneously (parallel computing)\n");
    printf("   • Each matrix element was computed by a separate GPU thread\n");
    printf("   • All 16 additions happened at the same time!\n");
    printf("   • This same approach scales to much larger matrices (1000x1000, etc.)\n\n");
   
    return 0;
}