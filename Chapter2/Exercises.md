# Chapter 2 - Heterogeneous Data Parallel Computing

## Exercises

_1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to the data index (i)?_

```
i = blockIdx.x*blockDim.x + threadIdx.x;
```

_2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?_

Since each thread processes two __adjacent__ elements each, the indices are doubled
```
i = (blockIdx.x*blockDim.x + threadIdx.x)*2 
```

_3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2blockDim.x consecutive elements that form two sections. All threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element?_

Since each thread only processes one element, and then the entire block moves on to the next section, 
```
i = 2*blockIdx.x*blockDim.x + threadIdx.x
```

_4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?_

The number of blocks will be `ceil(8000/1024) = 8`. Hence, the number of threads will be `8*1024 = 8192`. The last 92 threads will not perform any work

_5. If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?_

The second argument takes the number of bytes to be allocated. So, the appropriate expression would be `v*sizeof(int)`

_6. If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an  appropriate expression for the first argument of the cudaMalloc() call?_

The first argument of cudaMalloc expects a pointer to a generic pointer type. Hence, the appropriate expression would be `(void **) &A_d`

_7. If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?_

`cudaMemcpy(A_h, A_d, 3000, cudaMemcpyHostToDevice)`

_8. How would one declare a variable err that can appropriately receive the returned value of a CUDA API call?_

The variable has to be of the appropriate type - `cudaError_t err;` (Reference: [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8))

_9. Consider the following CUDA kernel and the corresponding host function that calls it:_
```
__global__ void foo_kernel(float a, float b, unsigned int N){
  unsigned int i=blockIdx.xblockDim.x + threadIdx.x;
  if(i , N) {
    b[i]=2.7fa[i] - 4.3f;
  }
}

void foo(float a_d, float b_d) {
  unsigned int N=200000;
  foo_kernel <<<(N + 128-1)/128, 128>>>(a_d,b_d, N);
}
```
_a. What is the number of threads per block?_

The number of threads in each block is the second argument in the grid launch configuration parameters - 128 in this snippet

_b. What is the number of threads in the grid?_

`num_threads = num_blocks*threads_per_block = floor((200000+128-1)/128)*128 = 200064`
**Note that integer division in C always truncates**

_c. What is the number of blocks in the grid?_

The number of blocks in the grid is the first argument in the grid launch configuratino parameters - `(200000+128-1)/128 = 1563` in this case
**Note that integer division in C always truncates**

_d. What is the number of threads that execute the code on line 02?_

Every thread launched in the grid will reach this line, so `200064`

_e. What is the number of threads that execute the code on line 04?_

Only the threads with global index `i < N` will reach this line, so `200000`

_10. A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?_

The same function can be declared as both `__device__` and `__host__`. The CUDA compiler will generate byte code for both variants simultaneously
