i was assigned to add two matrices with C in parallel, or to say using GPU
first i tried working with CUDA toolkit to develop one code specific to my NVIDIA GPU
but there were a lot problems with version compatibility between the CUDA compiler, the GPU,GPU driver, C/C++ compiler
and even their architecture

eventually as im so lazy and unwilling to waste time on these matters
i found [OpenCL](https://en.wikipedia.org/wiki/OpenCL), a free cross platform API for parallel programming using different resources

 i dont have enough technical knowledge or expertise to explain or guess exatcly how this works
 but for now it just solved my problem 
 and i could add two matrices in GPU in a Macbook M1, and the calculations were done in parallel
 if i wanted to do it with CPU it would have been easier to write but i suppose it wasnt a complete parallel then
 and the CPU had to iterate over each element in the matrices and add or substract them

 you can compile this code in Macbooks using:

 ` gcc -framework OpenCL Matrix_opencl.c -o matrix_adder `
 
