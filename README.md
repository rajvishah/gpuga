The following files compose the essentials of a simple genetic algorithms

1. ga_defs.h 		: includes all typedefs, structures and enum definitions
2. ga_init.c 		: structure init/destroy functions, set and print parameters functions
3. ga_1DPopulation.c	: Create, Init, Print 1DBIN genome/population
4. ga_1DBIN_kernels.cu	: Kernels for crossover, mutation evaluation and supporting device functions
5. ga_simplega.cu	: Core program performing population evolution

supporting files
1. ga_debug.h		: debug print macros 
2. ga_helpers.h		: random number realted functions(serial) to be defined here (common for all types)

The following are the example files, providing an outline how to use the above files for your application

1. example1 : Find a 1 dimensional binary string with alternate 0 and 1
Files : ex1.cu, ex1.h
Notes: Uses a serial fitness function.It's a basic example, doesn't need user data for fitness evaluation

2. example2 : Solve a 0-1 Knapsack problem
Files : ex2.cu, ex2.h
Notes: Uses a serial fitness function. Requires user to pass weights and costs. (See how UDATA is defined.)

3. example3 : Solve a 0-1 Knapsack problem (USER'S KERNEL)
Files : ex3.cu, ex3.h, usereval.cu
Notes: Let user provide his own function for fitness evaluation which makes use of cuda premitives.
	Given example calculates genome weights and scores using a parallel sum.

Compilation : nvcc ex1.cu,nvcc ex2.cu,nvcc ex3.cu,


For more details:
Contact: rajvi.shah@research.iiit.ac.in
