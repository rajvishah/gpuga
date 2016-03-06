*-----------------------------------------------*
This is an untested archieved code that is related to parts of my work on 
GPU accelerated Genetic Algorithms. 

Rajvi Shah, P J Narayanan, Kishore Kothapalli, 
"GPU-accelerated Genetic Algorithms", 
Workshop on Parallel Architectures for Bio-inspired Algorithms, in conjunction 
with Conference on Parallel Architectures and Compilation Techniques 
(PACT Workshop), 2010, Vienna, Austria. 

Over the last few months I have received multiple requests for the code but I 
could not find the final working copy of the code that I had used for the paper. 
I am publishing an older version of the code that I could fine AS IS, UNTESTED 
and WITHOUT ANY WARRANTY. If you find the code useful, please cite our paper. 

@INPROCEEDINGS{,
	AUTHOR       = "Rajvi Shah, P J Narayanan, Kishore Kothapalli",
	TITLE        = "GPU-Accelerated Genetic Algorithms",
	BOOKTITLE    = "Workshop on Parallel Architectures for Bio-ispired Algorithms in conunction with Parallel Architectures and Compilation Techniques (PACT Workshop)",
	YEAR         = "2010",
	PAGES        = "27--34",
}

The code was tested on old Tesla GPUs many years ago and haven't been 
tested/used since. Please understand that I will not be able to provide 
much help. 

I would be glad if anyone wants to extend this code. 

Rajvi Shah
rajvi.shah@research.iiit.ac.in
rajvi.a.shah@gmail.com
*-----------------------------------------------*


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


