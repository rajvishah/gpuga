#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "cudpp/cudpp.h"

#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>


int main()
{
	long int i,numElements;
	float *h_arr,*d_arr,*h_sorted_arr;
	float *h_ids,*d_ids,*h_sorted_ids;


	float base = 2.0;
	float exponent = 1.0;

	struct timeval tv1;
	struct timezone tz1;
	unsigned long start_seconds;
	unsigned long start_micro_seconds;
	unsigned long end_seconds;
	unsigned long end_micro_seconds;

	unsigned long difference_micro_seconds2 = 0; 


	printf("\nEnter the number of elements (pow of 2):");
	scanf("%f",&exponent);
	//	printf("\nExponent is %f",exponent);	

	numElements = (int)(pow(base,exponent));
	//	printf("\nNumElements: %lu",numElements);

	h_arr = (float *)calloc(numElements,sizeof(float));
	h_sorted_arr = (float *)calloc(numElements,sizeof(float));	
	h_sorted_ids = (float *)calloc(numElements,sizeof(float));
	h_ids = (float *)calloc(numElements,sizeof(float));

	for(i=0;i<numElements;i++)
	{	
		//	h_arr[i] = rand()%12982;
		h_arr[i] = i;
		h_ids[i] = i;
	}

	cudaMalloc((void**)&d_arr,numElements*sizeof(float));
	cudaMalloc((void**)&d_ids,numElements*sizeof(float));

	cudaMemcpy(d_arr,h_arr,numElements*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_ids,h_ids,numElements*sizeof(float),cudaMemcpyHostToDevice);


	//USING CUDPP

	CUDPPConfiguration config;
	config.algorithm = CUDPP_SORT_RADIX;
	config.datatype = CUDPP_FLOAT;
	config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

	CUDPPHandle plan;

	CUDPPResult result = CUDPP_SUCCESS; 
	result = cudppPlan(&plan, config, numElements, 1, 0);	
	if (CUDPP_SUCCESS != result)
	{
		printf("Error creating CUDPPPlan\n");
		exit(-1);
	}
	CUDPPHandle scanplan = 0;
	result = cudppPlan(&scanplan, config, numElements, 1, 0);  

	if (CUDPP_SUCCESS != result)
	{
		printf("Error creating CUDPPscanPlan\n");
		exit(-1);
	}

	gettimeofday(&tv1,&tz1);
	start_seconds=tv1.tv_sec;
	start_micro_seconds=tv1.tv_usec;

	result = cudppSort(plan, (void*)d_arr, (void*)d_ids, 32, numElements);            			 
	if (CUDPP_SUCCESS != result)
	{
		printf("Error in CUDPPSort\n");
		exit(-1);
	}


	gettimeofday(&tv1,&tz1);
	end_seconds=tv1.tv_sec;
	end_micro_seconds=tv1.tv_usec;

	end_micro_seconds += end_seconds * 1000000;
	start_micro_seconds += start_seconds * 1000000;
	difference_micro_seconds2 = end_micro_seconds - start_micro_seconds;

	cudaMemcpy(h_sorted_arr,d_arr,(numElements*sizeof(float)),cudaMemcpyDeviceToHost);	
	cudaMemcpy(h_sorted_ids,d_ids,(numElements*sizeof(float)),cudaMemcpyDeviceToHost);


	printf("\nMax::%f",h_sorted_arr[0]);

	//		  printf("\nMax score me: %f, max Id me %lu,\nmax score cudpp %f, max Id cudpp %d",max,maxId,h_arr[0],h_ids[0]);
	//		  printf("\nCPU Max score : %f",testmax);
	//		  for(i=0;i<200;i++)
	//		  printf("\nMax Id is: %ld",testid[i]);

	//		  printf("\nMy time %f\nCUDPPTime %f",difference_micro_seconds1,difference_micro_seconds2);


}
