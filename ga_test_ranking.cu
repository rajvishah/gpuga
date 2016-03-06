#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "cudpp/cudpp.h"
//#include "/home/allusers/NEW/GPU_SDK/C/common/inc/cudpp/cudpp.h"
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
__global__ void rankStatistics(long int numElements,float *tempMax,float *tempMin,int *tempMaxId,int *tempMinId,int idFlag);

int main()
{
	long int i,numElements;
	float *h_arr,*d_max_arr,*d_min_arr;
	int *h_ids;
	int *d_maxId,*d_minId;

	float base = 2.0;
	float exponent = 1.0;

	struct timeval tv1;
	struct timezone tz1;
	unsigned long start_seconds;
	unsigned long start_micro_seconds;
	unsigned long end_seconds;
	unsigned long end_micro_seconds;
	unsigned long difference_seconds1,difference_seconds2;
	unsigned long difference_micro_seconds1,difference_micro_seconds2; 


	printf("\nEnter the number of elements (pow of 2):");
	scanf("%f",&exponent);
	printf("\nExponent is %f",exponent);	
	numElements = (int)(pow(base,exponent));
	printf("\nNumElements: %lu",numElements);
	printf("\nHMM");

	float testmax = 0;
	long int testid[200],cnt=0;

	h_arr = (float *)calloc(numElements,sizeof(float));
	h_ids = (int *)calloc(numElements,sizeof(float));

	if(h_arr == NULL)
		printf("\nSuccess");
	for(i=0;i<numElements;i++)
	{	
		//	h_arr[i] = rand()%12982;
		h_arr[i] = i;
		h_ids[i] = i;
		if(h_arr[i] > testmax)
		{
			testmax = h_arr[i];
		}
	}

	printf("\narr generated");
	for(i=0;i<numElements;i++)
	{
		if(h_arr[i] == testmax)
		{
			testid[cnt] = i;
			cnt++;
		}
	}
	printf("\nid found");
	//	cudaMalloc((void**)&d_final_st,sizeof(STATS));
	cudaMalloc((void**)&d_max_arr,numElements*sizeof(float));
	cudaMalloc((void**)&d_min_arr,numElements*sizeof(float));
	cudaMalloc((void**)&d_maxId,numElements*sizeof(long int));
	cudaMalloc((void**)&d_minId,numElements*sizeof(long int));

	printf("\ndev mem allocated");
	cudaMemcpy(d_max_arr,h_arr,numElements*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_min_arr,h_arr,numElements*sizeof(float),cudaMemcpyHostToDevice);
	
	//	cudaMemcpy(d_max_arr,h_arr,numElements*sizeof(float),cudaMemcpyHostToDevice);
	//	cudaMemcpy(d_min_arr,h_arr,numElements*sizeof(float),cudaMemcpyHostToDevice);


/*	int threadsPerBlock 	= 	256;
	int numBlocks		=	(numElements/threadsPerBlock) + ((numElements % threadsPerBlock == 0)? 0:1);

	printf("\nnumblks = %d",numBlocks);
	long int counter = numElements;
	int idFlag = 1;

	int iter = 0;
	float max,min;
	long int maxId,minId;

	gettimeofday(&tv1,&tz1);
	start_seconds=tv1.tv_sec;
	start_micro_seconds=tv1.tv_usec;

	while(counter != 0)
	{
		printf("\nIter no: %d",iter++);
		rankStatistics<<<numBlocks,threadsPerBlock>>>(numElements,d_max_arr,d_min_arr,d_maxId,d_minId,idFlag);
		cudaThreadSynchronize();

		if((int)(counter/threadsPerBlock) > 0)
		{
			numElements = (counter/threadsPerBlock) + ((counter%threadsPerBlock == 0)?0:1);
			counter = numElements;
		}
		else
		{
			if(numElements > counter%threadsPerBlock)
				numElements %= threadsPerBlock;
			else
				counter = 0;
		}
		idFlag = 0;

	//	cudaMemcpy((&max),d_max_arr,sizeof(float),cudaMemcpyDeviceToHost);
	//	cudaMemcpy((&maxId),d_maxId,sizeof(long int),cudaMemcpyDeviceToHost);

   // 	printf("\nMax no %f,max id %d",max,maxId);

	}

	gettimeofday(&tv1,&tz1);
	end_seconds=tv1.tv_sec;
	end_micro_seconds=tv1.tv_usec;

	end_micro_seconds += end_seconds * 1000000;
	start_micro_seconds += start_seconds * 1000000;   				
	difference_micro_seconds1 = end_micro_seconds - start_micro_seconds;

	cudaMemcpy((&max),d_max_arr,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy((&maxId),d_maxId,sizeof(long int),cudaMemcpyDeviceToHost);

                    */


	    //USING CUDPP
		  cudaMemcpy(d_min_arr,h_arr,numElements*sizeof(float),cudaMemcpyHostToDevice);
		  cudaMemcpy(d_minId,h_ids,numElements*sizeof(int),cudaMemcpyHostToDevice);


		  CUDPPConfiguration config;
		  config.algorithm = CUDPP_SORT_RADIX;
		  config.datatype = CUDPP_FLOAT;
		  config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

		  CUDPPHandle plan;
		  CUDPPResult result = CUDPP_SUCCESS; 
		  result = cudppPlan(&plan, config, numElements, 1, 0);	

		  CUDPPHandle scanplan = 0;
		  result = cudppPlan(&scanplan, config, numElements, 1, 0);  

		  if (CUDPP_SUCCESS != result)
		  {
		  printf("Error creating CUDPPPlan\n");
		  exit(-1);
		  }

		  gettimeofday(&tv1,&tz1);
		  start_seconds=tv1.tv_sec;
		  start_micro_seconds=tv1.tv_usec;

		  result = cudppSort(plan, (void*)d_min_arr, (void*)d_minId, 32, numElements);            			 
		  if (CUDPP_SUCCESS != result)
		  {
			  printf("Error creating CUDPPSort\n");
			  exit(-1);
		  }


		  gettimeofday(&tv1,&tz1);
		  end_seconds=tv1.tv_sec;
		  end_micro_seconds=tv1.tv_usec;

		  end_micro_seconds += end_seconds * 1000000;
		  start_micro_seconds += start_seconds * 1000000;
		  difference_micro_seconds2 = end_micro_seconds - start_micro_seconds;

		  float newMax = 0.0;
		  int newMaxId = 0;

		  cudaMemcpy(h_arr,d_min_arr,(numElements*sizeof(float)),cudaMemcpyDeviceToHost);	
		  cudaMemcpy(h_ids,d_minId,(numElements*sizeof(int)),cudaMemcpyDeviceToHost);

		  printf("\nMax::%f",h_arr[0]);

//		  printf("\nMax score me: %f, max Id me %lu,\nmax score cudpp %f, max Id cudpp %d",max,maxId,h_arr[0],h_ids[0]);
//		  printf("\nCPU Max score : %f",testmax);
//		  for(i=0;i<200;i++)
//		  printf("\nMax Id is: %ld",testid[i]);

//		  printf("\nMy time %f\nCUDPPTime %f",difference_micro_seconds1,difference_micro_seconds2);


}



__global__ void rankStatistics(long int numElements,float *tempMax,float *tempMin,int *tempMaxId,int *tempMinId,int idFlag)
{
	__shared__ float minscores[256];
	__shared__ float maxscores[256];
	__shared__ long int minId[256];
	__shared__ long int maxId[256];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idFlag == 1 && idx < numElements)
	{
		tempMinId[idx] = idx;
		tempMaxId[idx] = idx;
	}

	int nActiveThreads;
	int halfPoint;
	float temp;
	long int maxThreads = numElements;

	__syncthreads();	
	if(idx < maxThreads)
	{
		minscores[threadIdx.x] = tempMin[idx];
		maxscores[threadIdx.x] = tempMax[idx];
		minId[threadIdx.x] = tempMinId[idx];
		maxId[threadIdx.x] = tempMaxId[idx];
	}

	if(maxThreads > blockDim.x)
		nActiveThreads = blockDim.x;
	else
		nActiveThreads = maxThreads;
	__syncthreads();
	while(nActiveThreads > 1)
	{
		halfPoint = (nActiveThreads >> 1);
		if(idx < maxThreads)
		{

			//printf("\nMT %d AT %d, HP %d",maxThreads,nActiveThreads,halfPoint);
			if(threadIdx.x < halfPoint)
			{
				temp = minscores[threadIdx.x + halfPoint];
				if (temp < minscores[threadIdx.x])
				{
					minscores[threadIdx.x] = temp;
					minId[threadIdx.x] = tempMinId[idx + halfPoint];
					tempMinId[idx] = minId[threadIdx.x];
				}	
			}

			if(threadIdx.x < halfPoint)
			{
				temp = maxscores[threadIdx.x + halfPoint];
				//  	printf("\ncomparing %f with %f, (%d - %d)",maxscores[threadIdx.x],temp,tempMaxId[idx],tempMaxId[idx + halfPoint]);
				if (temp > maxscores[threadIdx.x])
				{
					maxscores[threadIdx.x] = temp;
					maxId[threadIdx.x] = tempMaxId[idx + halfPoint];
					tempMaxId[idx] = maxId[threadIdx.x];
				}

			}
		}
		__syncthreads();
		nActiveThreads = (nActiveThreads >> 1);
	}
	if(idx < maxThreads && threadIdx.x == 0)
	{
		tempMin[blockIdx.x] = minscores[0];	
		tempMax[blockIdx.x] = maxscores[0];
		tempMaxId[blockIdx.x] = maxId[0];	
		tempMinId[blockIdx.x] = minId[0];

		//	    	printf("\nMax score: %f, max Id %d, block id %d",tempMax[blockIdx.x],maxId[blockIdx.x],blockIdx.x);
	}
	__syncthreads();
}
