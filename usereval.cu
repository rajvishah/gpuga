/* This file defines a evaluation function */
/* This enables user to write his own kernel for fitness evaluation */
/* In this example, instead of using a serial score sum per genome a parallel sum is used */
/* cudaParallelSumShared and cudaParallelSumGlobal both perform the same task but the former makes use of shared memory */
/* User needs to set the kernel parameters carefully */
/* user needs to take care of loading UDATA and storing scores back */

#define _USEREVALFUNC_
#define weights(i) tempStorage[weightIdx + (i)]
#define score(i) tempStorage[scoreIdx + (i)]
#include "ga_debug.h"

// Do not use this directly
// This kernel does correct calculations but fails on thread exit
// Faster but Needs be Debugged
__global__ void cudaParallelSumShared(BIN1D *fitWPop,BIN1D *fitVPop,float *dscores,const int width,int popSize,UDATA *udata)
{
	const int dimX = blockDim.x;
	
	int weightIdx = 0;	
	int scoreIdx = (dimX*width*sizeof(int));

	extern __shared__ int tempStorage[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int tempWx = threadIdx.x;
	int tempWy = y;
	
	if(y < width && x < popSize)
	{
		weights(tempWy + width*tempWx) = (int)(fitWPop[x*width + y] * udata->weights[y]);
		score(tempWy + width*tempWx) = (int)(fitVPop[x*width + y] * udata->values[y]);
	}

	__syncthreads();

	int numIter = ceil(log((float)width)/log((float)2));
	int iter = 1;
	int temp,divisor;

	while(iter <= numIter)
	{
		__syncthreads();
		temp = pow((float)2,(float)(iter-1));
		divisor = 2*temp;
		if(x < popSize)
		{
			if(y%divisor == 0)
			{
				if(y + temp < width)
				{
					score(tempWy + width*tempWx) += score((tempWy+temp) + width*tempWx);
					weights(tempWy + width*tempWx) += weights((tempWy+temp) + width*tempWx);
				}
			}
//			if(y == 0)
//			printf("\nGenome %d, Iter %d, value %d",x,iter,scores(tempWy + width*tempWx));
		}
		
		
		__syncthreads();
		iter = iter + 1;
	}
	
	if(x < popSize)
	{
		if(weights(width*tempWx) <= udata->maxWeight)
		{
			dscores[x] = 1.0 * score(width*tempWx);
		}
		else
		{
			dscores[x] = 0.0;
		}
	}
	__syncthreads();
//	printf("\n(%d %d) All well",x,y);
	return;
}

// This kernel makes use of global memory to store intermediate results
// Shared memory kernel is much faster - 
// This is just to demonstrate how to set up your own evaluation function and kernel
__global__ void cudaParallelSumGlobal(BIN1D *fitWPop,BIN1D *fitVPop,float *scores,int width,int popSize,UDATA *udata)
{

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(y < width && x < popSize)
	{
		//printf("\nGenome %d, Gene %d, value %d",x,y,fitVPop[x*width + y]);
		fitWPop[x*width + y] = fitWPop[x*width + y] * udata->weights[y];
		fitVPop[x*width + y] = fitVPop[x*width + y] * udata->values[y];
		//printf("\nGenome %d, Gene %d, value %d",x,y,fitVPop[x*width + y]);
	}
	
	__syncthreads();
		
	int numIter = ceil(log((float)width)/log((float)2));
	int iter = 1;
	int temp,divisor;

	while(iter <= numIter)
	{

		temp = pow((float)2,(float)(iter-1));
		divisor = 2*temp;
		if(x < popSize)
		{
			if(y%divisor == 0)
			{
				if(y + temp < width)
				{
					fitWPop[x*width + y] = fitWPop[x*width + y] + fitWPop[x*width + (y+temp)];
					fitVPop[x*width + y] = fitVPop[x*width + y] + fitVPop[x*width + (y+temp)];
				}
			}
//			if(y == 0)
//			printf("\nGenome %d, Iter %d, value %d",x,iter,fitVPop[x*width + y]);
		}
		
		
		__syncthreads();
		iter = iter + 1;
	}
	__syncthreads();
	
	if(x < popSize)
	{
	//	printf("UDATA MAXW = %d",udata->maxWeight);
	//	printf("\nGenome %d, score %f",x,(float)fitVPop[x*width]);
		if(fitWPop[x*width] <= udata->maxWeight)
		{
			scores[x] = 1.0 * fitVPop[x*width];
		}
		else
		{
			scores[x] = 0.0;
		}
	}
}

void evaluate1DBINPopulationUser(BIN1D *devPop,GAContext *ga,GNMContext *genome,GAStats *stats,UDATA *udata)
{

	cudaError_t status;

	int width = genome->width;
	int popsize = ga->popsize;
	long int size;

	// Pointers to memory to be allocated on device memory only once
	static UDATA *dudata;
	static BIN1D *fitWPop;
	static BIN1D *fitVPop;
	static float *scores;
	static int allocFlag = 0;
	
	int xDim = 256/genome->width;
	int yDim = genome->width;
	dim3 threadsPerBlock(xDim,yDim);
	int numBlocks = ((ga->popsize)/xDim) + ((ga->popsize % xDim == 0)? 0:1);;
	int sharedMemSize = 2*(xDim)*(yDim)*sizeof(int);

	if(allocFlag != 1)
	{
		allocFlag = 1;
	
		size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
		status = cudaMalloc((void **)&fitWPop,size);
		if(status != cudaSuccess)
		{
			DPRINTF("\nIn Usr Function evaluate1DBINPop:: Could not allocate memory for population on device");
			return;
		}

		status = cudaMalloc((void **)&fitVPop,size);
		if(status != cudaSuccess)
		{
			DPRINTF("\nIn Usr Function evaluate1DBINPop:: Could not allocate memory for population on device");
			return;
		}

		status = cudaMalloc((void **)&scores,(ga->popsize)*sizeof(float));
		if(status != cudaSuccess)
		{
			DPRINTF("\nIn Usr Function evaluate1DBINPop:: Could not allocate memory for population on device");
			return;
		}		

		status = cudaMalloc((void **)&dudata,sizeof(UDATA));
		if(status != cudaSuccess)
		{
			DPRINTF("\nIn Usr Function evaluate1DBINPop:: Could not allocate memory for population on device");
			return;
		}
		status = cudaMemcpy(dudata,udata,sizeof(UDATA),cudaMemcpyHostToDevice);
		if(status != cudaSuccess)
		{
			DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not copy population on device");
			return;
		}


	}
	size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
	status = cudaMemcpy(fitWPop,devPop,size,cudaMemcpyDeviceToDevice);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function evalPopulationU:: Could not copy population on device");
		return;
	}

	status = cudaMemcpy(fitVPop,devPop,size,cudaMemcpyDeviceToDevice);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function evalPopulationU:: Could not copy population on device");
		return;
	}
		
	cudaParallelSumShared<<<numBlocks,threadsPerBlock,sharedMemSize>>>(fitWPop,fitVPop,scores,width,popsize,dudata);
//	cudaParallelSumGlobal<<<numBlocks,threadsPerBlock,sharedMemSize>>>(fitWPop,fitVPop,scores,width,popsize,dudata);
	cudaThreadSynchronize();

/*	status = cudaMemcpy(stats->scores,scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToHost);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function evalPopulationU:: Could not copy scores to host");
		return;
	}*/
	
/*	for(int i = 0;i<(ga->popsize);i++)
	printf("\n%f",stats->scores[i]);*/
	if(stats->currIter > stats->maxIter)
	{
		cudaFree(fitVPop);
		cudaFree(fitWPop);
		cudaFree(scores);
	}
}
