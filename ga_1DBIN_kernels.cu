#include <stdio.h>
#include <cuda.h>


// DEVICE FUNCTIONS
__device__ int randcu(GARand *rng)
{
	int num;
	int id;
	id = *(rng->randID)%(rng->RPERIOD);
	*(rng->randID) += 1;
//	printf("\n%d",*(rng->randID));
	num = rng->randGPU[id];
	return num;
}

__device__ int select(GARand *rng,GAContext *ga,GAStats *stats)
{
	int genomeID;
	if(ga->selection == UNIFORMSEL)
	{
		genomeID = randcu(rng);
		genomeID %= (ga->popsize);
	}
	/* else if */
	return genomeID;
} 

// POPULATION EVAL RELATED KERNELS
// CALLS FITNESS FUNC

#ifndef _USEREVALFUNC_


extern __device__ float FitnessFunc(BIN1D *g,GNMContext genome,UDATA *udata);
__global__ void evaluate1DBINPopulation(BIN1D *devPop,GAContext devga,GNMContext devgenome,GAStats devstats,UDATA *dudata)
{
	BIN1D *g;

	int POPSIZE = devga.popsize;
	int WIDTH   = devgenome.width;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < POPSIZE)
	{
//		KDPRINTF("\nGenome is %d",idx);
		g = &(devPop[idx*WIDTH]);
		devstats.scores[idx] = FitnessFunc(g,devgenome,dudata);	 
	}
}

#endif

// CROSSOVER RELATED KERNELS

// ONEPOINT PREPROCESS
__global__ void xOverPreProcess(GAContext devga,GNMContext devgenome,GAStats devstats,GARand rng,
					 int *momID,int *dadID,int *xOverPt)
{
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int popSize	=	 devga.popsize;
	int width 	= 	 devgenome.width;
	
	if(x < popSize)
	{
		if(x%2 == 0)
		{
			xOverPt[x] = randcu(&rng)%width;
			xOverPt[x + 1] = xOverPt[x];

			momID[x] = select(&rng,&devga,&devstats);
			momID[x + 1] = momID[x];

			dadID[x] = select(&rng,&devga,&devstats);
			dadID[x + 1] = dadID[x];

		}
	}
}

// ONEPOINT CROSSOVER
__global__ void xOnePoint1DBIN(BIN1D *newPop,BIN1D *oldPop,GAContext devga,GNMContext devgenome,
				GAStats devstats,GARand rng, int *momID,int *dadID,int *xOverPt)
{
	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int popSize	=	 devga.popsize;
	int width 	= 	 devgenome.width;
	
	int writeIdx,writeIdy,readIdx,readIdy;
	if(x < popSize)
	{
		writeIdx = x;
		writeIdy = y;
		
		if(x % 2 == 0)
		{
			if(y < xOverPt[x])
			{
				readIdx = momID[x];
				readIdy = y;
			}
			else	
			{
				readIdx = dadID[x];
				readIdy = y;
			}
		}
		else
		{
			if(y < xOverPt[x])
			{
				readIdx = dadID[x];
				readIdy = y;
			}
			else	
			{
				readIdx = momID[x];
				readIdy = y;
			}
		}
		newPop[(writeIdx*width) + writeIdy] = oldPop[readIdy+(readIdx*width)];
//		KDPRINTF("\nI am thread %d reading genome %d gene %d value %d",x,readIdx,readIdy,oldPop[readIdy+(readIdx*width)]);
	}
}

// FLIP BIT MUTATION KERNEL 
__global__ void flipMutate1DBIN(BIN1D *newPop,BIN1D *oldPop,GAContext devga,GNMContext devgenome,GAStats devstats,GARand rng)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int popSize	=	 devga.popsize;
	int width 	= 	 devgenome.width;
	
	float geneMut = devga.pMut/(width);
	float toss;
	
	if(x < popSize)
	{
		toss = ((float)randcu(&rng))/(rng.RMAX);
		if(toss < geneMut)
		{
			newPop[(x*width) + y] = 1  - newPop[y+(x*width)];
//			KDPRINTF("\nI am genome %d , my %d bit got muatated",x,y);
		}
	}
}

__global__ void xOnePointmFlip1DBIN(BIN1D *newPop,BIN1D *oldPop,GAContext devga,GNMContext devgenome,
					GAStats devstats,GARand rng, int *momID,int *dadID,int *xOverPt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int popSize	=	 devga.popsize;
	int width 	= 	 devgenome.width;
	
	float geneMut = devga.pMut/(width);
	float toss;
	
	int writeIdx,writeIdy,readIdx,readIdy;
	if(x < popSize)
	{
		writeIdx = x;
		writeIdy = y;
		
		if(x % 2 == 0)
		{
			if(y < xOverPt[x])
			{
				readIdx = momID[x];
				readIdy = y;
			}
			else	
			{
				readIdx = dadID[x];
				readIdy = y;
			}
		}
		else
		{
			if(y < xOverPt[x])
			{
				readIdx = dadID[x];
				readIdy = y;
			}
			else	
			{
				readIdx = momID[x];
				readIdy = y;
			}
		}
		toss = ((float)randcu(&rng))/(rng.RMAX);
		if(toss < geneMut)
		{
			newPop[(writeIdx*width) + writeIdy] = 1 - oldPop[readIdy+(readIdx*width)];
//			KDPRINTF("\nI am genome %d , my %d bit got muatated",x,y);
		}
		else
		{
			newPop[(writeIdx*width) + writeIdy] = oldPop[readIdy+(readIdx*width)];
		}
		
//		KDPRINTF("\nI am thread %d reading genome %d gene %d value %d",x,readIdx,readIdy,oldPop[readIdy+(readIdx*width)]);
	}
}

// Statistics update related kernels
__global__ void rankStatistics(long int numElements,float *tempMax,float *tempMin,long int *tempMaxId,long int *tempMinId,int idFlag)
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

        //              printf("\nMT %d AT %d, HP %d",maxThreads,nActiveThreads,halfPoint);
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
        //                      printf("\ncomparing %f with %f, (%d - %d)",maxscores[threadIdx.x],temp,tempMaxId[idx],tempMaxId[idx + halfPoint]);
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

        //      printf("\nMax score: %f, max Id %d, block id %d",tempMax[blockIdx.x],maxId[blockIdx.x],blockIdx.x);
        }
        __syncthreads();
}
















































