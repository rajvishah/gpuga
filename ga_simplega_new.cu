//////////////////////////////////////////////////////////////////////////////
/////////////////////      MAIN PROGRAM      /////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include "ga_debug.h"
#include "ga_defs.h"
#include "ga_1DPopulation.c"
#include "ga_1DBIN_kernels.cu"


FLAG gaEvolve1DBINPopulation(BIN1D **popAdd,GAContext *ga,GNMContext *genome,GAStats *stats,UDATA *udata)
{
	cudaError_t status;
	size_t size;

	BIN1D *hostPop;
	BIN1D *devOldPop;
	BIN1D *devNewPop;

	GAContext devga;
	GNMContext devgenome;
	GAStats devstats;

	struct timeval tv1;
	struct timezone tz1;
	unsigned long start_seconds,start_micro_seconds,end_seconds,end_micro_seconds,difference_seconds,difference_micro_seconds=0;

	float *tempMax,*tempMin;
	long int *tempMaxId,*tempMinId;

	copyGAStructures(ga,genome,stats,&devga,&devgenome,&devstats);

	Construct1DBINPopulation(&hostPop,ga,genome);
	Init1DBINPopulation(hostPop,ga,genome);
	//Print1DBINPopulation(hostPop,ga,genome);

	/////////////////////////////////////////////////////////////////////
	////////////////////	HANDLE RNG FOR GPU	/////////////////////
	/////////////////////////////////////////////////////////////////////

	// ESTIMATE APPRX USAGE OF RANDOM NUMBERS 

	/* Crossover - popsize for selection, popsize for to xover or not, popsize for crossover points
	// Mutation - popsize*genomewidth for every bit's pMut 
	// 3*popsize + popsize*genomewidth
	// To be random enough, we generate 2*popsize*genomewidth random integers*/

	long int randLimit;
	long int *randNums;
	long int *randInit;
	long int init = 0;

	randInit = &init;
	GARand rng;

	randLimit = (ga->popsize)*(genome->width);

	// ALLOCATE MEMORY FOR RANDOM NUMS ON HOST
	randNums = (long int *)calloc(randLimit,sizeof(long int));
	if(randNums == NULL)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for randList on host");
		return FALSE;
	}

	// GENERATE RANDOM NUMS
	// OPTIMIZATIONS TO BE DONE :: REPLACE THIS PROCESS BY CUDPP RAND
	for(long int i=0;i<randLimit;i++)
	{
		randNums[i] = rand();
	}
	rng.RMAX = RAND_MAX;
	rng.RPERIOD = randLimit;
	// ALLOCATE MEMORY FOR RANDOM NUMS ON DEVICE
	status = cudaMalloc((void**)&rng.randGPU,(randLimit*sizeof(long int)));
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for randList on device");
		free(randNums);
		return FALSE;
	}
	status = cudaMalloc((void**)&rng.randGPU,(randLimit*sizeof(long int)));	

	// COPY RANDOM NUMBERS TO DEVICE
	status = cudaMemcpy(rng.randGPU,randNums,(randLimit*sizeof(long int)),cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not copy population on device");
		return FALSE;
	}	

	free(randNums);	

	status = cudaMalloc((void**)&rng.randID,sizeof(long int));
	status = cudaMemcpy(rng.randID,randInit,sizeof(long int),cudaMemcpyHostToDevice);
	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	if(stats->printTimings == TRUE)
	{
		gettimeofday(&tv1,&tz1);
		start_seconds=tv1.tv_sec;
		start_micro_seconds=tv1.tv_usec;
	}

	// ALLOCATE MEMORY FOR SCORES & SELECTION PROBS

	// ON HOST
	(stats->scores) = (float *)calloc(ga->popsize,sizeof(float));
	if(stats->scores == NULL)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for scores on host");
		return FALSE;
	}

	/*	// ON HOST selProbs
		(stats->selProbs) = (float *)calloc(ga->popsize,sizeof(float));
		if(stats->selProbs == NULL)
		{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for sel probs on host");
		return FALSE;
		}*/

	// ON DEVICE
	status = cudaMalloc((void**)&devstats.scores,(ga->popsize)*sizeof(float));
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for scores on device");
		free(stats->scores);
		return FALSE;
	}	

	/*	// ON DEVICE
		status = cudaMalloc((void**)&devstats.selProbs,(ga->popsize)*sizeof(float));
		if(status != cudaSuccess)
		{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for scores on device");
		free(stats->scores);
		return FALSE;
		}	*/


	// NEW:: MALLOC FOR MAX,MIN,IDs

	status = cudaMalloc((void **)&tempMax,(ga->popsize)*sizeof(float));
	status = cudaMalloc((void **)&tempMin,(ga->popsize)*sizeof(float));
	status = cudaMalloc((void **)&tempMaxId,(ga->popsize)*sizeof(long int)); 
	status = cudaMalloc((void **)&tempMinId,(ga->popsize)*sizeof(long int));


	//	ALLOCATE MEMORY FOR POPULATION - Old Pop and New Pop (BUFFER)

	size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
	status = cudaMalloc((void **)&devOldPop,size);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for population on device");
		free(stats->scores);
		cudaFree(devstats.scores);
		return FALSE;
	}	

	size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
	status = cudaMalloc((void **)&devNewPop,size);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for population on device");
		free(stats->scores);
		cudaFree(devstats.scores);
		cudaFree(devOldPop);
		return FALSE;
	}

	// ALLOCATE MEMORY FOR XOVER VARIABLES
	int *momID;
	int *dadID;
	int *xOverID; 

	size = (ga->popsize)*sizeof(int);

	status = cudaMalloc((void **)&momID,size);
	if(status != cudaSuccess)
	{
		DPRINTF("\nCould not allocate memory on device");
		return FALSE;
	}

	status = cudaMalloc((void **)&dadID,size);
	if(status != cudaSuccess)
	{
		DPRINTF("\nCould not allocate memory on device");
		cudaFree(dadID);
		return  FALSE;
	}

	if(genome->xover == ONEPOINT)
	{
		status = cudaMalloc((void **)&xOverID,size);
		if(status != cudaSuccess)
		{
			DPRINTF("\nCould not allocate memory on device");
			cudaFree(momID);
			cudaFree(dadID);
			return  FALSE;
		}
	}
	/* else if other types change size */

	//   COPY POPULATION TO DEVICE
	size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
	status = cudaMemcpy(devOldPop,hostPop,size,cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not copy population on device");
		return FALSE;
	}	

	// MODIFIED //

	///////////////////////////////////////////////////////////////////////
	/////////////////	EVALUATE POPULATION 	///////////////////////
	/////////////////////////////////////////////////////////////////////// 

#ifdef _USEREVALFUNC_

	int threadsPerBlock1,numBlocks1;
	evaluate1DBINPopulationUser(devOldPop,ga,genome,stats,udata);


#else

	// COPY USER DATA STRUCTURE TO DEVICE MEMORY	
	UDATA *dudata;

	status = cudaMalloc((void **)&dudata,sizeof(UDATA));
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Usr Function evaluate1DBINPop:: Could not allocate memory for population on device");
		return FALSE;
	}
	status = cudaMemcpy(dudata,udata,sizeof(UDATA),cudaMemcpyHostToDevice);
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not copy population on device");
		return FALSE;
	}


	int threadsPerBlock1 = 64;
	int numBlocks1 = ((ga->popsize)/threadsPerBlock1) + ((ga->popsize % threadsPerBlock1 == 0)? 0:1); 
	evaluate1DBINPopulation<<<numBlocks1,threadsPerBlock1>>>(devOldPop,devga,devgenome,devstats,dudata);

	cudaThreadSynchronize();


/*	//  COPY SCORES TO HOST
	status = cudaMemcpy(stats->scores,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToHost);
	if(status != cudaSuccess)
	{
		DPRINTF("\n0 In Function gaEvolve1DBINPopulation:: Could not copy scores on device");
		return FALSE;
	}*/

#endif


	//////////////////////////////////////////////////////////////////////
	///////////////////	  UPDATE STATISTICS 	//////////////////////
	//////////////////////////////////////////////////////////////////////

	// GPU Version To be added Here

	status = cudaMemcpy(tempMax,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToDevice);
	status = cudaMemcpy(tempMin,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToDevice);

	//		printf("\nReady To launch");
	threadsPerBlock1 = 256;
	numBlocks1 = ((ga->popsize)/threadsPerBlock1) + ((ga->popsize % threadsPerBlock1 == 0)? 0:1);


	long int numElements = ga->popsize;
	long int counter = numElements;
	int idFlag = 1;

	while(counter != 0)
	{
		rankStatistics<<<numBlocks1,threadsPerBlock1>>>(numElements,tempMax,tempMin,tempMaxId,tempMinId,idFlag);
		cudaThreadSynchronize();

		if((int)(counter/threadsPerBlock1) > 0)
		{
			numElements = (counter/threadsPerBlock1) + ((counter%threadsPerBlock1 == 0)?0:1);
			counter = numElements;
		}
		else
		{
			if(numElements > counter%threadsPerBlock1)
				numElements %= threadsPerBlock1;
			else
				counter = 0;
		}
		idFlag = 0;
	}

	cudaMemcpy(&(stats->maxScore),tempMax,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&(stats->minScore),tempMin,sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(&(stats->bestGenome),tempMaxId,sizeof(long int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&(stats->worstGenome),tempMinId,sizeof(long int),cudaMemcpyDeviceToHost);


	///////////////////////////////////////////////////////////////////
	/////////////////////      SINGLE STEP    /////////////////////////
	///////////////////////////////////////////////////////////////////

	while(stats->terminationFlag != TRUE)
	{

		/////////////////////////////////////////////////////////////////////
		/////////////	      CROSSOVER PREPROCESS      /////////////////////
		/////////////////////////////////////////////////////////////////////

		if(genome->xover == ONEPOINT)
		{

			int threadsPerBlock2 = 256;
			int numBlocks2 = ((ga->popsize)/threadsPerBlock2) + ((ga->popsize % threadsPerBlock2 == 0)? 0:1); 
			xOverPreProcess<<<numBlocks2,threadsPerBlock2>>>(devga,devgenome,devstats,rng,momID,dadID,xOverID);
			cudaThreadSynchronize();

			// EITHER DO XOVER & MUTATION SEPARATELY - UNCOMMENT THIS
			/*
			   int xDim = 256/genome->width;
			   int yDim = genome->width;
			   int numBlocks3 = ((ga->popsize)/xDim) + ((ga->popsize % xDim == 0)? 0:1);;
			   dim3 threadsPerBlock3(xDim,yDim);
			   xOnePoint1DBIN<<<numBlocks3,threadsPerBlock3>>>(devNewPop,devOldPop,devga,devgenome,devstats,rng,momID,dadID,xOverID);
			   cudaThreadSynchronize();*/
		}

		/*	else if other types to be added here	*/

		/////////////////////////////////////////////////////////////////////
		/////////////	        CROSSOVER PROCESS       /////////////////////
		/////////////////////////////////////////////////////////////////////

		if(genome->xover == ONEPOINT && ga->mutSeparate == TRUE)
		{
			int xDim = 256/genome->width;
			int yDim = genome->width;
			int numBlocks3 = ((ga->popsize)/xDim) + ((ga->popsize % xDim == 0)? 0:1);;
			dim3 threadsPerBlock3(xDim,yDim);
			xOnePoint1DBIN<<<numBlocks3,threadsPerBlock3>>>(devNewPop,devOldPop,devga,devgenome,devstats,rng,momID,dadID,xOverID);
			cudaThreadSynchronize();
		}

		/*	else if other types to be added here	*/

		/////////////////////////////////////////////////////////////////////
		/////////////     MUTATION  IN POPULATION       /////////////////////
		/////////////////////////////////////////////////////////////////////

		if(genome->mutation == FLIP && ga->mutSeparate == TRUE)
		{
			int xDim = 256/genome->width;
			int yDim = genome->width;
			int numBlocks = ((ga->popsize)/xDim) + ((ga->popsize % xDim == 0)? 0:1);;
			dim3 threadsPerBlock(xDim,yDim);
			flipMutate1DBIN<<<numBlocks,threadsPerBlock>>>(devNewPop,devOldPop,devga,devgenome,devstats,rng);
			cudaThreadSynchronize();
		}
		/*	else if other types to be added here	*/


		/////////////////////////////////////////////////////////////////////
		/////////////     XOVER & MUTATION IN ONE       /////////////////////
		/////////////////////////////////////////////////////////////////////

		if(genome->xover == ONEPOINT && genome->mutation == FLIP && ga->mutSeparate == FALSE)
		{
			int xDim = 256/genome->width;
			int yDim = genome->width;
			int numBlocks = ((ga->popsize)/xDim) + ((ga->popsize % xDim == 0)? 0:1);;
			dim3 threadsPerBlock(xDim,yDim);
			xOnePointmFlip1DBIN<<<numBlocks,threadsPerBlock>>>(devNewPop,devOldPop,devga,devgenome,devstats,rng,momID,dadID,xOverID);
			cudaThreadSynchronize();
		}

		/*	else if other types to be added here	*/

		/////////////////////////////////////////////////////////////////////
		////////////////	EVALUATE POPULATION 	/////////////////////
		/////////////////////////////////////////////////////////////////////

#ifdef _USEREVALFUNC_

		evaluate1DBINPopulationUser(devNewPop,ga,genome,stats,udata);

#else

		int threadsPerBlock1 = 16;
		int numBlocks1 = ((ga->popsize)/threadsPerBlock1) + ((ga->popsize % threadsPerBlock1 == 0)? 0:1); 
		evaluate1DBINPopulation<<<numBlocks1,threadsPerBlock1>>>(devNewPop,devga,devgenome,devstats,dudata);
		cudaThreadSynchronize();

#endif

		/////////////////////////////////////////////////////////////////////
		///////////////////	  UPDATE STATISTICS 	//////////////////////
		//////////////////////////////////////////////////////////////////////

		// GPU based 

		status = cudaMemcpy(tempMax,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToDevice);
		status = cudaMemcpy(tempMin,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToDevice);

		threadsPerBlock1 = 256;
		numBlocks1 = ((ga->popsize)/threadsPerBlock1) + ((ga->popsize % threadsPerBlock1 == 0)? 0:1);

		numElements = ga->popsize;
		counter = numElements;
		idFlag = 1;

		while(counter != 0)
		{
			rankStatistics<<<numBlocks1,threadsPerBlock1>>>(numElements,tempMax,tempMin,tempMaxId,tempMinId,idFlag);
			cudaThreadSynchronize();

			if((int)(counter/threadsPerBlock1) > 0)
			{
				numElements = (counter/threadsPerBlock1) + ((counter%threadsPerBlock1 == 0)?0:1);
				counter = numElements;
			}
			else
			{
				if(numElements > counter%threadsPerBlock1)
					numElements %= threadsPerBlock1;
				else
					counter = 0;
			}
			idFlag = 0;
		}

		stats->prevBest = stats->bestGenome;
		stats->prevWorst = stats->worstGenome;

		stats->prevMax = stats->maxScore;
		stats->prevMin = stats->minScore;
		stats->prevAve = stats->aveScore;
		
		cudaMemcpy(&(stats->maxScore),tempMax,sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(&(stats->minScore),tempMin,sizeof(float),cudaMemcpyDeviceToHost);
		cudaMemcpy(&(stats->bestGenome),tempMaxId,sizeof(long int),cudaMemcpyDeviceToHost);
		cudaMemcpy(&(stats->worstGenome),tempMinId,sizeof(long int),cudaMemcpyDeviceToHost);

//		BIN1D *tempgnm;
//		tempgnm = (BIN1D *)calloc((genome->width),sizeof(BIN1D));

/*		printf("\nIteration is : %d\nBest Score is : %f\nBest Genome is (%d):",stats->currIter,stats->maxScore,stats->bestGenome);
        cudaMemcpy(hostPop,devNewPop,(ga->popsize)*(genome->width)*sizeof(BIN1D),cudaMemcpyDeviceToHost);
		for(int tempijk=0;tempijk<genome->width;tempijk++)
			printf("%d",hostPop[((stats->bestGenome)*(genome->width))+ tempijk]);
		printf("\n");*/

		if(ga->elitism == TRUE)
		{
			int replaceID;
			int bestID;

			// DO ONLY IF BEST GENOME IN PREV GENERATION
			if(stats->maxScore < stats->prevMax)
			{
				bestID = stats->prevBest;
				stats->maxScore = stats->prevMax;
				// DO IT ACCORDING TO REPLCMENT POLICY
				if(ga->replace == RANDOM)
				{
					replaceID = rand()%(ga->popsize);
				}
				stats->scores[replaceID] = stats->maxScore;
				stats->bestGenome = replaceID;
				//printf("\n%d",replaceID);
	
//				bestOff = bestID*(genome->width)*sizeof(BIN1D);
//				replaceOff = replaceID*(genome->width)*sizeof(BIN1D);
				cudaMemcpy(&(devNewPop[replaceID*(genome->width)]),&(devOldPop[bestID*(genome->width)]),(genome->width)*sizeof(BIN1D),cudaMemcpyDeviceToDevice);
			}
		}

        size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
        status = cudaMemcpy(devOldPop,devNewPop,size,cudaMemcpyDeviceToDevice);

/*        printf("\nAfter Elitism");
 		printf("\nIteration is : %d\nBest Score is : %f\nBest Genome is (%d):",stats->currIter,stats->maxScore,stats->bestGenome); 
        cudaMemcpy(hostPop,devNewPop,(ga->popsize)*(genome->width)*sizeof(BIN1D),cudaMemcpyDeviceToHost);                          
 		for(int tempijk=0;tempijk<genome->width;tempijk++)                                                                         
 			printf("%d",hostPop[((stats->bestGenome)*(genome->width))+ tempijk]);                                                  
 		printf("\n");                                                                                                              */




		// Update termination //
		stats->currIter += 1;
		if(ga->termination == ITERATION)
		{
			if(stats->currIter >= stats->maxIter)
			{
				stats->terminationFlag = TRUE;
			}
		}

	}
	if(stats->printTimings == TRUE)
	{
		gettimeofday(&tv1,&tz1);
		end_seconds=tv1.tv_sec;
		end_micro_seconds=tv1.tv_usec;                    

		end_micro_seconds += end_seconds * 1000000;
		start_micro_seconds += start_seconds * 1000000;

		difference_micro_seconds = end_micro_seconds - start_micro_seconds;
		difference_seconds = difference_micro_seconds/1000000;
		difference_micro_seconds %= 1000000;

		printf("\nProgram took %lu.%06lu seconds to complete\n", difference_seconds,difference_micro_seconds);
	}
	
	size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
	status = cudaMemcpy(hostPop,devNewPop,size,cudaMemcpyDeviceToHost);

	(*popAdd) = hostPop;
	cudaFree(tempMax);
	cudaFree(tempMaxId);
	cudaFree(tempMin);
	cudaFree(tempMinId);
	cudaFree(devNewPop);
	cudaFree(devOldPop);
	cudaFree(devstats.scores);
	cudaFree(devstats.selProbs);
	cudaFree(rng.randGPU);
	cudaFree(momID);
	cudaFree(dadID);
	cudaFree(xOverID);
	return TRUE;
}


FLAG gaEvolvePopulation(void **addPop,GAContext *ga,GNMContext *genome,GAStats *stats,UDATA *udata)
{
	FLAG status;
	status = FALSE;
	srand(time(NULL));

	int gType = genome->type;
	int gDim = genome->dim;

	if(gType == BINARY && gDim == 1)
	{
		status = gaEvolve1DBINPopulation((BIN1D **)addPop,ga,genome,stats,udata);
	}
	/*	else if(gType == CHAR && gDim == 1)
		{

		}
		else if(gType == INT && gDim == 1)
		{

		}
		else if(gType == FLOAT && gDim == 1)
		{

		}
		else if(gType == BINARY && gDim == 2)
		{

		}
		else if(gType == CHAR && gDim == 2)
		{

		}
		else if(gType == INT && gDim == 2)
		{

		}
		else if(gType == FLOAT && gDim == 2)
		{

		}
		else if(gType == BINARY && gDim == 2)
		{

		}
		else if(gType == CHAR && gDim == 2)
		{

		}
		else if(gType == INT && gDim == 2)
		{

		}
		else if(gType == FLOAT && gDim == 2)
		{

		}						*/
	return status;
}
