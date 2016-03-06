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


void updatePopulationStatistics(GAStats *stats,GAContext *ga)
{
	
	//int maxFlag=0,minFlag=0,min2Flag=0;
	stats->prevBest = stats->bestGenome;
	stats->prevWorst = stats->worstGenome;
	
	stats->prevMax = stats->maxScore;
	stats->prevMin = stats->minScore;
	stats->prevAve = stats->aveScore;

	// The following has to be changed according to sel scheme
	float fitnessSum = 0.0;	
	float max1=0.0;
	float max2=0.0;
	float min1;
	min1 = stats->scores[0];
	stats->worstGenome = 0;
	

	for(int i=0;i<ga->popsize;i++)
	{
		if(stats->scores[i] >= max1)
		{
			max1 = stats->scores[i];
			stats->bestGenome = i;
			stats->maxScore = max1;
		}
		if(stats->scores[i] > max2 && stats->scores[i] < max1)
		{
			max2 = stats->scores[i];
		}
		if(stats->scores[i] <= min1)
		{
			stats->worst2Genome = stats->worstGenome;
			min1 = stats->scores[i];
			stats->worstGenome = i;
			stats->minScore = min1;
		}
		
		fitnessSum += stats->scores[i];
	}

	stats->aveScore = fitnessSum/(ga->popsize);

	// If selection scheme = UNIFORM do nothing
	//memcpy(&stats->selProbs,&stats->scores,(ga->popsize)*sizeof(float));

	// If selection scheme = ROULETTE WHEEL
	// selProbs = Prefix sum of score/FitnessSum

	// If selection scheme = RANK

	stats->currIter += 1;

	if(ga->termination == ITERATION)
	{
		if(stats->currIter > stats->maxIter)
		{
			stats->terminationFlag = TRUE;
		}
	}
	/* else if other replacement schemes here */
}



FLAG gaEvolve1DBINPopulation(BIN1D **popAdd,GAContext *ga,GNMContext *genome,GAStats *stats,UDATA *udata)
{
	cudaError_t status;
	size_t size;
		
	BIN1D *hostPop;
	BIN1D *hostNewPop;
	BIN1D *devOldPop;
	BIN1D *devNewPop;

	GAContext devga;
	GNMContext devgenome;
	GAStats devstats;

	struct timeval tv1;
	struct timezone tz1;
	unsigned long start_seconds,cpu_start_seconds,nu_start_seconds,u_start_seconds;
	unsigned long start_micro_seconds,cpu_start_useconds,nu_start_useconds,u_start_useconds;
	unsigned long end_seconds,cpu_end_seconds,nu_end_seconds,u_end_seconds;
	unsigned long end_micro_seconds,cpu_end_useconds,nu_end_useconds,u_end_useconds;
	unsigned long difference_seconds = 0;
	unsigned long difference_micro_seconds = 0; 
	unsigned long cpu_seconds = 0;
	unsigned long cpu_useconds = 0;
        unsigned long nu_seconds = 0;
        unsigned long nu_useconds = 0;
        unsigned long u_seconds = 0;
        unsigned long u_useconds = 0;

	float percent_cpu = 0.0;

	copyGAStructures(ga,genome,stats,&devga,&devgenome,&devstats);

	Construct1DBINPopulation(&hostPop,ga,genome);
	Construct1DBINPopulation(&hostNewPop,ga,genome);
	
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

	randLimit = 2*(ga->popsize)*(genome->width);

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

	// ON HOST selProbs
	(stats->selProbs) = (float *)calloc(ga->popsize,sizeof(float));
	if(stats->selProbs == NULL)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for sel probs on host");
		return FALSE;
	}

	// ON DEVICE
	status = cudaMalloc((void**)&devstats.scores,(ga->popsize)*sizeof(float));
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for scores on device");
		free(stats->scores);
		return FALSE;
	}	

	// ON DEVICE
	status = cudaMalloc((void**)&devstats.selProbs,(ga->popsize)*sizeof(float));
	if(status != cudaSuccess)
	{
		DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not allocate memory for scores on device");
		free(stats->scores);
		return FALSE;
	}	

	
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


	int threadsPerBlock1 = 16;
	int numBlocks1 = ((ga->popsize)/threadsPerBlock1) + ((ga->popsize % threadsPerBlock1 == 0)? 0:1); 
	evaluate1DBINPopulation<<<numBlocks1,threadsPerBlock1>>>(devOldPop,devga,devgenome,devstats,dudata);

	cudaThreadSynchronize();


        //  COPY SCORES TO HOST
        status = cudaMemcpy(stats->scores,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToHost);
        if(status != cudaSuccess)
        {
                DPRINTF("\n0 In Function gaEvolve1DBINPopulation:: Could not copy scores on device");
                return FALSE;
        }

	#endif


        //////////////////////////////////////////////////////////////////////
	///////////////////	  UPDATE STATISTICS 	//////////////////////
	//////////////////////////////////////////////////////////////////////

        updatePopulationStatistics(stats,ga);	

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
		/////////////	      CROSSOVER PREPROCESS      /////////////////////
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


	        if(stats->printTimings == TRUE)
        	{
                	gettimeofday(&tv1,&tz1);
	                cpu_start_seconds=tv1.tv_sec;
        	        cpu_start_useconds=tv1.tv_usec;
	        }
		
		//  COPY SCORES TO HOST
		status = cudaMemcpy(stats->scores,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToHost);
		if(status != cudaSuccess)
		{
		        DPRINTF("\n1 In Function gaEvolve1DBINPopulation:: Could not copy scores on device");
		        return FALSE;
		}

		#endif


		/////////////////////////////////////////////////////////////////////
		////////////////	POP & SCORES TO CPU 	/////////////////////
		/////////////////////////////////////////////////////////////////////

		//  COPY POPULATION TO HOST
		status = cudaMemcpy(hostNewPop,devNewPop,(genome->width)*(ga->popsize)*sizeof(float),cudaMemcpyDeviceToHost);
		if(status != cudaSuccess)
		{
		        DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not copy scores on device");
		        return FALSE;
		}

		/////////////////////////////////////////////////////////////////////
		///////////////////	  UPDATE STATISTICS 	//////////////////////
		//////////////////////////////////////////////////////////////////////

            	if(stats->printTimings == TRUE)
                {
                        gettimeofday(&tv1,&tz1);
                        u_start_seconds=tv1.tv_sec;
                        u_start_useconds=tv1.tv_usec;
                }


		updatePopulationStatistics(stats,ga);	

                // CPU Copy + Update Timings //
                if(stats->printTimings == TRUE)
                {
                        gettimeofday(&tv1,&tz1);
                        u_end_seconds=tv1.tv_sec;
                        u_end_useconds=tv1.tv_usec;

                        u_end_useconds += u_end_seconds * 1000000;
                        u_start_useconds += u_start_seconds * 1000000;
                        u_useconds += u_end_useconds - u_start_useconds;

                }

		// GPU based 

		float *tempMax,*tempMin;
		long int *tempMaxId,*tempMinId;
		float chkMax,chkMin;
		
	        status = cudaMalloc((void **)&tempMax,(ga->popsize)*sizeof(float));
        	status = cudaMalloc((void **)&tempMin,(ga->popsize)*sizeof(float));
		status = cudaMalloc((void **)&tempMaxId,(ga->popsize)*sizeof(long int)); 
		status = cudaMalloc((void **)&tempMinId,(ga->popsize)*sizeof(long int));

		status = cudaMemcpy(tempMax,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToDevice);
		status = cudaMemcpy(tempMax,devstats.scores,(ga->popsize)*sizeof(float),cudaMemcpyDeviceToDevice);

 		if(stats->printTimings == TRUE)
                {
                        gettimeofday(&tv1,&tz1);
                        nu_start_seconds=tv1.tv_sec;
                        nu_start_useconds=tv1.tv_usec;
                }

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
//        	cudaMemcpy((&(stats->maxScore),tempMax,sizeof(float),cudaMemcpyDeviceToHost);
//	        cudaMemcpy((&(stats->minScore)),tempMin,sizeof(float),cudaMemcpyDeviceToHost);
//        	cudaMemcpy((&(stats->bestGenome),tempMaxId,sizeof(long int),cudaMemcpyDeviceToHost);
//	        cudaMemcpy((&(stats->worstGenome),tempMinId,sizeof(long int),cudaMemcpyDeviceToHost);
//	      	printf("\nMax score : %f, max Id : %lu",max,maxId);

        



                // CPU Copy + Update Timings //
                if(stats->printTimings == TRUE)
                {
                        gettimeofday(&tv1,&tz1);
                        nu_end_seconds=tv1.tv_sec;
                        nu_end_useconds=tv1.tv_usec;

                        nu_end_useconds += nu_end_seconds * 1000000;
                        nu_start_useconds += nu_start_seconds * 1000000;
                        nu_useconds += nu_end_useconds - nu_start_useconds;

                }


                status = cudaMemcpy((&chkMax),tempMax,sizeof(float),cudaMemcpyDeviceToHost);
                status = cudaMemcpy((&chkMin),tempMin,sizeof(float),cudaMemcpyDeviceToHost);

		//printf("\nCuda calculated : %f",chkMax);
		//printf("\nActual : %f",stats->maxScore);

		cudaFree(tempMax);
      	      	cudaFree(tempMaxId);
	      	cudaFree(tempMin);
	      	cudaFree(tempMinId);

		if(ga->elitism == TRUE)
		{
			int replaceID;
			int bestID;
			
			// DO ONLY IF BEST GENOME IN PREV GENERATION
			if(stats->maxScore < stats->prevMax)
			{
				bestID = stats->prevBest;
				stats->maxScore = stats->prevMax;
				// REPLACE ACCORDING TO REPLACEMENT POLICY
				if(ga->replace == WORST)
				{
					replaceID = stats->worstGenome;
					stats->worstGenome = stats->worst2Genome;
					stats->minScore = stats->scores[stats->worstGenome];
				}
				else if(ga->replace == BEST)
				{
					replaceID = stats->bestGenome;
				}
				else
				{
					replaceID = rand()%(ga->popsize);
				}
				stats->scores[replaceID] = stats->maxScore;
				stats->bestGenome = replaceID;
				memcpy(&(hostNewPop[replaceID*(genome->width)]),&(hostPop[bestID*(genome->width)]),(genome->width)*sizeof(BIN1D));

				//printf("\nReplaced %d by %d",replaceID,bestID);


				//   COPY POPULATION TO DEVICE
				size = (ga->popsize)*(genome->width)*sizeof(BIN1D);
				status = cudaMemcpy(devOldPop,hostNewPop,size,cudaMemcpyHostToDevice);
				if(status != cudaSuccess)
				{
					DPRINTF("\nIn Function gaEvolve1DBINPopulation:: Could not copy population on device");
					return FALSE;
				}
			}
		
			
		}
		memcpy(hostPop,hostNewPop,(ga->popsize)*(genome->width)*sizeof(BIN1D));

		// CPU Copy + Update Timings //
	        if(stats->printTimings == TRUE)
        	{
                	gettimeofday(&tv1,&tz1);
	                cpu_end_seconds=tv1.tv_sec;
	                cpu_end_useconds=tv1.tv_usec;
			
			cpu_end_useconds += cpu_end_seconds * 1000000;
			cpu_start_useconds += cpu_start_seconds * 1000000;
	                cpu_useconds += cpu_end_useconds - cpu_start_useconds;
				
        	}

		
//		printf("\n\nIteration %d",stats->currIter);
//		Print1DBINPopulation(hostPop,ga,genome,stats);
//		printf("\nBest Genome %d: Fitness %f",stats->bestGenome,stats->maxScore);
		
	} 

	if(stats->printTimings == TRUE)
	{
		gettimeofday(&tv1,&tz1);
		end_seconds=tv1.tv_sec;
		end_micro_seconds=tv1.tv_usec;
                    
		end_micro_seconds += end_seconds * 1000000;
                start_micro_seconds += start_seconds * 1000000;
                difference_micro_seconds += end_micro_seconds - start_micro_seconds;


/*		if(start_micro_seconds>=end_micro_seconds)
		{
		difference_micro_seconds = start_micro_seconds-end_micro_seconds;
		difference_seconds = end_seconds-start_seconds-1;
		}
		else
		{
		difference_micro_seconds = end_micro_seconds-start_micro_seconds;
		difference_seconds = end_seconds - start_seconds;
		}	*/
		percent_cpu = ((float)(cpu_useconds)*100)/difference_micro_seconds;
		difference_seconds = difference_micro_seconds/1000000;
		difference_micro_seconds %= 1000000;
		cpu_seconds = cpu_useconds/1000000;
		cpu_useconds %= 1000000;
                printf("\nProgram took %lu.%06lu seconds to complete\n", difference_seconds,difference_micro_seconds);
		printf("\nProgram took %lu.%06lu seconds for cpu computation\n", cpu_seconds,cpu_useconds);
		printf("\nCPU took %2.2f time of total\n",percent_cpu);

		percent_cpu = ((float)(u_useconds))/nu_useconds;

	        u_seconds = u_useconds/1000000;
                u_useconds %= 1000000;
	        nu_seconds = cpu_useconds/1000000;
                nu_useconds %= 1000000;

		printf("\nProgram took %lu.%06lu seconds for old update\n", u_seconds,u_useconds);
		printf("\nProgram took %lu.%06lu seconds for new update\n", nu_seconds,nu_useconds);
		printf("\nNU is %2.2f times faster\n",percent_cpu);



	}

	(*popAdd) = hostPop;
	free(hostNewPop);
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
