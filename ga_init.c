#include <stdio.h>
#include <stdlib.h>
#include "ga_defs.h"

void GAdestroyContexts(GAContext *ga,GNMContext *genome,GAStats *stats)
{

	// If you add any pointers to GA structures , free it here
	if(NULL != stats->scores)
		free(stats->scores);

}

void InitGAContext(GAContext *ga)
{
	ga->popsize	= 100;
	ga->pMut 	= 0.1;
	ga->pCross 	= 0.9;
	ga->randSeed 	= 0;
	ga->elitism 	= TRUE;
	ga->selection 	= UNIFORMSEL;
	ga->replace 	= RANDOM;
	ga->termination = ITERATION;
	ga->mutSeparate = TRUE;
}

void InitGNMContext(GNMContext *genome)
{
	genome->dim 	= 1;
	genome->width 	= 50;
	genome->depth 	= 1;
	genome->height 	= 1;
	genome->type 	= BINARY;
	genome->InitMethod = UNIFORM;
	genome->xover = ONEPOINT;
	genome->mutation = FLIP;
}

void InitGAStatsContext(GAStats *stats)
{
	//stats->scores[ga->popsize] = {0.0};

	stats->maxScore = 0.0;
	stats->minScore = 0.0;
	stats->aveScore = 0.0;

	stats->bestGenome = -1;
	stats->worstGenome = -1;
	
	stats->currIter = 0;
	stats->maxIter = 100;

	stats->printTimings = TRUE;	
	stats->terminationFlag = FALSE;
}

void GAPrintParameters(GAContext *ga,GNMContext *genome,GAStats *stats)
{
	// Printing only basic ones
	printf("\n\n--------------");
	printf("\nGA Parameters:");
	printf("\n--------------");
	printf("\nPopulation	%d",ga->popsize);
	printf("\nGenome length	%d",genome->width);
	printf("\npMut 		%.2f",ga->pMut);
	printf("\npCross	%.2f",ga->pCross);
	printf("\nMax Iter	%d",stats->maxIter);
	printf("\n");
}

void GASetParameters(GAContext *ga,GNMContext *genome,GAStats *stats)
{
	char c;

	InitGAContext(ga);
	InitGNMContext(genome);
	InitGAStatsContext(stats);

	printf("\nKeep Default Parameters? [y/n]::");
	scanf("%c",&c);
	if(c == 'y')
	{
		return;
	}
	else
	{
		printf("\nEnter Genome Type \n[0 - Binary]\n[Others not added yet]\n>>");
		scanf("%d",&(genome->type));
		printf("\nEnter Genome Dimension \n[1 - 1D, 2- 2D, 3 - 3D]\n>>");
		scanf("%d",&(genome->dim));
		if(genome->dim == 1)
		{
			printf("\nEnter Genome Length \n>>");
			scanf("%d",&(genome->width));
		}
		else
		{
			printf("\nSupport for only 1D Binary added yet - Taking default dim 1");
			genome->dim = 1;
			printf("\nEnter Genome Length \n>>");
			scanf("%d",&(genome->width));
		}	
		printf("\nEnter Population Size \n>>");
		scanf("%d",&(ga->popsize));
		printf("\nEnter Population Initialization Method \n[0 - RESET]\n[1 - SET]\n[2 - UNIFORM]\n>>");
		scanf("%d",&(genome->InitMethod));	
		printf("\nEnter probability of crossover\n>>");
		scanf("%f",&(ga->pCross));
		printf("\nEnter probability of mutation\n>>");
		scanf("%f",&(ga->pMut));
		printf("\nEnter maximum iterations\n>>");
		scanf("%d",&(stats->maxIter));
		printf("\nPrint Timings after evolution?[y/n]::");
		c = getchar();
		scanf("%c",&c);
		if(c == 'n')
		{
			stats->printTimings = FALSE;
		}


		// Add options for these later
		printf("\nSELECTION METHOD\n>>Taken Default : Uniform");
		printf("\nCROSSOVER METHOD\n>>Taken Default : One Point");
		printf("\nMUTATION METHOD\n>>Taken Default : Flip Bit");
		printf("\nTERMINATION METHOD\n>>Taken Default : Upon Iterations");
	}
}	


