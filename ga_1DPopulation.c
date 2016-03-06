#include <stdio.h>
#include <stdlib.h>
#include "ga_init.c"
#include "ga_helpers.c"
#include "ga_debug.h"


void copyGAStructures(GAContext *gaA,GNMContext *genomeA,GAStats *statsA,GAContext *gaB,GNMContext *genomeB,GAStats *statsB)
{
	gaB->popsize	= gaA->popsize;
	gaB->pMut 	= gaA->pMut;
	gaB->pCross 	= gaA->pCross;
	gaB->randSeed 	= gaA->randSeed;
	gaB->elitism 	= gaA->elitism;
	gaB->selection 	= gaA->selection;
	gaB->replace 	= gaA->replace;
	gaB->termination = gaA->termination;

	genomeB->dim 	= genomeA->dim;
	genomeB->width 	= genomeA->width;
	genomeB->depth 	= genomeA->depth;
	genomeB->height = genomeA->height;
	genomeB->type 	= genomeA->type;
	genomeB->InitMethod =genomeA->InitMethod;

	statsB->maxScore = statsA->maxScore;
	statsB->minScore = statsA->minScore;
	statsB->aveScore = statsA->aveScore;

	statsB->bestGenome = statsA->bestGenome;
	statsB->worstGenome = statsA->worstGenome;
	
	statsB->currIter = statsA->currIter;
	statsB->maxIter = statsA->maxIter;

}

void Construct1DBINPopulation(BIN1D **pop,GAContext *ga,GNMContext *genome)
{
	BIN1D *population;
	int popsize = ga->popsize;
	int g_len = genome->width;
	int memsize = popsize*g_len*sizeof(BIN1D);
 	population = (BIN1D *)malloc(memsize);
	(*pop) = population;
}

void Init1DBINGenome(BIN1D *g,GNMContext *genome)
{	
	int j;
	for(j=0;j<genome->width;j++)
	{
		if(genome->InitMethod == RESET)
			g[j] = 0;
		else if(genome->InitMethod == SET)
			g[j] = 1;
		else
			g[j] = GAFlipCoin(0.5);
	}
}


void DebugInit1DBINGenome(BIN1D *g,GNMContext *genome,int value)
{	
	int j;
	for(j=0;j<genome->width;j++)
	{
		g[j] = value;
	}
		
}

void Init1DBINPopulation(BIN1D *population,GAContext *ga,GNMContext *genome)
{
	int i;
	BIN1D *g;
	for(i=0;i<ga->popsize;i++)
	{
		g = &population[i*(genome->width)];
//		DebugInit1DBINGenome(g,genome,i);
		Init1DBINGenome(g,genome);
	}
}



void Print1DBINGenome(BIN1D *g,GNMContext *genome)
{
	int i;
	for(i=0;i<genome->width;i++)
	DPRINTF("%d",g[i]);
	DPRINTF("\n");
}


void Print1DBINPopulation(BIN1D *population,GAContext *ga,GNMContext *genome,GAStats *stats)
{
	int i;
	BIN1D *g;
	for(i=0;i<ga->popsize;i++)
	{
		DPRINTF("\nPrinting genome %d/%d :: Score %f \n",i,ga->popsize,stats->scores[i]);
		g = &population[i*(genome->width)];
		Print1DBINGenome(g,genome);
	}
	//DPRINTF("\nDone");
}

void Print1DBINSolution(void *population,GNMContext *genome,GAStats *stats)
{
	BIN1D *pop;
	pop = (BIN1D *)population;
	int id = stats->bestGenome;
	float score = stats->maxScore;

	BIN1D *sol;
	sol = &pop[id*(genome->width)];
	printf("\nBest Score: %f\nBest Genome:\n",score);
	Print1DBINGenome(sol,genome);
}


	
