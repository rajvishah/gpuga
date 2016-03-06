/* This is an example of 0-1 Knapsack problem solved using Genetic algorithms */
/* This example makes use of traditional flow and define a serial fitness function */
/* To know how to use your own evaluate Function - see ex3.cu			*/

#include <stdio.h>
#include "ex2.h"
#include "ga_simplega_new.cu"


int main()
{

	void *population;
	UDATA udata;

	GAContext ga;
	GNMContext genome;
	GAStats stats;


	GASetParameters(&ga,&genome,&stats);
	GAPrintParameters(&ga,&genome,&stats);	


	for(int i=0;i<genome.width;i++)
	{
		if(i%2 == 0)
		{
			udata.weights[i] = 20;
			udata.values[i] = 20;
		}
		else
		{
			udata.weights[i] = 1;
			udata.values[i] = 1;
		}
		//udata.values[i] = (genome.width - i);
		udata.maxWeight = 2000;
//		printf("\n%d",genome.width);
	}

	gaEvolvePopulation(&population,&ga,&genome,&stats,&udata);
	Print1DBINSolution(population,&genome,&stats);

	GAdestroyContexts(&ga,&genome,&stats);
	
	return 0;
}

__device__ float FitnessFunc(BIN1D *g,GNMContext genome,UDATA *udata)
{
	int i;
	int weight= 0;
	int value = 0;
	float score = 0.0;
	for(i=0;i<(genome.width);i++)
	{
		if(g[i] == 1)
		{
			weight = weight + (udata->weights[i]);
			value = value + (udata->values[i]);
		} 
	}
	if(weight <= (udata->maxWeight))
	{
		score = 1.0 * value;
	}
	else
	{
		score = 0.0;
	}
	return score;
}

