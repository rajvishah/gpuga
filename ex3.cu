/* This is an example of 0-1 Knapsack problem solved using Genetic algorithms */
/* This example makes use of user's own evaluation function */
/* Advanced users can write their own evaluation function exploiting GPU usage */
/* usereval.cu : is just a sample for guidelines and cannot be copied for any problem */


#include <stdio.h>
#include "ex3.h"
#include "usereval.cu"
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




