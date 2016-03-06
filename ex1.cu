/* This is an example of finding a 1D binary string with alternate 0 & 1  */
/* This can be used as a template example for any pattern finding program  */
/* You can easily modify the fitness function to find maximas & minimas  */
/* See the header file: you need to define UDATA even if you are not passing any data*/

#include <stdio.h>

#include "ex1.h"
#include "ga_simplega_new.cu"
//#include "ga_simplega_old.cu"


int main()
{
	void *population;
	UDATA udata;

	GAContext ga;
	GNMContext genome;
	GAStats stats;

	GASetParameters(&ga,&genome,&stats);

	printf("\nEnter popsize:");
	scanf("%d",&(ga.popsize));
	printf("\nEnter width:");
	scanf("%d",&(genome.width));
//	printf("\nEnter iteration:");
//	scanf("%d",&(stats.maxIter));
	GAPrintParameters(&ga,&genome,&stats);	
	gaEvolvePopulation(&population,&ga,&genome,&stats,&udata);
	Print1DBINSolution(population,&genome,&stats);

	GAdestroyContexts(&ga,&genome,&stats);
	
	return 0;
}

__device__ float FitnessFunc(BIN1D *g,GNMContext genome,UDATA *udata)
{
	int ctr = 0;
	int i = 0;
	float score = 0.0;
	for(i=0;i<(genome.width);i++)
	{
		if(ctr%2 == 0 && g[i] == 0)
		{
			score += 1.0;
		}
		else if(ctr%2 != 0 && g[i] == 1)
		{
			score += 1.0;
		}
		ctr++;
	}
	return score;
}

