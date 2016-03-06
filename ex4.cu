/* This is an example of finding a 1D binary string with alternate 0 & 1  */
/* This can be used as a template example for any pattern finding program  */
/* You can easily modify the fitness function to find maximas & minimas  */
/* See the header file: you need to define UDATA even if you are not passing any data*/

#include <stdio.h>

#include "ex4.h"
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
	GAPrintParameters(&ga,&genome,&stats);	
	gaEvolvePopulation(&population,&ga,&genome,&stats,&udata);
	Print1DBINSolution(population,&genome,&stats);

	GAdestroyContexts(&ga,&genome,&stats);
	
	return 0;
}

__device__ float FitnessFunc(BIN1D *g,GNMContext genome,UDATA *udata)
{
//	int ctr = 0;
	int i = 0;
	int bp = 0;

	int sign_mult_x = 1;
	int sign_mult_y = 1;

	float x = 0.0;
	float y = 0.0;
	float score = 0.0;
	float p;

	bp = (int)(genome.width/2);
//	printf("\nbp is %d",bp);

	if(g[0] == 1)
		sign_mult_x = -1;
	
	if(g[bp] == 1)
		sign_mult_y = -1;
	
//	printf("\nGenome x is:");	
	for(i=1;i<bp;i++)
	{     
//		  printf("%d",g[i]);
		  p = 3 - i;
		  x += (g[i]*pow(2,p));
	}
 
	x*= sign_mult_x;
// 	printf(":\t%f\n",x);

//	printf("\nGenome y is:");	     
	for(i=bp + 1;i<genome.width;i++) 
	{                                
//		printf("%d",g[i]);           
		p = 3 + bp -i;              
//		printf("(%d)",(int)p);
		y += (g[i]*pow(2,p));               
	}                                
	y*= sign_mult_y;                 
//    printf(":\t%f\n",y);   

	if(x > 5 || y > 5 || x < -5|| y < -5)
	{
		score = -100000000;
	}
	else
	{
		score = pow((1-x),2);
		score+=(100*pow((y - pow(x,2)),2));
//		score = ((1-x)*(1-x)) + (100*(y - (x*x))*(y - (x*x)));
		if(score > 0.0)
			score *= -1;
	}
/*	if(score == 0.0)
	{
		printf("\nScore is %f\nGenome is:",score);
		for(i=0;i<(genome.width);i++)
		printf("%d",g[i]);
		printf("\nx = %f,y = %f",x,y);
	}*/
	return score;
}

