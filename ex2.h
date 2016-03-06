//#define _UDATAEXIST_
#define PSIZE 50
#include "ga_defs.h"
struct _udata
{
	int weights[PSIZE];
	int values[PSIZE];
	int maxWeight;
};
typedef struct _udata UDATA;

__device__ float FitnessFunc(BIN1D *g,GNMContext genome,UDATA *udata);

