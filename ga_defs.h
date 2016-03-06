#ifndef _gadefsheader_
#define _gadefsheader_

// DATA-TYPE ALIAS //
typedef unsigned int BIN1D;

// ENUM DATA TYPES DEFINITIONS //

enum _bool
{
	FALSE = 0,
	TRUE
};
typedef enum _bool FLAG;	

enum _type
{
	BINARY = 0,
	CHAR,
	INT,
	REAL
};
typedef enum _type Type;

enum _genomeInit
{
	RESET = 0,
	SET,
	UNIFORM
};
typedef enum _genomeInit GNM_INIT;

enum _genomeXover
{
	ONEPOINT = 0,
	TWOPOINT,
	ARITHMATIC
};
typedef enum _genomeXover GNM_XOVER;

enum _genomeMut
{
	FLIP = 0,
	SWAP,
	GAUSSIAN
};
typedef enum _genomeMut GNM_MUTATE;

enum _replacement
{
	BEST = 0,
	WORST,
	RANDOM
};
typedef enum _replacement GNM_REPLACE;

enum _selection
{
	UNIFORMSEL = 0,
	RANKSEL ,
	ROULETTESEL
};
typedef enum _selection GNM_SELECT;

enum _termination
{
	ITERATION = 0,
	CONVERGENCE,
};
typedef enum _termination GA_TERMINATE;


// CONTEXT STRUCTURES DEFINITIONS //

struct _GAContext
{
	int randSeed;
	
	long int popsize;
	float pMut;
	float pCross;

	FLAG elitism;
	FLAG mutSeparate;
	GNM_REPLACE replace;
	GNM_SELECT selection;
	GA_TERMINATE termination;		
};
typedef struct _GAContext GAContext;

struct _GenomeContext
{
	int dim;
	int width;
	int height;
	int depth;
	Type type;
	GNM_INIT InitMethod;
	GNM_XOVER xover;
	GNM_MUTATE mutation;
};

typedef struct _GenomeContext GNMContext;

struct _GAStats
{
	float *scores;
	float *selProbs;
	
	float prevMax;
	float prevMin;
	float prevAve;

	float maxScore;
	float minScore;
	float aveScore;

	int prevBest;
	int prevWorst;
	
	int bestGenome;
	int best2Genome;
	int worstGenome;
	int worst2Genome;

	int currIter;
	int maxIter;

	FLAG printTimings;
	FLAG terminationFlag;
};
typedef struct _GAStats GAStats;

// Structure for RANDOM NUM
struct _randNOS
{
	long int *randGPU;
	long int *randID;
	long int RMAX;
	long int RPERIOD;
};
typedef struct _randNOS GARand;

#endif
