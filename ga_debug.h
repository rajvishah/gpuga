#ifndef DEBUG_H
#define DEBUG_H
#include <stdarg.h>
#include <stdio.h>

#ifndef DBGMODE
# define DPRINTF printf
# else
# define DPRINTF 1 ? 0 :
# endif

#ifndef EMUMODE
# define KDPRINTF printf
# else
# define KDPRINTF 1 ? 0 :
# endif

/*
#define LOC 
printf("debug:%s:%d:",__FILE__,__LINE__)
#define DPRINTF(fmt,...) printf(fmt,__VA_ARGS__)
#else
#define DPRINTF(fmt,...)
#endif*/



#endif /* DEBUG_H */

