#ifndef __UTIL_H__
#define __UTIL_H__
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>

#define LOCK(vert, lock) while(!__sync_bool_compare_and_swap(lock+vert,0,-1))
#define UNLOCK(vert, lock) lock[vert]=0
typedef long index_t;
typedef long vertex_t;
typedef double path_t;
typedef long depth_t;

#define INFTY (float)10000000 
#define ORPHAN	(unsigned char)254
#define UNVIS		(long)-1

const int INF = 0x7fffffff;
const int V = 32;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline off_t fsize(const char *filename) {
	struct stat st; 
	if (stat(filename, &st) == 0)
		return st.st_size;
	return -1; 
}

#endif
