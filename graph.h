#ifndef __GRAPH_H__
#define __GRAPH_H__
#include "../util.h"
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "wtime.h"
class graph
{
	public:
		index_t *beg_pos;
		vertex_t *csr;
		path_t *weight;
		index_t vert_count;
		index_t edge_count;

	public:
		graph(){};
		~graph(){};
		graph(const char *graph_file);
};
#endif
