/**
 File name: bc_bottom_up_gpu.cu
 Author: Yuede Ji
 Last update: 1:15 10-28-2015
 Description: GPU bc on small graph
    () read begin position, csr, weight value from binary file
    (2) betweenness centrality
    (3) atomic lock

**/

#ifndef __BC_GPU_H__
#define __BC_GPU_H__

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "wtime.h"
#include "graph.h"
#include "../util.h"
//#include "apsp_gpu_multi.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
//{
//   if (code != cudaSuccess) 
//   {
//      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//      if (abort) exit(code);
//   }
//}
//const char output_file[] = "bc_bottom_up_gpu.bc";
//const char sp_count_file[] = "bc_bottom_up_gpu.sp_count";
//const char dist_file[] = "bc_bottom_up_gpu.dist";
//const char sa_file[] = "bc_bottom_up_gpu.sa";

//path_t bc_global[V*V];

__global__ void bc_merge(path_t * dev_bc, index_t v_count)
{
    int id = threadIdx.x;
    int bc_id = id + blockDim.x;
    int limit = blockDim.x * blockDim.x;
    if(id < v_count)
    {
        while(bc_id < limit)
        {
            dev_bc[id] += dev_bc[bc_id];
            bc_id += blockDim.x;
        }
    //__syncthreads();
    }
}

__global__ void bc_all(index_t * dev_beg_pos, 
        index_t * dev_csr, 
        path_t *dev_weight, 
        path_t * dev_bc,
        index_t v_count)
{
    index_t dest = threadIdx.x;
    index_t root = blockIdx.x;
    __shared__ int shared_sp_count[V];
    __shared__ path_t shared_dist[V];
    index_t id = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ path_t shared_bc_tmp[V];
    __shared__ int level;
    
    shared_dist[dest] = INF;
    __shared__ index_t shared_sa[V];
    shared_sp_count[dest] = 0;
    __shared__ bool flag;
    shared_bc_tmp[dest] = 0;
    shared_sa[dest] = INF;
    
	if(dest == root)
    {
        shared_dist[root] = 0;
        shared_sp_count[root] = 1;
        level = 0;
        flag = true;
        shared_sa[root] = 0;
    }
    
    __syncthreads();

    //Step 1: sssp 
    int iteration=0;
    
    if(dest < v_count)
    {
        while(flag)
        {
            __syncthreads();///the problem!
            bool flag_one = false;
            ++iteration;
            //__syncthreads();
            for(index_t j=dev_beg_pos[dest]; j<dev_beg_pos[dest+1]; ++j)
            {
                index_t nebr=dev_csr[j];
                if(shared_dist[dest] > shared_dist[nebr] + dev_weight[j] && shared_sa[nebr] < iteration)
                {
                    shared_sa[dest] = iteration;
                    shared_dist[dest] = shared_dist[nebr] + dev_weight[j];
                    shared_sp_count[dest] = 0;
                    //level = iteration;
                    flag_one = true;
                }
            }
            //if(flag)
            //if(flag)
            flag = false;
            __syncthreads(); 
            if(flag_one)
            {
                //if(!flag)
                //    flag = true;
                for(index_t j=dev_beg_pos[dest]; j<dev_beg_pos[dest+1]; ++j)
                {
                    index_t nebr=dev_csr[j];
                    if(shared_dist[dest] == shared_dist[nebr] + dev_weight[j]) 
                    {
                        shared_sp_count[dest] += shared_sp_count[nebr];
                    }
                }
                if(!flag)
                    flag = true;
            }
            __syncthreads();
        }

        level = iteration;
        //__syncthreads();

        //printf("\n");
       //Step 2: bc_one
        //printf("%u %d\n", root, level);
        //int cur = level;
       
        while(level>=0)
        {
            if(shared_sa[dest] == level)
            {
                for(index_t j=dev_beg_pos[dest]; j<dev_beg_pos[dest+1]; ++j)
                {
                    if(shared_dist[dev_csr[j]] == shared_dist[dest] + dev_weight[j])
                    {
                        //if(shared_sp_count[dev_csr[j]] != 0)
                        shared_bc_tmp[dest] += shared_sp_count[dest]*(1.0 + shared_bc_tmp[dev_csr[j]])/(shared_sp_count[dev_csr[j]]);
                    }
                }
            }
            __syncthreads();
            if(dest == 0)
            {
                level = level - 1;
            }
            
            //--cur;
            __syncthreads();
        }
        
        //__syncthreads();
        if(dest == root)
            shared_bc_tmp[root] = 0;
        dev_bc[id] = shared_bc_tmp[dest];
    }
}



void bc_gpu_launch(graph *g, path_t *bc_global)
{
    index_t v = g->vert_count;
    index_t e = g->edge_count;
    
    index_t *dev_beg_pos;
    vertex_t *dev_csr;
    path_t *dev_weight;
//    path_t *dev_bc;
    
//    index_t *dev_sa_global;
//    int *dev_sp_count_global;
    path_t *dev_bc_global;
//    path_t *dev_dist_global;


    cudaMalloc( (void **) &dev_beg_pos, (v+1)*sizeof(index_t));
    cudaMalloc( (void **) &dev_csr, e*sizeof(vertex_t));
    cudaMalloc( (void **) &dev_weight, e*sizeof(path_t));
//    cudaMalloc( (void **) &dev_bc, V*V*sizeof(path_t));
//
//    cudaMalloc( (void **) &dev_sa_global, V*V*sizeof(index_t));
//    cudaMalloc( (void **) &dev_sp_count_global, V*V*sizeof(int));
//    cudaMalloc( (void **) &dev_dist_global, V*V*sizeof(path_t));
    cudaMalloc( (void **) &dev_bc_global, v*v*sizeof(path_t));

    
    cudaMemcpy(dev_beg_pos, g->beg_pos, (v+1)*sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csr, g->csr, e*sizeof(index_t),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weight, g->weight, e*sizeof(path_t),cudaMemcpyHostToDevice);
    cudaMemset(dev_bc_global, 0, v*v*sizeof(path_t));
    
//    cudaMemset(dev_sa_global, 0, V*V*sizeof(index_t));
//    cudaMemset(dev_sp_count_global, 0, V*V*sizeof(int));
//    cudaMemset(dev_dist_global, 0, V*V*sizeof(path_t));
//    cudaMemset(dev_bc_global, 0, V*V*sizeof(index_t));
    
//    double t_sum = 0;
    double btm = wtime();

    bc_all<<<v, v>>>(dev_beg_pos, dev_csr, dev_weight, dev_bc_global, v);
    cudaDeviceSynchronize();
//    gpuErrchk(cudaPeekAtLastError());
    
//    std::cout<<"BC calculation time (ms),"<<(wtime()-btm) * 1000<<"\n";
    
    bc_merge<<<1, v>>>(dev_bc_global, v);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
//    std::cout<<"BC kernel time (ms),"<<(wtime()-btm) * 1000<<"\n";

//    t_sum = wtime() - btm;
//    FILE * fp_time = fopen("time.info", "a");
//    fprintf(fp_time, "%g\n", t_sum);
//    fclose(fp_time);

    
//    cudaMemcpy(sa_global, dev_sa_global, V*V*sizeof(index_t), cudaMemcpyDeviceToHost);
//    cudaMemcpy(sp_count_global, dev_sp_count_global, V*V*sizeof(int), cudaMemcpyDeviceToHost);
//    cudaMemcpy(dist_global, dev_dist_global, V*V*sizeof(path_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(bc_global, dev_bc_global, v*v*sizeof(path_t), cudaMemcpyDeviceToHost);

//    cudaFree(dev_sa_global);
    cudaFree(dev_beg_pos);
    cudaFree(dev_csr);
    cudaFree(dev_weight);
    cudaFree(dev_bc_global);
}

#endif
