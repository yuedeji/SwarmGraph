/**
 File name: apsp_gpu.cu
 Author: Yuede Ji
 Last update: 13:15 11-16-2015
 Description: GPU apsp on small graph
    (1) read begin position, csr, weight value from binary file
    (2) all-pairs shortest path

**/
#ifndef __APSP_GPU_H__
#define __APSP_GPU_H__


#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "wtime.h"
#include "graph.h"
#include "cc_gpu.h"
//#include "util.h"

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//const char output_file[] = "apsp_gpu.result";
//
//const int INF = 0x7fffffff;
//const int V = 218;
//
//path_t dist[V*V];

//__global__ void apsp(index_t * dev_beg_pos, 
//        index_t * dev_csr, 
//        path_t *dev_weight, 
//        path_t * dev_dist,
//        int root) 
//{
//    __shared__ path_t shared_dist[V];
//    __shared__ bool flag;
//
//    //index_t root = blockIdx.x;
//    index_t tmp_id = threadIdx.x;
//    while(tmp_id < V)
//    {
//        shared_dist[tmp_id] = INF;
//        tmp_id += blockDim.x;
//    }
//    __syncthreads();
//
//    if(threadIdx.x == 0)
//    {
//        flag = true;
//        shared_dist[root] = 0;
//    }
//    __syncthreads();
//    
//    while(flag)
//    {
//        __syncthreads();
//        index_t dest = threadIdx.x;
//        flag = false;
//        __syncthreads();///the problem!
//        //Step 1: sssp 
//        while(dest < V)
//        {
//            for(index_t j=dev_beg_pos[dest]; j<dev_beg_pos[dest+1]; ++j)
//            {
//                index_t nebr=dev_csr[j];
//                if(shared_dist[dest] > shared_dist[nebr] + dev_weight[j])
//                {
//                    shared_dist[dest] = shared_dist[nebr] + dev_weight[j];
//                    flag = true;
//                }
//            }
//            __syncthreads();
//            dest += blockDim.x;
//        }
//    }
//    
//    tmp_id = threadIdx.x;
//    while(tmp_id < V)
//    {
//        dev_dist[tmp_id+blockIdx.x * blockDim.x] = shared_dist[tmp_id];
//        tmp_id += blockDim.x;
//    }
//    __syncthreads();
//}



void apsp_gpu_launch(graph *g, path_t *dist)//, int start_vert, int end_vert)
{
//    std::cout<"start\n";
    index_t v = g->vert_count;
    index_t e = g->edge_count;
    
    index_t *dev_beg_pos;
    vertex_t *dev_csr;
    path_t *dev_weight;
    path_t *dev_dist;
//    cudaStream_t streams[4];
//    for(int i = 0; i < 4; ++i)
//        cudaStreamCreate(&streams[i]);
//    int result = cudaStreamCreate(&stream_1);

    double btm = wtime();
//    cudaMallocAsync( (void **) &dev_beg_pos, (v+1)*sizeof(index_t), stream_1);
    cudaMalloc( (void **) &dev_beg_pos, (v+1)*sizeof(index_t));
    cudaMemcpy(dev_beg_pos, g->beg_pos, (v+1)*sizeof(index_t), cudaMemcpyHostToDevice);

    cudaMalloc( (void **) &dev_csr, e*sizeof(vertex_t));
    cudaMemcpy(dev_csr, g->csr, e*sizeof(index_t),cudaMemcpyHostToDevice);

    cudaMalloc( (void **) &dev_weight, e*sizeof(path_t));
    cudaMemcpy(dev_weight, g->weight, e*sizeof(path_t),cudaMemcpyHostToDevice);

    cudaMalloc( (void **) &dev_dist, v*v*sizeof(path_t));
//    double malloc_time = wtime();
    
    cudaMemset(dev_dist, 0, v*v*sizeof(path_t));
    double memcpy_time = wtime();
    
//    printf("malloc time %lf (ms), memcpy time %lf (ms)\n", (malloc_time - btm) * 1000, (memcpy_time - btm) * 1000);
//    printf("copy time %lf (ms)\n", (memcpy_time - btm) * 1000);
//    printf("V = %d, vert_count = %d\n", V, v);
//    if(v != V)
//        return;
//    cudaDeviceSynchronize();
    btm = wtime();
    double t_sum = 0;
//    printf("before kernel\n");
    apsp_all<<<v, v>>>(dev_beg_pos, dev_csr, dev_weight, dev_dist, v);
//    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();

//    for(int root=0; root<V; ++root)
//    {
//        apsp<<<1, V>>>(dev_beg_pos, dev_csr, dev_weight, dev_dist, root);
//        cudaDeviceSynchronize();
//    }
    t_sum = wtime() - btm;
//    std::cout<<"APSP kernel time (ms), "<<(wtime()-btm) * 1000<<"\n";
//    FILE * fp_time = fopen("time.info", "a");
//    fprintf(fp_time, "%g\n", t_sum);
//    fclose(fp_time);
    
//    double before_copy_time = wtime();
    cudaMemcpy(dist, dev_dist, v*v*sizeof(path_t), cudaMemcpyDeviceToHost);

//    for(int i = 0; i < g->vert_count; ++i)
//    {
//        printf("%lf,", dist[i]);
//    }
//    double after_copy_time = wtime();
//    printf("copy back time (ms), %lf\n", (after_copy_time - before_copy_time) * 1000);
//
    cudaFree(dev_beg_pos);
    cudaFree(dev_csr);
    cudaFree(dev_weight);
    cudaFree(dev_dist);
}

#endif
