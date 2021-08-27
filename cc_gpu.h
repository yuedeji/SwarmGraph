/**
 File name: cc_gpu.cu
 Author: Yuede Ji
 Last update: 14:20 11-16-2015
 Description: GPU cc on small graph
    (1) read begin position, csr, weight value from binary file
    (2) closeness centrality

**/
#ifndef __CC_GPU_H__
#define __CC_GPU_H__

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include "wtime.h"
#include "graph.h"

using namespace std;

//const char output_file[] = "cc_gpu.result";
//
//const int INF = 0x7fffffff;
//const int V = 218;
//
//path_t dist[V*V];
//path_t global_cc[V];
//
//void print_debug()
//{
//    FILE * fp = fopen(output_file, "w");
//    for(int i=0; i<V; ++i)
//    {
//        fprintf(fp, "%g\n", global_cc[i]);
//    }
//    fclose(fp);
//}
__global__ void cc_merge(path_t *dev_dist,
        path_t *dev_cc,
        index_t v_count)
{
    int thread_id = threadIdx.x;
    path_t sum = 0;
    for(index_t i=thread_id*v_count; i<thread_id*v_count+v_count; ++i)
    {
        if(dev_dist[i] < INF && dev_dist[i] > 0)
            sum += 1.0/dev_dist[i];
    }
    //printf("%g\n", sum);
    dev_cc[thread_id] = sum;
}

__global__ void apsp_all(index_t * dev_beg_pos, 
        index_t * dev_csr, 
        path_t *dev_weight, 
        path_t * dev_dist,
        index_t v_count) 
{
    __shared__ path_t shared_dist[V];
    __shared__ bool flag;

    index_t root = blockIdx.x;
    //index_t id = threadIdx.x + blockIdx.x * blockDim.x;
    index_t tmp_id = threadIdx.x;
//    printf("blockIdx.x = %d, blockDim.x = %d, threadIdx.x = %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    
    if(root < v_count && tmp_id < v_count)
    {
        while(tmp_id < v_count)
        {
            shared_dist[tmp_id] = INF;
            tmp_id += blockDim.x;
        }
        __syncthreads();
        
        if(threadIdx.x == 0)
        {
            shared_dist[root] = 0;
            flag = true;
        }
        
        __syncthreads();

        //Step 1: sssp 
        
        while(flag)
        {
            __syncthreads();///the problem!
            flag = false;
            index_t dest = threadIdx.x;
            __syncthreads();///the problem!
            while(dest < v_count)
            {
                for(index_t j=dev_beg_pos[dest]; j<dev_beg_pos[dest+1]; ++j)
                {
                    index_t nebr=dev_csr[j];
                    if(shared_dist[dest] > shared_dist[nebr] + dev_weight[j])
                    {
                        shared_dist[dest] = shared_dist[nebr] + dev_weight[j];
                        flag = true;
                    }
                }
                __syncthreads();
                dest += blockDim.x;
                __syncthreads();
            }
        }
        tmp_id = threadIdx.x;
        while(tmp_id < v_count)
        {
            dev_dist[tmp_id+blockIdx.x*blockDim.x] = shared_dist[tmp_id];
            tmp_id += blockDim.x;
        }
        __syncthreads();
    }
}



void cc_gpu_launch(graph *g, path_t *global_cc)//, int start_vert, int end_vert)
{
    index_t v = g->vert_count;
    index_t e = g->edge_count;
    
    index_t *dev_beg_pos;
    vertex_t *dev_csr;
    path_t *dev_weight;
    path_t *dev_dist;
    path_t *dev_cc;

    cudaMalloc( (void **) &dev_beg_pos, (v+1)*sizeof(index_t));
    cudaMalloc( (void **) &dev_csr, e*sizeof(vertex_t));
    cudaMalloc( (void **) &dev_weight, e*sizeof(path_t));
    cudaMalloc( (void **) &dev_dist, v*v*sizeof(path_t));
    cudaMalloc( (void **) &dev_cc, v*sizeof(path_t));
    
    cudaMemcpy(dev_beg_pos, g->beg_pos, (v+1)*sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_csr, g->csr, e*sizeof(index_t),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_weight, g->weight, e*sizeof(path_t),cudaMemcpyHostToDevice);
    cudaMemset(dev_dist, 0, v*v*sizeof(path_t));
    cudaMemset(dev_cc, 0, v*sizeof(path_t));
    
//    double t_sum = 0;
    double btm = wtime();

    apsp_all<<<v, v>>>(dev_beg_pos, dev_csr, dev_weight, dev_dist, v);
    cudaDeviceSynchronize();
//    cc_merge<<<1, v>>>(dev_dist, dev_cc, v);
//    cudaDeviceSynchronize();
    
//    t_sum = wtime() - btm;
//    FILE * fp_time = fopen("time.info", "a");
//    fprintf(fp_time, "%g\n", t_sum);
//    fclose(fp_time);
//    std::cout<<"Kernel time (ms), "<<(wtime()-btm) * 1000<<"\n";
    
    cudaMemcpy(global_cc, dev_cc, v*sizeof(path_t), cudaMemcpyDeviceToHost);

    cudaFree(dev_beg_pos);
    cudaFree(dev_csr);
    cudaFree(dev_weight);
    cudaFree(dev_dist);
//    cudaFree(dev_cc);
}

//int main(int args, char ** argv)
//{
//    if(args != 5)
//        exit(-1);
//    const char *beg_filename = argv[1];
//    const char *csr_filename = argv[2];
//    const char *weight_filename = argv[3];
//    const int thd_count = atoi(argv[4]);
//    
//    graph *g = new graph(beg_filename, csr_filename, weight_filename);
//
//    double btm= wtime();
//
//    cc_gpu_launch(g);
//    std::cout<<"Total time: "<<wtime()-btm<<" seconds\n";
//
//    print_debug();
//
//    return 0;
//}
#endif
