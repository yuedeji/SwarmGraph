/**
 File name: minions.cu
 Author: Yuede Ji
 Last update: 21:51 08-26-2021
 Description: GPU bc on small graph
    (1) read begin position, csr, weight value from binary file
    (2) betweenness centrality

**/

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <omp.h>
#include "wtime.h"
#include "graph.h"
#include "../util.h"
#include "apsp_gpu.h"
#include "bc_gpu.h"
#include "cc_gpu.h"
#include <queue>

//using namespace std;
const int gpu_count = 8;
std::queue <graph *> graph_queue[gpu_count]; 
int lock[gpu_count];

struct Graph_list 
{ 
    char ** graph_name; 
    int graph_num;
    int tid;
};



void *load_graph_function(void * ptr)
{
    double begin_time = wtime();
    Graph_list * graph_list = (Graph_list *) ptr;
//    printf("graph_list size %d\n", graph_list->graph_num);
    int i = 0;
//    int buffer_size = 50; min(0.1, 100)
    int buffer_size = min(max(1, int((graph_list->graph_num) * 0.1)), 1000);
    
    int tid = graph_list->tid;
    int step = (graph_list->graph_num) / gpu_count;
    int index_beg = tid * step;
    int index_end = (tid == gpu_count - 1 ? graph_list->graph_num : index_beg + step);
    
//    printf("graph_list size %d, tid = %d, index_beg = %d, index_end = %d\n", graph_list->graph_num, graph_list->tid, index_beg, index_end);
    i = index_beg;
    while(i < index_end)
    {
//        printf("i = %d, size = %d\n", i, graph_queue.size());
// Bug: forget to add the second terminate condition        
        while(graph_queue[tid].size() < buffer_size && i < index_end)
        {
             
            graph *g = new graph(graph_list->graph_name[i]);
            i++;
//            while(__sync_lock_test_and_set(&lock[tid], 1));
            graph_queue[tid].push(g);
//            __sync_lock_release(&lock[tid]);
//            printf("i = %d, graph_queue.size = %d, graph_num = %d\n", i, graph_queue.size(), graph_list->graph_num);
        }

//        printf("outer while i = %d\n", i);
    }
    double end_time = wtime();
    printf("Tid, %d, Load graph time (s), %lf\n", tid, (end_time - begin_time));
//    return NULL;
//    while(!graph_queue.empty())
//    {
//        graph *g = graph_queue.front();
//        graph_queue.pop();
//        printf("%d, %d\n", g->vert_count, g->edge_count);
//    }
}

void work_on_one(graph *g, int device_id)
{
    index_t v = g->vert_count;
    index_t e = g->edge_count; 

     cudaError_t err = cudaGetLastError();
//    path_t * dist = (path_t *)calloc(v*v, sizeof(path_t));
//    apsp_gpu_launch(g, dist);
//    free(dist);
//     if ( err != cudaSuccess )
//     {
//        printf("CUDA Error APSP: %s\n", cudaGetErrorString(err));
//        // Possibly: exit(-1) if program cannot continue....
//     }

    path_t * global_cc = (path_t *)malloc(v*v*sizeof(path_t));
    cc_gpu_launch(g, global_cc);
    free(global_cc);
     err = cudaGetLastError();
     if ( err != cudaSuccess )
     {
        printf("CUDA Error CC: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
     }
//
//    path_t * global_bc = (path_t *)malloc(v*v*sizeof(path_t));
//    bc_gpu_launch(g, global_bc);
//    free(global_bc);
//     err = cudaGetLastError();
//     if ( err != cudaSuccess )
//     {
//        printf("CUDA Error BC: %s\n", cudaGetErrorString(err));
//        // Possibly: exit(-1) if program cannot continue....
//     }
}

void * compute_one_gpu(void *ptr)
{
    Graph_list * graph_list = (Graph_list *) ptr;
    int tid = graph_list->tid;

    int step = (graph_list->graph_num) / gpu_count;
    int index_beg = tid * step;
    int index_end = (tid == gpu_count - 1 ? graph_list->graph_num : index_beg + step);
    
    int graph_count = index_end - index_beg;
    int i = 0;

//    printf("tid = %d, graph_count = %d\n", tid, graph_count);
//    int device_id = 0;
// set the correct gpu device, and use the correct graph_queue
    
    cudaSetDevice(tid);

    double begin_time_compute = wtime();
    while(i < graph_count)
    {
        while(!graph_queue[tid].empty())
        {
//            while(__sync_lock_test_and_set(&lock[tid], 1));
            graph *g = graph_queue[tid].front();
            graph_queue[tid].pop();
//            __sync_lock_release(&lock[tid]);
//            printf("%d, %d, %d, %d\n", i, graph_queue.size(), g->vert_count, g->edge_count);
//            for(int j = 0; j < g->vert_count; ++j)
//            {
//                printf("%d %d, ", g->beg_pos[j], g->beg_pos[j+1]);
//            }
            double before_time = wtime();
            work_on_one(g, tid);
            double after_time = wtime();
//            printf("i = %d, time = %lf, graph_queue[%d] size = %d\n", i, (after_time - before_time), tid, graph_queue[tid].size());
            delete g;            
            i++;
        }
    }
//// work stealing
//    for(int j = 0; j < gpu_count; ++j)
//    {
//        printf("tid = %d, j = %d, gpu_count = %d, queue_size = %d\n", tid, j, gpu_count, graph_queue[j].size());
//        while(!graph_queue[j].empty())
//        {
//            graph *g = graph_queue[j].front();
//            graph_queue[j].pop();
//            double before_time = wtime();
//            work_on_one(g, j);
//            double after_time = wtime();
//            delete g;            
//            printf("j = %d, time = %lf\n", j, (after_time - before_time));
//        }
//    }

    double end_time_compute = wtime();
    printf("Tid, %d, compute time (s), %lf\n", tid, (end_time_compute - begin_time_compute));
}

int main(int args, char ** argv)
{
    if(args != 3)
    {
        printf("Usage: ./minios.cu <graph_file_list.txt> <graph_count (int)>\n");
        exit(-1);
    }
/// Step 1: Load all the graphs
    const char *graph_list_file = argv[1];
    int graph_count = int(atoi(argv[2]));
    printf("Graph count, %d\nGraph size, %s\n", graph_count, graph_list_file);

//    char graph_name[graph_count][256];

    char **graph_name = (char **) malloc(graph_count * sizeof(char *));
    FILE * fp = fopen(graph_list_file, "r");
    const int length = 256;
//    char * line = (char *) malloc(length * sizeof(char));
    for(int i = 0; i < graph_count; i++)
    {
//        printf("%d ", i);    
        char line[length];
        fgets(line, sizeof(line), fp);
        int l = strlen(line);
//        printf("%s", line);    
        graph_name[i] = (char *) malloc((length) * sizeof(char));
        line[l - 1] = '\0';
        strcpy(graph_name[i], line);
    }
    fclose(fp);
//    printf("ok\n"); 

    Graph_list graph_list[gpu_count];
    Graph_list *ptr[gpu_count];
//    gpu_info gpu_list[gpu_count];
//    gpu_info *ptr2[gpu_count];

//    printf("Graph count, %d\nGraph size, %s\n", graph_count, graph_list_file);
    for(int i = 0; i < gpu_count; ++i)
    {
        graph_list[i].graph_name = graph_name;
        graph_list[i].graph_num = graph_count;
        graph_list[i].tid = i;
//        gpu_list[i].tid = i;
        ptr[i] = &(graph_list[i]);
        lock[i] = 0;
//        ptr2[i] = &(gpu_list[i]);
    }

//    Graph_list *ptr = &graph_list;
    
    pthread_t thread_1[gpu_count];
    pthread_t thread_2[gpu_count];

    double begin_time = wtime();
    for(int i = 0; i < gpu_count; ++i)
    {
        int iret1 = pthread_create(&thread_1[i], NULL, load_graph_function, (void *) ptr[i]);
        int iret2 = pthread_create(&thread_2[i], NULL, compute_one_gpu, (void *)ptr[i]);
    }


    void *ret;  
    for(int i = 0; i < gpu_count; ++i)
    {
        pthread_join(thread_1[i], &ret);
        pthread_join(thread_2[i], &ret);
    }
    double end_time = wtime();
    printf("Total time (s), %lf\n", (end_time - begin_time));

    return 0;
}
