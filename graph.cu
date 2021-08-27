#include "graph.h"
#include <math.h>

void get_vert_edge_count(const char * file_name, index_t &vert_count, index_t &edge_count)
{
    int l = strlen(file_name);
    int v_count = 0;
    int e_count = 0;
    int _count = 0;
    int split_count = 0;
    for(int i = 0; i < l; ++i)
    {
        if(file_name[i] == '/')
            split_count++;
    }

    int i = 0;
    int temp_split_count = 0;
    for(; i < l; ++i)
    {
        if(file_name[i] == '/')
        {
            temp_split_count++;
            if(temp_split_count == split_count)
                break;
        }
    }
    ++i;
    for(; i < l; ++i)
    {
        if(file_name[i] == '_')
        {
            _count ++;
            if(_count == 3)
            {
                break;
            }
        }
    }
    int j = i;
    ++j;
//    printf("j = %d, l = %d\n", j, l);
    for(; file_name[j] != '_' && j < l; ++j)
    {
//        printf("%c\n", file_name[j]);
        v_count = v_count * 10 + (file_name[j] - '0');
    }
    ++j;
//    printf("j = %d, l = %d\n", j, l);
    for(; file_name[j] != '.' && j < l; ++j)
    {
//        printf("%c\n", file_name[j]);
        e_count = e_count * 10 + (file_name[j] - '0');
    }

    vert_count = int(pow(2, v_count));
    edge_count = e_count;
//    printf("vert_count = %d, edge_count = %d\n", vert_count, edge_count);
}

graph::graph(const char *edgelist_file)
{
//    printf("%s\n", edgelist_file);
	double tm=wtime();
    
    FILE *fp = fopen(edgelist_file, "r");
    vertex_t src, dest;
// Version 1: for the file that provides vert_count and edge_count
//    fscanf(fp, "%d%d", &vert_count, &edge_count);
//    printf("vert_count, %d, edge_count, %d\n", vert_count, edge_count);
// Version 2: for the file that does not provide, but the file name follows "*_scale_edgenumber.graph"
    get_vert_edge_count(edgelist_file, vert_count, edge_count);
//	std::cout<<vert_count<<" verts, "<<edge_count<<" edges ";

    beg_pos = new index_t[vert_count + 1];
    csr = new vertex_t[edge_count];
    weight = new path_t[edge_count];
//Bug: The edgelist should be sorted

//    beg_pos = (index_t *) calloc(vert_count + 1, sizeof(index_t));
//    csr = (vertex_t *) calloc(edge_count, sizeof(vertex_t)); 
//    weight = (path_t *) calloc(edge_count, sizeof(depth_t));    
    index_t index = 0;
    vertex_t prev = 0;
    beg_pos[0] = 0;
    while(fscanf(fp, "%d%d", &src, &dest) != EOF)
    {
//        printf("%d, %d, index = %d, prev = %d\n", src, dest, index, prev);
        weight[index] = 1.0;
        csr[index] = dest;

// change accordingly
        if(src != prev)
        {
            while(prev < src)
            {
                beg_pos[++prev] = index;
            }
        }
        index ++;
    }
    beg_pos[++prev] = index;

    while(prev < vert_count)
    {
        beg_pos[++prev] = index;
    }

    beg_pos[vert_count] = edge_count;
//    for(vertex_t v = 0; v < vert_count; ++v)
//    {
//        printf("%d ", beg_pos[v]);
//        for(vertex_t i = beg_pos[v]; i < beg_pos[v+1]; ++i)
//            printf("%d ", csr[i]);
//        printf("\n");
//    }
//    printf("\n");
//    for(vertex_t v = 0; v < edge_count; ++v)
//    {
//        printf("%d ", csr[v]);
//    }
//    printf("\n");

    fclose(fp);
//	std::cout<<"Graph load (success): "<<vert_count<<" verts, "<<edge_count<<" edges "<<wtime()-tm<<" second(s)\n";


}


