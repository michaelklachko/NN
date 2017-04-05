#ifndef _DEFS_H
#define _DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#define FALSE 0
#define TRUE 1

#define MAX_INPUT_EDGES 10000000  // max edges to read from input file

/* Graph data structure : compressed-adjacency-list (CSR) form */
typedef struct 
{
  int nv;            // number of vertices
  int ne;            // number of edges
  int *nbr;          // array of neighbors of all vertices
  int *firstnbr;     // index in nbr[] of first neighbor of each vtx
} graph;

/* Predecessor list data structure for betweenness centrality */
typedef struct
{
    int* list;
    int count;
    int degree;
} plist;

/* Function declarations */
double betweennessCentrality_parallel(graph*, double*);
double betweennessCentrality_serial(graph*, double*);
double generateTorus(graph*, int, int);
double generateGrid(graph*, int, int);
int read_edge_list (int**, int**);
graph* graph_from_edge_list (int*, int*, int);
void print_CSR_graph (graph*);
void prefix_sums(int*, int*, int);
double get_seconds(void);

#endif
