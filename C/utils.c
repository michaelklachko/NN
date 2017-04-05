#include "defs.h"

double get_seconds() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + ((1e-6)*tp.tv_usec));
}


void prefix_sums(int *input, int* result, int n) {
  int i;
  result[0] = 0;
  for (i=1; i<n+1; i++) {
    result[i] = result[i-1] + input[i-1];
  }
}


int read_edge_list (int **tailp, int **headp) {
  int nedges, nr, t, h;
  *tailp = (int *) calloc(MAX_INPUT_EDGES, sizeof(int));
  *headp = (int *) calloc(MAX_INPUT_EDGES, sizeof(int));
  nedges = 0;
  nr = scanf("%i %i",&t,&h);
  while (nr == 2) {
    if (nedges >= MAX_INPUT_EDGES) {
      fprintf(stderr,"Limit of %d input edges exceeded.\n",MAX_INPUT_EDGES);
      exit(1);
    }
    (*tailp)[nedges] = t;
    (*headp)[nedges++] = h;
    nr = scanf("%i %i",&t,&h);
  }
  return nedges;
}


graph * graph_from_edge_list (int *tail, int* head, int nedges) {
  graph *G;
  int i, e, v, maxv;
  G = (graph *) calloc(1, sizeof(graph));
  G->ne = nedges;
  maxv = 0;

  // count vertices
  for (e = 0; e < G->ne; e++) {
    if (tail[e] > maxv) maxv = tail[e];
    if (head[e] > maxv) maxv = head[e];
  }
  G->nv = maxv+1;
  G->nbr = (int *) calloc(G->ne, sizeof(int));
  G->firstnbr = (int *) calloc(G->nv+1, sizeof(int));

  // count neighbors of vertex v in firstnbr[v+1],
  for (e = 0; e < G->ne; e++) G->firstnbr[tail[e]+1]++;

  // cumulative sum of neighbors gives firstnbr[] values
  for (v = 0; v < G->nv; v++) G->firstnbr[v+1] += G->firstnbr[v];

  // pass through edges, slotting each one into the CSR structure
  for (e = 0; e < G->ne; e++) {
    i = G->firstnbr[tail[e]]++;
    G->nbr[i] = head[e];
  }
  // the loop above shifted firstnbr[] left; shift it back right
  for (v = G->nv; v > 0; v--) G->firstnbr[v] = G->firstnbr[v-1];
  G->firstnbr[0] = 0;
  return G;
}


void print_CSR_graph (graph *G) {
  int vlimit = 20;
  int elimit = 50;
  int e,v;
  fprintf(stderr,"\nGraph has %d vertices and %d edges.\n",G->nv,G->ne);
  fprintf(stderr,"firstnbr =");
  if (G->nv < vlimit) vlimit = G->nv;
  for (v = 0; v <= vlimit; v++) fprintf(stderr," %d",G->firstnbr[v]);
  if (G->nv > vlimit) fprintf(stderr," ...");
  fprintf(stderr,"\n");
  fprintf(stderr,"nbr =");
  if (G->ne < elimit) elimit = G->ne;
  for (e = 0; e < elimit; e++) fprintf(stderr," %d",G->nbr[e]);
  if (G->ne > elimit) fprintf(stderr," ...");
  fprintf(stderr,"\n\n");
}
