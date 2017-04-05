#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int initialize(int dim1, int dim2, float array[][dim2])
{
    printf("\n\nInitializing (%d, %d)\n\n", dim1, dim2);
    srand(time(NULL));

    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array[i][j] = rand()/(float)RAND_MAX;
    return 0;
}


int dot(int dim1, int dim2, int dim3, float array1 [][dim2], float array2[][dim2], float result[][dim3])
{
    int i, j, k;
    float temp;

    for(i=0; i<dim1; i++)
	for(k=0; k<dim3; k++)
	{
	    temp = 0;
	    for(j=0; j<dim2; j++)
		temp += array1[i][j] * array2[k][j];
	    result[i][k] = temp;
	}

    return 0;
}


int main(int argc, char** argv)
{
    if(argc != 3)
    {
	printf("\n\nUsage: ./test [array size] [n_iters], try ./test 100 4\n\n");
	return 1;
    }
    int dim3 = atoi(argv[1]);
    int n_iters = atoi(argv[2]);
   
    int dim1 = 100000;
    int dim2 = 1000;

    int i;

    printf("\n\nMultiplying (%d, %d) and (%d, %d) arrays %d times\n", dim1, dim2, dim3, dim2, n_iters);

    clock_t begin1 = clock();

    float (*array1)[dim2] = malloc(dim1*sizeof(*array1));
    float (*array2)[dim2] = malloc(dim3*sizeof(*array2));
    float (*result)[dim3] = malloc(dim1*sizeof(*result));

    //printf("\n\narray1: %p, array2: %p, result: %p\n\n", array1, array2, result);

    clock_t end1 = clock();
    clock_t begin2 = clock();
    
    initialize(dim1, dim2, array1);
    initialize(dim3, dim2, array2);

    clock_t end2 = clock();
    printf("\nMultiplying arrays...\n");
    clock_t begin3 = clock();

    for(i=0; i<n_iters; i++)
	dot(dim1, dim2, dim3, array1, array2, result);

    printf("\n\nSanity check: result[5][5]: %.2f\n\n", result[5][5]);

    clock_t end3 = clock();

    printf("\n\nExecution Time:");
    printf("\nAllocating memory:    %.1f seconds\n", (float)(end1 - begin1)/CLOCKS_PER_SEC);
    printf("Initializing arrays: %.1f seconds\n", (float)(end2 - begin2)/CLOCKS_PER_SEC);
    printf("Dot product:         %.1f seconds\n\n", (float)(end3 - begin3)/CLOCKS_PER_SEC);

    return 0;
}
