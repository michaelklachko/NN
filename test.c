#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>


float** alloc_2d_float(int rows, int cols, int contiguous)
{
        int i;
        float **array = (float **)malloc(rows * sizeof(float*));

	if(!contiguous)
	    for(i=0; i<rows; i++)
	    {
		array[i] = malloc(cols * sizeof(float));
		assert(array[i] && "Can't allocate memory for labels");
	    }
	else
	{
	    float *data = (float *)malloc(rows*cols*sizeof(float));
	    assert(data && "Can't allocate contiguous memory");

	    for(i=0; i<rows; i++)
                array[i] = &(data[cols * i]);
        }
	return array;
}


int initialize(float** array, int dim1, int dim2)
{
    srand(time(NULL));

    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array[i][j] = rand()/RAND_MAX;

    return 0;
}


int dot(float** array1, float** array2, float** result, int dim1, int dim2, int dim3)
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
    if(argc != 4)
    {
	printf("\n\nUsage: ./test [array size] [n_iters] [contiguous memory allocation], try ./test 100 4 0\n\n");
	return 1;
    }
    int dim3 = atoi(argv[1]);
    int n_iters = atoi(argv[2]);
    int contiguous = atoi(argv[3]);
   
    int dim1 = 100000;
    int dim2 = 1000;

    int i;

    printf("\n\nMultiplying (%d, %d) and (%d, %d) arrays %d times, ", dim1, dim2, dim3, dim2, n_iters);

    if(contiguous)
	printf("contiguous memory allocation.\n\n");
    else
	printf("noncontiguous memory allocation.\n\n");

    clock_t begin1 = clock();

    float** array1 = alloc_2d_float(dim1, dim2, contiguous);
    float** array2 = alloc_2d_float(dim3, dim2, contiguous);
    float** result = alloc_2d_float(dim1, dim3, contiguous);

    clock_t end1 = clock();
    printf("\nInitializing arrays...\n");
    clock_t begin2 = clock();
    
    initialize(array1, dim1, dim2);
    initialize(array2, dim3, dim2);

    clock_t end2 = clock();
    printf("\nMultiplying arrays...\n");
    clock_t begin3 = clock();

    for(i=0; i<n_iters; i++)
	dot(array1, array2, result, dim1, dim2, dim3);

    clock_t end3 = clock();

    printf("\n\nExecution Time:");
    printf("\nAllocating memory:    %.1f seconds\n", (float)(end1 - begin1)/CLOCKS_PER_SEC);
    printf("Initializing arrays: %.1f seconds\n", (float)(end2 - begin2)/CLOCKS_PER_SEC);
    printf("Dot product:         %.1f seconds\n\n", (float)(end3 - begin3)/CLOCKS_PER_SEC);

    return 0;
}
