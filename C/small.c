#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float** alloc_2d_float(int rows, int cols, int contiguous){
    int i;
    float** array = malloc(rows * sizeof(float*));
    if(contiguous){
	float* data = malloc(rows*cols*sizeof(float));
	for(i=0; i<rows; i++)
            array[i] = &(data[cols * i]);
    }
    else
	for(i=0; i<rows; i++)
	    array[i] = malloc(cols * sizeof(float));
    return array;
}

int initialize(float** array, int dim1, int dim2){
    srand(time(NULL));
    int i, j;
    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array[i][j] = rand()/RAND_MAX;
    return 0;
}

int main(int argc, char** argv){
    int i,j,k, dim1=100000, dim2=1000, dim3=300;
    int contiguous=0;
    float temp;

    float** array1 = alloc_2d_float(dim1, dim2, contiguous);
    float** array2 = alloc_2d_float(dim3, dim2, contiguous);
    float** result = alloc_2d_float(dim1, dim3, contiguous);
   
    initialize(array1, dim1, dim2);
    initialize(array2, dim3, dim2);

    for(i=0; i<dim1; i++)
	for(k=0; k<dim3; k++){
	    temp = 0;
	    for(j=0; j<dim2; j++)
		temp += array1[i][j] * array2[k][j];
	    result[i][k] = temp;
	}
    return 0;
}
