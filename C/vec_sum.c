#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <immintrin.h>

#define N 32


int print_array(float* a, int n)
{
    int i;
    for(i=0; i<n; i++)
	printf("%.1f, ", a[i]);

    return 0;
}


int main()
{

    int i, j;
    float sum=0;
    float sum_test=0;

    /*
    float *data = (float *)malloc(rows*cols*sizeof(float));
    float **array = (float **)malloc(rows * sizeof(float*));
    int i;
    for(i=0; i<rows; i++)
	array[i] = &(data[cols * i]);
    */

    float* a = malloc(N * sizeof(float));
    float* r = malloc(N * sizeof(float));

    //initialize a:
    for(i=0; i<N; i++)
	a[i] = i;

    __m256  vec_a1, vec_a2, vec_r;

    for(i=0; i<N; i=i+16)
    {
	vec_a1 = _mm256_loadu_ps(&a[i]);
	vec_a2 = _mm256_loadu_ps(&a[i+8]);
	vec_r = _mm256_hadd_ps(vec_a1, vec_a2);
	_mm256_storeu_ps(&r[i/2], vec_r);
    }
    
    for(i=0; i<N/2; i++)
	sum += r[i];

    printf("\na:     ");
    print_array(a, N);
    printf("\na1: ");
    print_array(a, N/2);
    printf("\na2: ");
    print_array(&a[N/2], N/2);

    printf("\n\nr: ");
    print_array(r, N/2);

    for(i=0; i<N; i++)
	sum_test += a[i];

    printf("\n\nshould be: %.1f", sum_test);
    printf("\nresult:  %.1f", sum);
    printf("\n\n");

    return 0;
}

