#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <immintrin.h>

#define N 16


int print_array(float* a, int n)
{
    int i;
    for(i=0; i<N; i++)
	printf("%.1f, ", a[i]);

    return 0;
}


int main()
{

    int i, j;
    /*
    float *data = (float *)malloc(rows*cols*sizeof(float));
    float **array = (float **)malloc(rows * sizeof(float*));
    int i;
    for(i=0; i<rows; i++)
	array[i] = &(data[cols * i]);
    */

    float* a = malloc(N * sizeof(float));
    float* b = malloc(N * sizeof(float));
    float* c = malloc(N * sizeof(float));
    float* r = malloc(N * sizeof(float));
    float* r_test = malloc(N * sizeof(float));

    for(i=0; i<N; i++)
    {
	a[i] = i;
	b[i] = 3*i-2;
	c[i] = -3*i;
    }

    __m256  vec_a1, vec_b1, vec_c1, vec_r1, vec_a2, vec_b2, vec_c2, vec_r2;

    printf("\na:     ");
    print_array(a, N);
    printf("\nb:     ");
    print_array(b, N);
    //printf("\n\narray c: ");
    //print_array(c, N);

    for(i=0; i<N; i=i+8)
    {
	vec_a1 = _mm256_loadu_ps(&a[i]);
	vec_b1 = _mm256_loadu_ps(&b[i]);
	/*
	vec_c1 = _mm256_loadu_ps(&c[i]);

	vec_a2 = _mm256_loadu_ps(&a[i+1]);
	vec_b2 = _mm256_loadu_ps(&b[i+1]);
	vec_c2 = _mm256_loadu_ps(&c[i+1]);

	vec_r1 = _mm256_fmadd_ps(vec_a1, vec_b1, vec_c1);
	vec_r2 = _mm256_fmadd_ps(vec_a2, vec_b2, vec_c2);
	*/

	vec_r1 = _mm256_hadd_ps(vec_a1, vec_b1);

	_mm256_storeu_ps(&r[i], vec_r1);
	//_mm256_storeu_ps(&r[i+1], vec_r2);
    }

    //for(i=0; i<N; i++)
	//r_test[i] = a[i] * b[i] + c[i];
    

    for(i=0; i<N; i+=4)
    {	
	r_test[i] = a[i] + a[i+1];
	r_test[i+1] = a[i+2] + a[i+3];

	r_test[i+2] = b[i] + b[i+1];
	r_test[i+3] = b[i+2] + b[i+3];
    }

    printf("\n\nshould be: ");
    print_array(r_test, N);
    printf("\nresults:   ");
    print_array(r, N);
    printf("\n\n");

    return 0;
}

