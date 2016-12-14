#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <immintrin.h>

#define N 64


int print_array(float* a, int n)
{
    int i;
    for(i=0; i<n; i++)
	printf("%.1f, ", a[i]);

    return 0;
}

float fma_dot(float*a, float* b, int n)
{
    __m256 vec_a, vec_b, vec_c;
    int i;

    vec_c = _mm256_setzero_ps();

    for(i=0; i<n; i+=8)
    {
	vec_a = _mm256_loadu_ps(&a[i]);
	vec_b = _mm256_loadu_ps(&b[i]);
	vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
    }

    vec_c = _mm256_hadd_ps(vec_c, vec_c);
    vec_c = _mm256_hadd_ps(vec_c, vec_c);
    
    return ((float*)&vec_c)[0] + ((float*)&vec_c)[4]; 
}


int mult_avx(float* a, float* b, float* m, int n)
{
    __m256 vec_a, vec_b, vec_m;
    int i;

    for(i=0; i<n; i+=8)
    {
	vec_a = _mm256_loadu_ps(&a[i]);
	vec_b = _mm256_loadu_ps(&b[i]);
	vec_m = _mm256_mul_ps(vec_a, vec_b);
	_mm256_storeu_ps(&m[i], vec_m);
    }
    return 0;
}


float hadd(float* a, int n)
{
    int i;

    if(n == 8)
    {	//when the vector we want to sum is 8 floats, just add them serially:
	float sum = 0;
	for(i=0; i<8; i++)
	    sum += a[i];
	return sum;
    }

    else if(n > 8)
    {
	__m256  vec_a1, vec_a2, vec_r;
    
	for(i=0; i<n; i=i+16)
	{
	    //load two 256 bit (8 floats) vectors from a:
	    vec_a1 = _mm256_loadu_ps(&a[i]);   
	    vec_a2 = _mm256_loadu_ps(&a[i+8]);
	    //perform horizontal (pairwise) addition:
	    vec_r = _mm256_hadd_ps(vec_a1, vec_a2);
	    //store resulting 256 bit vector in the part of a which was already read:
	    _mm256_storeu_ps(&a[i/2], vec_r);
	}
	//recursively process the first half of a which now contains the results of additions:
	return hadd(a, n/2);
    }
    
    else   //this should never happen
    {
	printf("\n\nInvalid size of vector: %d, should be multiple of 8\n\n", n);
	assert(n >= 8);
	return 1;   //this line is to prevent compiler warning
    }
    
}


int main()
{
    printf("\n\nDot product of two N element arrays, using AVX multiplication and horizontal addition (hadd) with 256 bit vectors:\n\n");

    int i;
    float result=0;
    float result_test=0;
    float fma_result=0;

    float* a = malloc(N * sizeof(float));
    float* b = malloc(N * sizeof(float));
    float* m = malloc(N * sizeof(float));

    //initialize a:
    for(i=0; i<N; i++)
    {
	b[i] = i - 12;
	a[i] = 0.05*i - .2; 
    }

    //sequential addtion to check the answer:
    for(i=0; i<N; i++)
	result_test += a[i] * b[i];
    
    mult_avx(a, b, m, N);

    result = hadd(m, N);

    fma_result = fma_dot(a, b, N);

    print_array(a, N);
    printf("\n");
    print_array(b, N);
    printf("\n");
    print_array(m, N);
    printf("\n");
    printf("\n\nshould be: %.1f", result_test);
    printf("\nresult:  %.1f", result);
    printf("\nfma result:  %.1f", fma_result);
    printf("\n\n");

    return 0;
}

