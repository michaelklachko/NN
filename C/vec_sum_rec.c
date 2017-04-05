#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include <immintrin.h>

#define N 128


int print_array(float* a, int n)
{
    int i;
    for(i=0; i<n; i++)
	printf("%.1f, ", a[i]);

    return 0;
}


float hadd(float* a, int n)
{
    printf("\nentering hadd, n=%d\n", n);
    print_array(a, n);

    int i;

    if(n == 8)
    {	//when the vector we want to sum is 8 floats, just add them serially:
	float sum = 0;
	for(i=0; i<8; i++)
	    sum += a[i];
	printf("\n\nReached the base case, returning sum=%.1f\n", sum);
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
	    printf("\n");
	    print_array(a, n);
	}
	//recursively process the first half of a which now contains the results of additions:
	return hadd(a, n/2);
    }
    
    else   //this should never happen
    {
	printf("\n\nInvalid size of vector: %d, should be multiple of 8\n\n", n);
	assert(n >= 8);
	return 0;   //this line is to prevent compiler warning
    }
    
}


int main()
{
    printf("\n\nSumming all elements of array, using AVX horizontal addition (hadd) with 256 bit vectors:\n\n");

    int i;
    float sum=0;
    float sum_test=0;

    float* a = malloc(N * sizeof(float));

    //initialize a:
    for(i=0; i<N; i++)
	a[i] = 0.05*i - 3.2; 
    
    //sequential addtion to check the answer:
    for(i=0; i<N; i++)
	sum_test += a[i];
    
    sum = hadd(a, N);

    printf("\n\nshould be: %.1f", sum_test);
    printf("\nresult:  %.1f", sum);
    printf("\n\n");

    return 0;
}

