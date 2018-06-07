#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define ntrain 60000
#define ntest 10000
#define img_size 784
#define n_out 10

struct data
{
    float** train_images;
    int* train_labels;
    float** test_images;
    int* test_labels;
    int** train_labels_onehot;
};

struct params
{
    float** W1;
    float** W2;
    float* b1;
    float* b2;
};

struct accuracy
{
    float* training;
    float* test;
};

float max(float* array, int length)
{
    int i;
    float max_value = 0;
    for(i=0; i<length; i++)
	if(array[i] > max_value)
	    max_value = array[i];

    return max_value;
}

float ReLU(float z)
{
    if(z>0)
	return z;
    else
	return 0;
}


int ReLU_vec(float** array, float** result, int dim1, int dim2)
{
    int i, j;
    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    result[i][j] = ReLU(array[i][j]);
    
    return 0;
}


float ReLU_prime(float z)
{
    if(z>0)
	return 1;
    else
	return 0;
}

int ReLU_prime_vec(float** array, int dim1, int dim2)
{
    int i, j;
    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array[i][j] = ReLU_prime(array[i][j]);
    
    return 0;
}

int onehot(int z, int a[n_out])
{
    //int a[n_out];  is it better to use a local variable here?
    int i;

    for(i=0; i<n_out; i++)
	a[i]=0;

    a[z]=1;

    return 0;
}

int initialize_weights(float** W1, float** W2, float* b1, float* b2, int n_hidden)
{
    srand(time(NULL));

    int i, j;

    for(i=0; i<img_size; i++)
	for(j=0; j< n_hidden; j++)
	    W1[i][j] =  sqrt(2.0 / img_size) * (2.0*rand()/RAND_MAX - 1);

    for(i=0; i<n_hidden; i++)
    {
	b1[i] = 0;
	for(j=0; j<n_out; j++)
	    W2[i][j] = sqrt(2.0 / n_hidden) * (2.0*rand()/RAND_MAX - 1);
    }

    for(i=0; i<n_out; i++)
	b2[i] = 0;

    return 0;
}
        

int print_weights(struct params* p)
{
    int i, j;

    printf("\n\nFirst layer Weights and biases:\n\n");
    for(i=0; i<10; i++)
	printf("%.4f ", p->b1[i]);
    printf("\n");

    for(i=0; i<10; i++)
    {
	printf("\n");
	for(j=0; j<10; j++)
	    printf("%.4f ", p->W1[i][j]);
    }
    
    printf("\n\nSecond layer Weights and biases:\n\n");
    for(i=0; i<10; i++)
	printf("%.4f ", p->b2[i]);
    printf("\n");

    for(i=0; i<10; i++)
    {
	printf("\n");
	for(j=0; j<10; j++)
	    printf("%.4f ", p->W2[i][j]);
    }
    return 0;
}


int print_image(int i, float** images)
{
    int j, k;

    printf("\n\n\nLabel: %d\n\n", i);

    for(j=0; j<756; j+=28)
	{
	    printf("\n");
	    for(k=0; k<28; k++)
		printf("%.1f ", images[i][j+k]);
	}
    return 0;
}

int print_array(float** array, int dim1, int dim2)
{
    printf("\n\n");
    int i, j;
    for(i=0; i<dim1; i++)
    {
	printf("\n");
	for(j=0; j<dim2; j++)
	    printf("%.4f ", array[i][j]);
    }
    return 0;
}

int load_mnist(struct data* d)
{
    FILE *tr_i, *tr_l, *te_i, *te_l;
    int i, j;

    tr_i = fopen("mnist_text/train_images.txt", "r");
    tr_l = fopen("mnist_text/train_labels.txt", "r");
    te_i = fopen("mnist_text/test_images.txt", "r");
    te_l = fopen("mnist_text/test_labels.txt", "r");

    if(tr_i == NULL || tr_l == NULL || te_i == NULL || te_l == NULL)
    {
	printf("\n\nCan't open MNIST file\n\n");
	return 1;
    }

    for(i=0; i<ntrain; i++)
    {
	fscanf(tr_l, "%d", &d->train_labels[i]);
	for(j=0; j<img_size; j++)
	    fscanf(tr_i, "%f", &d->train_images[i][j]);
    }
    for(i=0; i<ntest; i++)
    {
	fscanf(te_l, "%d", &d->test_labels[i]);
	for(j=0; j<img_size; j++)
	    fscanf(te_i, "%f", &d->test_images[i][j]);
    }
    
    fclose(tr_i);
    fclose(tr_l);
    fclose(te_i);
    fclose(te_l);

    return 0;
}


int transpose(float** array, float** result, int dim1, int dim2)
{
    //printf("\n\nEntered Transpose function\n\n");
    //printf("dim1: %d, dim2: %d\n", dim1, dim2);

    int i, j;
    for(i=0; i<dim1; i++)
    {
	//printf("\n%d\n\n", i);
	for(j=0; j<dim2; j++)
	{
	    //printf("%d, ", j);
    	    result[j][i] = array[i][j];
	}
    }
    return 0;
}


float hadd(float* a, int n)
{
    int i;

    if(n == 8)
    {   //when the vector we want to sum is 8 floats, just add them serially:
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


float avx_dot1(float* a, float* b, float* m, int N)
{
    int i;
    float result=0;

    __m256 vec_a, vec_b, vec_m;

    for(i=0; i<N; i+=8)
    {
        vec_a = _mm256_loadu_ps(&a[i]);
        vec_b = _mm256_loadu_ps(&b[i]);
        vec_m = _mm256_mul_ps(vec_a, vec_b);
        _mm256_storeu_ps(&m[i], vec_m);
    }

    result = hadd(m, N);

    return result;
}



float avx_dot(float*a, float* b, int n)
{
    __m256 vec_a, vec_b, vec_c;
    int i;
    printf("\n111\n");
    vec_c = _mm256_setzero_ps();

    printf("\n112\nn: %d\n", n);
    for(i=0; i<n; i+=8)
    {
	printf("i: %d, ", i);
        vec_a = _mm256_loadu_ps(&a[i]);
	printf("vec_a loaded, ");
        vec_b = _mm256_loadu_ps(&b[i]);
	printf("vec_b loaded, ");
        vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
	printf("vec_c updated.\n");
    }

    printf("\n113\n");
    vec_c = _mm256_hadd_ps(vec_c, vec_c);
    printf("\n114\n");
    vec_c = _mm256_hadd_ps(vec_c, vec_c);

    return ((float*)&vec_c)[0] + ((float*)&vec_c)[4];
}




float **alloc_2d_float(int rows, int cols){
        float *data = (float *)malloc(rows*cols*sizeof(float));
        float **array = (float **)malloc(rows * sizeof(float*));
        int i;
        for(i=0; i<rows; i++)
                array[i] = &(data[cols * i]);
        return array;
}

int dot1(float** array1, float** array2, float** result, int dim1, int dim2, int dim3, int transposed)
{
    /*

    Traditional dot product is (dim1, dim2) x (dim2, dim3) = (dim1, dim3)

    One way to make it more cache friendly is to multiply two matrices with the same second dimension:

    (dim1, dim2)x(dim3, dim2) = (dim1, dim3)
    
    because this allows to multiply rows of a1 with rows of a2, rather than rows of a1 with columns of a2.

    This requires transposing a2, unless it's already transposed as needed by backprop function.
    We can either do transpose operation within backprop function, or within dot function.

    We have to consider 4 scenarios:
    
    1. neither a1 or a2 are transposed in backprop (transposed = 0)
	    (dim1, dim2) x (dim2, dim3)
	    transpose a2, do cache friendly (CF) dot product.

    2. a1 is transposed (transposed = 1) 
	    (dim2, dim1) x (dim2, dim3)
	    worst case, transpose both a1 and a2, do CF dot product

    3. a2 is already transposed (transposed = 2) 
	    (dim1, dim2) x (dim3, dim2)
	    perfect case, just do CF dot product

    4. do a traditional dot product (transposed = 3)

    */

    int i, j, k;
    float temp;
    //float** A;
    //float** B;
    
    /*
    
    if(transposed == 0)         //(dim1, dim2)x(dim2, dim3)  
    {
	B = alloc_2d_float(dim3, dim2);  
	
	A = array1;
	transpose(array2, B, dim2, dim3);

    }
    
    else if(transposed == 1)    //(dim2, dim1)x(dim2, dim3)
    {
	A = alloc_2d_float(dim1, dim2);
	B = alloc_2d_float(dim3, dim2);
	
	transpose(array1, A, dim2, dim1);
	transpose(array2, B, dim2, dim3);
    }
    
    else if(transpose == 2)     //(dim1, dim2)x(dim3, dim2)
    {
	A = array1;
	B = array2;
    }
    */
    
    if(transposed == 3)
	{
	    //do traditional dot product
	    for(i=0; i<dim1; i++)
		for(k=0; k<dim3; k++)
		{
		    temp = 0;
		    for(j=0; j<dim2; j++)
			temp += array1[i][j] * array2[j][k];
		    result[i][k] = temp;
		}
	}
    else    //(transposed == 0, 1, 2)
    {
	
	printf("\ndim1: %d, dim2: %d, dim3: %d\n\n", dim1, dim2, dim3);
	for(i=0; i<dim1; i++)
	    for(k=0; k<dim3; k++)  
		result[i][k] = avx_dot(array1[i], array2[k], dim2);
	 
	
	/*
	if(dim2 == 10)
	{
	    float* temp_vec1 = calloc(16, sizeof(float)); //for avx_dot intermediate result
	    float* a16 = calloc(16, sizeof(float)); //for avx_dot intermediate result
	    float* b16 = calloc(16, sizeof(float)); //for avx_dot intermediate result

	    for(i=0; i<dim1; i++)
		for(k=0; k<dim3; k++)
		{   
		    for(j=0; j<dim2; j++)
		    {
			a16[j] = array1[i][j];
			b16[j] = array2[k][j];
		    }
		    result[i][k] = avx_dot1(a16, b16, temp_vec1, 16);
		}

	    free(temp_vec1);

	}
	else
	{
	    //printf("\n\ndim2: %d\n\n", dim2);
	    float* temp_vec2 = malloc(dim2 * sizeof(float)); //for avx_dot intermediate result

	    for(i=0; i<dim1; i++)
		for(k=0; k<dim3; k++)
		    result[i][k] = avx_dot1(array1[i], array2[k], temp_vec2, dim2);
	    
	    free(temp_vec2);
	}
	*/

    }
    return 0;
}


int dot(float** array1, float** array2, float** result, int dim1, int dim2, int dim3)
{
    //(dim1, dim2)x(dim2, dim3)

    int i, j, k;
    float temp;

    for(i=0; i<dim1; i++)
	for(k=0; k<dim3; k++)
	{
	    temp = 0;
	    for(j=0; j<dim2; j++)
		//temp += array1[i][j] * array2[j][k];
		temp += array1[i][j] * array2[k][j];
	    result[i][k] = temp;
	}

    return 0;
}

int add_vec(float** array, float* vec, int dim1, int dim2)
{
    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array[i][j] += vec[j];

    return 0;
}

int substract(float** array1, int** array2, float** result, int dim1, int dim2)
{
    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    result[i][j] = array1[i][j] - (float)array2[i][j];
    
    return 0;
}

int sum_columns(float** array, float* vec, int vec_length, int batch_size)
{

    int i, j;

    //array shape: (batch_size, vec_length): sum all elements in each column to make a vector
    //picking one value per row method:
    /*
    for(j=0; j<vec_length; j++)
	for(i=0; i<batch_size; i++)
	    vec[j] += array[i][j];
    */
    //filling each row incrementally (might be more efficient):
    for(j=0; j<vec_length; j++)
	vec[j] = array[0][j];
    for(i=1; i<batch_size; i++)
	for(j=0; j<vec_length; j++)
	    vec[j] += array[i][j];

    
    return 0;
}


int product(float** array1, float** array2, int dim1, int dim2)
{
    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array1[i][j] *= array2[i][j];

    return 0;
}


int feedforward(float** batch, float** z_hidden, float** output_hidden, float** z_out, 
		int batch_size, int n_hidden, struct params* p)
{

    //need to transpose both W1 and W2 for cache friendly dot product
    dot1(batch, p->W1, z_hidden, batch_size, img_size, n_hidden, 3);

    add_vec(z_hidden, p->b1, batch_size, n_hidden);

    ReLU_vec(z_hidden, output_hidden, batch_size, n_hidden);

    dot1(output_hidden, p->W2, z_out, batch_size, n_hidden, n_out, 3);

    add_vec(z_out, p->b2, batch_size, n_out);

    return 0;
}


int backprop(float** error_out, float** batch, float** z_hidden, float** output_hidden,
	    float** W2, struct params* grad, int batch_size, int n_hidden, 
	    float** error_hidden, float** output_hidden_transposed,
	    float** batch_transposed, float** error_hidden_transposed, float** error_out_transposed)
{


    //grad->b1 or &(grad->b1)?  is grad->b1 passing by value, or by reference?
    sum_columns(error_out, grad->b2, n_out, batch_size);

    //printf("\n11\n");
    transpose(output_hidden, output_hidden_transposed, batch_size, n_hidden);
    //dot(output_hidden_transposed, error_out, grad->W2, n_hidden, batch_size, n_out);

    //This is the worst case (transposed = 1):
    //we untranspose the first array, and transpose the second one:
    //(dim2, dim1)x(dim2, dim3) --> (dim1, dim2)x(dim3, dim2) = (dim1, dim3)
    
    transpose(error_out, error_out_transposed, batch_size, n_out);

    //printf("\n12\n");       //(batch_size, n_hidden) x (n_out, batch_size)
    dot1(output_hidden_transposed, error_out_transposed, grad->W2, n_hidden, batch_size, n_out, 1);
    
    //printf("\n13\n");
    //transpose(W2, W2_transposed, n_hidden, n_out);
    //dot(error_out, W2_transposed, error_hidden, batch_size, n_out, n_hidden);

    //perfect case: second array is already transposed:
    dot1(error_out, W2, error_hidden, batch_size, n_out, n_hidden, 2);

    //printf("\n14\n");
    ReLU_prime_vec(z_hidden, batch_size, n_hidden);

    //printf("\n15\n");
    //z_hidden was modified above
    product(error_hidden, z_hidden, batch_size, n_hidden);

    //printf("\n16\n");
    sum_columns(error_hidden, grad->b1, n_hidden, batch_size);

    //printf("\n17\n");
    
    transpose(batch, batch_transposed, batch_size, img_size);
    //dot(batch_transposed, error_hidden, grad->W1, img_size, batch_size, n_hidden);
    
    //worst case again:
    transpose(error_hidden, error_hidden_transposed, batch_size, n_hidden);
    //printf("\n18\n");
    dot1(batch_transposed, error_hidden_transposed, grad->W1, img_size, batch_size, n_hidden, 1);
    //printf("\n19\n");
    
    return 0;
}


int update_parameters(struct params* p, struct params* grad, float scale, int n_hidden)
{
    int i, j;

    for(i=0; i<img_size; i++)
	for(j=0; j<n_hidden; j++)
	    p->W1[i][j] -= scale * grad->W1[i][j];

    for(i=0; i<n_hidden; i++)
    {
	p->b1[i] -= scale * grad->b1[i];
	for(j=0; j<n_out; j++)
	    p->W2[i][j] -= scale * grad->W2[i][j];
    }
    for(i=0; i<n_out; i++)
	p->b2[i] -= scale * grad->b2[i];

    return 0;
}

int argmax(float** array, int* result, int dim1, int dim2)
{
    int i, j;
    float max;
    
    for(i=0; i<dim1; i++)
    {
	max = -10000;
	int position = -10000; //we want to segfault if position not updated
	for(j=0; j<dim2; j++)
	    if(array[i][j] > max)
	    {
		max = array[i][j];
		position = j;
	    }
	result[i] = position;
    }

    return 0;
}

int count_correct(int* predictions, int* labels, int length)
{
    int i;
    int n_correct = 0;

    for(i=0; i<length; i++)
	if(predictions[i] == labels[i])
	    n_correct++;

    return n_correct;
}

int test_accuracy(struct data* d, struct params* p, struct accuracy* results, 
		float** z_hidden_train, float** output_hidden_train, float** z_out_train, 
		float** z_hidden_test, float** output_hidden_test, float** z_out_test,  
		int n_hidden, int i)
{
    int train_correct, test_correct;
    int training_predictions[ntrain];
    int test_predictions[ntest];

    feedforward(d->train_images, z_hidden_train, output_hidden_train, z_out_train, ntrain, n_hidden, p);

    argmax(z_out_train, training_predictions, ntrain, n_out);
    
    train_correct = count_correct(training_predictions, d->train_labels, ntrain);

    results->training[i] = 100 * train_correct / (float)ntrain;

    feedforward(d->test_images, z_hidden_test, output_hidden_test, z_out_test, ntest, n_hidden, p);

    argmax(z_out_test, test_predictions, ntest, n_out);

    test_correct = count_correct(test_predictions, d->test_labels, ntest);

    results->test[i] = 100 * test_correct / (float)ntest;

    return 0;
}


int main(int argc, char** argv)
{
    //path = "/mnt/c/Users/Michael/Desktop/Research/Data/mnist/";
    char path[1000];
    strcpy(path, "/mnt/c/Users/Michael/Desktop/Research/Data/mnist/");

    int batch_size;
    int n_epochs;
    float learning_rate;
    int n_hidden;
 
    if(argc != 5)
    {
	printf("\n\nUsage: ./mlp [n_hidden] [batch_size] [learning_rate] [n_epochs], try ./mlp 50 200 0.2 4\n\n");
	return 1;
    }
    n_hidden = atoi(argv[1]);
    batch_size = atoi(argv[2]);
    learning_rate = atof(argv[3]);
    n_epochs = atoi(argv[4]);
    
    float scale = learning_rate/batch_size;


    float** W1;
    float b1[n_hidden];
    float** W2;
    float b2[n_out];
 
    float** grad_W1;
    float grad_b1[n_hidden];
    float** grad_W2;
    float grad_b2[n_out];
    

    float** train_images;
    int train_labels[ntrain];
    float** test_images;
    int test_labels[ntest];
    int** train_labels_onehot;
    
    struct params p;
    struct data d;
    struct params grad;
    struct accuracy results;
    
    printf("\nThis program trains a two layer fully connected neural network to recognize \
handwritten digits (MNIST)\n\nNetwork Size: %d, Minibatch Size: %d, Learning Rate: %.2f, \
Training for %d epochs.\n\n", n_hidden, batch_size, learning_rate, n_epochs);

    int i, j;

    W1 = malloc(img_size * sizeof(float*));
    for(i=0; i<img_size; i++)
	W1[i] = malloc(n_hidden * sizeof(float));

    W2 = malloc(n_hidden * sizeof(float*));
    for(i=0; i<n_hidden; i++)
	W2[i] = malloc(n_out * sizeof(float));

    grad_W1 = malloc(img_size * sizeof(float*));
    for(i=0; i<img_size; i++)
	grad_W1[i] = malloc(n_hidden * sizeof(float));

    grad_W2 = malloc(n_hidden * sizeof(float*));
    for(i=0; i<n_hidden; i++)
	grad_W2[i] = malloc(n_out * sizeof(float));

    printf("\nInitializing weights...\n");

    initialize_weights(W1, W2, b1, b2, n_hidden);

    train_images = malloc(ntrain * sizeof(float*));
    for(i=0; i<ntrain; i++)
	train_images[i] = malloc(img_size * sizeof(float));

    test_images = malloc(ntest * sizeof(float*));
    for(i=0; i<ntest; i++)
	test_images[i] = malloc(img_size * sizeof(float));

    train_labels_onehot = malloc(ntrain * sizeof(int*));
    for(i=0; i<ntrain; i++)
	train_labels_onehot[i] = malloc(n_out * sizeof(int));


    p.W1 = W1;
    p.W2 = W2;
    p.b1 = b1;
    p.b2 = b2;

    grad.W1 = grad_W1;
    grad.b1 = grad_b1; 
    grad.W2 = grad_W2;
    grad.b2 = grad_b2;
       

    d.train_images = train_images;
    d.train_labels = train_labels;
    d.test_images = test_images;
    d.test_labels = test_labels;

    printf("\nLoading MNIST dataset...\n");
    load_mnist(&d);  //first arg is a pointer to struct data

    for(i=0; i<ntrain; i++)
	onehot(train_labels[i], train_labels_onehot[i]);

    d.train_labels_onehot = train_labels_onehot;

    float** z_hidden;  //(batch_size, n_hidden)
    float** output_hidden; //same
    float** z_out;      //(batch_size, n_out)
    float** output_hidden_transposed;  //do we really need it?

    z_hidden = malloc(batch_size * sizeof(float*));
    for(i=0; i<batch_size; i++)
	z_hidden[i] = malloc(n_hidden * sizeof(float));

    output_hidden = malloc(batch_size * sizeof(float*));
    for(i=0; i<batch_size; i++)
	output_hidden[i] = malloc(n_hidden * sizeof(float));

    output_hidden_transposed = malloc(n_hidden * sizeof(float*));
    for(i=0; i<n_hidden; i++)
	output_hidden_transposed[i] = malloc(batch_size * sizeof(float));

    z_out = malloc(batch_size * sizeof(float*));
    for(i=0; i<batch_size; i++)
	z_out[i] = malloc(n_out * sizeof(float));

    float** error_out;  //output layer errors
    float** error_hidden; //hidden layer errors

    float** error_out_transposed;  //output layer errors
    float** error_hidden_transposed; //hidden layer errors

    error_out = malloc(batch_size * sizeof(float*));
    for(i=0; i<batch_size; i++)
	error_out[i] = malloc(n_out * sizeof(float));

    error_hidden = malloc(batch_size * sizeof(float*));
    for(i=0; i<batch_size; i++)
	error_hidden[i] = malloc(n_hidden * sizeof(float));

    error_out_transposed = malloc(n_out * sizeof(float*));
    for(i=0; i<n_out; i++)
	error_out_transposed[i] = malloc(batch_size * sizeof(float));

    error_hidden_transposed = malloc(n_hidden * sizeof(float*));
    for(i=0; i<n_hidden; i++)
	error_hidden_transposed[i] = malloc(batch_size * sizeof(float));


    float*** batches;    //3D array: (nbatches, batch_size, img_size)

    int nbatches = ntrain/batch_size;

    batches = malloc((nbatches) * sizeof(float*));
    for(i=0; i<nbatches; i++)
    {
	batches[i] = malloc(batch_size * sizeof(float*));
	for(j=0; j<batch_size; j++)
	{
	    batches[i][j] = malloc(img_size * sizeof(float));
	    batches[i][j] = train_images[i*batch_size + j];  
	}
    }

    int*** batch_labels;  //3D array: (nbatches, batch_size, 10) because they are onehot labels
    
    batch_labels = malloc(nbatches * sizeof(int*));
    for(i=0; i<nbatches; i++)
    {
	batch_labels[i] = malloc(batch_size * sizeof(int));
	for(j=0; j<batch_size; j++)
	{
	    batch_labels[i][j] = malloc(n_out * sizeof(int*));
	    batch_labels[i][j] = train_labels_onehot[i*batch_size + j];
	}
    }

    float** W2_transposed;
    float** batch_transposed;

    W2_transposed = malloc(n_out * sizeof(float*));
    for(i=0; i<n_out; i++)
	W2_transposed[i] = malloc(n_hidden * sizeof(float));

    batch_transposed = malloc(img_size * sizeof(float*));
    for(i=0; i<img_size; i++)
	batch_transposed[i] = malloc(batch_size * sizeof(float));

    float* training_results;
    float* test_results;

    training_results = malloc(n_epochs * sizeof(float));
    test_results = malloc(n_epochs * sizeof(float));
	
    results.training = training_results;
    results.test = test_results;


    float** z_hidden_train;  //(60k, n_hidden)
    float** output_hidden_train; //same
    float** z_out_train;      //(60k, n_out)

    z_hidden_train = malloc(ntrain * sizeof(float*));
    for(i=0; i<ntrain; i++)
	z_hidden_train[i] = malloc(n_hidden * sizeof(float));

    output_hidden_train = malloc(ntrain * sizeof(float*));
    for(i=0; i<ntrain; i++)
	output_hidden_train[i] = malloc(n_hidden * sizeof(float));

    z_out_train = malloc(ntrain * sizeof(float*));
    for(i=0; i<ntrain; i++)
	z_out_train[i] = malloc(n_out * sizeof(float));

    float** z_hidden_test;  //(10k, n_hidden)
    float** output_hidden_test; //same
    float** z_out_test;      //(10k, n_out)

    z_hidden_test = malloc(ntest * sizeof(float*));
    for(i=0; i<ntest; i++)
	z_hidden_test[i] = malloc(n_hidden * sizeof(float));

    output_hidden_test = malloc(ntest * sizeof(float*));
    for(i=0; i<ntest; i++)
	output_hidden_test[i] = malloc(n_hidden * sizeof(float));

    z_out_test = malloc(ntest * sizeof(float*));
    for(i=0; i<ntest; i++)
	z_out_test[i] = malloc(n_out * sizeof(float));


    //***** Training Starts Here ******
    printf("\nTraining network...\n");
    
    clock_t begin = clock();

    for(i=0; i<n_epochs; i++)
    {
	for(j=0; j<nbatches; j++)
	{
	    //should we pass by value, or pass by reference? for example, p.W1 vs &p.W1
	    feedforward(batches[j], z_hidden, output_hidden, z_out, batch_size, n_hidden, &p);
	    //printf("\n1\n");
	    substract(z_out, batch_labels[j], error_out, batch_size, n_out);
	    //printf("\n2\n");
	    
	    backprop(error_out, batches[j], z_hidden, output_hidden, p.W2, &grad, batch_size, 
		    n_hidden, error_hidden, output_hidden_transposed, batch_transposed,
		    error_hidden_transposed, error_out_transposed);

	    //printf("\n3\n");
	    update_parameters(&p, &grad, scale, n_hidden);
	}

	test_accuracy(&d, &p, &results, z_hidden_train, output_hidden_train, z_out_train, 
			z_hidden_test, output_hidden_test, z_out_test, n_hidden, i);


	printf("\nEpoch %d: training dataset accuracy: %.2f, test dataset accuracy: %.2f", 
		i, results.training[i], results.test[i]); 

    }

    clock_t end = clock();

    float train_best = max(results.training, n_epochs);
    float test_best = max(results.test, n_epochs);

    printf("\n\nBest Accuracy: %.2f (training dataset), %.2f (test dataset)\n\n", train_best, test_best);
    printf("\n---- Program ran for %.1f seconds ----\n\n", (float)(end - begin)/CLOCKS_PER_SEC);

    return 0;
}
