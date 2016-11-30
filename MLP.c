#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define ntrain 60000
#define ntest 10000
#define img_size 784
#define n_out 10

struct data
{
    double** train_images;
    int* train_labels;
    double** test_images;
    int* test_labels;
    int** train_labels_onehot;
};

struct params
{
    double** W1;
    double** W2;
    double* b1;
    double* b2;
};

struct accuracy
{
    double* training;
    double* test;
};

double max(double* array, int length)
{
    int i;
    double max_value = 0;
    for(i=0; i<length; i++)
	if(array[i] > max_value)
	    max_value = array[i];

    return max_value;
}

double ReLU(double z)
{
    if(z>0)
	return z;
    else
	return 0;
}


int ReLU_vec(double** array, double** result, int dim1, int dim2)
{
    int i, j;
    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    result[i][j] = ReLU(array[i][j]);
    
    return 0;
}


double ReLU_prime(double z)
{
    if(z>0)
	return 1;
    else
	return 0;
}

int ReLU_prime_vec(double** array, int dim1, int dim2)
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

int initialize_weights(double** W1, double** W2, double* b1, double* b2, int n_hidden)
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


int print_image(int i, double** images, int* labels)
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

int print_array(double** array, int dim1, int dim2)
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

int load_mnist(struct data* d, char* path)
{
    FILE *tr_i, *tr_l, *te_i, *te_l;
    int i, j;

    tr_i = fopen(strcat(path, "train_images.txt"), "r");
    tr_l = fopen("/mnt/c/Users/Michael/Desktop/Research/Data/mnist/train_labels.txt", "r");
    te_i = fopen("/mnt/c/Users/Michael/Desktop/Research/Data/mnist/test_images.txt", "r");
    te_l = fopen("/mnt/c/Users/Michael/Desktop/Research/Data/mnist/test_labels.txt", "r");

    if(tr_i == NULL || tr_l == NULL || te_i == NULL || te_l == NULL)
    {
	printf("\n\nCan't open MNIST file\n\n");
	return 1;
    }

    for(i=0; i<ntrain; i++)
    {
	fscanf(tr_l, "%d", &d->train_labels[i]);
	for(j=0; j<img_size; j++)
	    fscanf(tr_i, "%lf", &d->train_images[i][j]);
    }
    for(i=0; i<ntest; i++)
    {
	fscanf(te_l, "%d", &d->test_labels[i]);
	for(j=0; j<img_size; j++)
	    fscanf(te_i, "%lf", &d->test_images[i][j]);
    }
    
    fclose(tr_i);
    fclose(tr_l);
    fclose(te_i);
    fclose(te_l);

    return 0;
}


int dot(double** array1, double** array2, double** result, int dim1, int dim2, int dim3)
{
    //(dim1, dim2)x(dim2, dim3)

    int i, j, k;
    double temp;

    for(i=0; i<dim1; i++)
	for(k=0; k<dim3; k++)
	{
	    temp = 0;
	    for(j=0; j<dim2; j++)
		temp += array1[i][j] * array2[j][k];
	    result[i][k] = temp;
	}

    return 0;
}

int transpose(double** array, double** result, int dim1, int dim2)
{
    int i, j;
    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    result[j][i] = array[i][j];

    return 0;
}


int add_vec(double** array, double* vec, int dim1, int dim2)
{
    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array[i][j] += vec[j];

    return 0;
}

int substract(double** array1, int** array2, double** result, int dim1, int dim2)
{
    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    result[i][j] = array1[i][j] - (double)array2[i][j];
    
    return 0;
}

int sum_columns(double** array, double* vec, int vec_length, int batch_size)
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


int product(double** array1, double** array2, int dim1, int dim2)
{
    int i, j;

    for(i=0; i<dim1; i++)
	for(j=0; j<dim2; j++)
	    array1[i][j] *= array2[i][j];

    return 0;
}


int feedforward(double** batch, double** z_hidden, double** output_hidden, double** z_out, 
		int batch_size, int n_hidden, struct params* p)
{

    dot(batch, p->W1, z_hidden, batch_size, img_size, n_hidden);

    add_vec(z_hidden, p->b1, batch_size, n_hidden);

    ReLU_vec(z_hidden, output_hidden, batch_size, n_hidden);

    dot(output_hidden, p->W2, z_out, batch_size, n_hidden, n_out);

    add_vec(z_out, p->b2, batch_size, n_out);

    return 0;
}


int backprop(double** error_out, double** batch, double** z_hidden, double** output_hidden,
	    double** z_out, double** W2, struct params* grad, int batch_size, int n_hidden, 
	    double** error_hidden, double** output_hidden_transposed, double** W2_transposed,
	    double** batch_transposed)
{


    //grad->b1 or &(grad->b1)?  is grad->b1 passing by value, or by reference?
    sum_columns(error_out, grad->b2, n_out, batch_size);

    transpose(output_hidden, output_hidden_transposed, batch_size, n_hidden);

    dot(output_hidden_transposed, error_out, grad->W2, n_hidden, batch_size, n_out);
    
    transpose(W2, W2_transposed, n_hidden, n_out);

    dot(error_out, W2_transposed, error_hidden, batch_size, n_out, n_hidden);

    ReLU_prime_vec(z_hidden, batch_size, n_hidden);

    //z_hidden was modified above
    product(error_hidden, z_hidden, batch_size, n_hidden);

    sum_columns(error_hidden, grad->b1, n_hidden, batch_size);

    transpose(batch, batch_transposed, batch_size, img_size);

    dot(batch_transposed, error_hidden, grad->W1, img_size, batch_size, n_hidden);
    
    return 0;
}


int update_parameters(struct params* p, struct params* grad, double scale, int n_hidden)
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

int argmax(double** array, int* result, int dim1, int dim2)
{
    int i, j;
    double max;
    
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
		double** z_hidden_train, double** output_hidden_train, double** z_out_train, 
		double** z_hidden_test, double** output_hidden_test, double** z_out_test,  
		int n_hidden, int i)
{
    int train_correct, test_correct;
    int training_predictions[ntrain];
    int test_predictions[ntest];

    feedforward(d->train_images, z_hidden_train, output_hidden_train, z_out_train, ntrain, n_hidden, p);

    argmax(z_out_train, training_predictions, ntrain, n_out);
    
    train_correct = count_correct(training_predictions, d->train_labels, ntrain);

    results->training[i] = 100 * train_correct / (double)ntrain;

    feedforward(d->test_images, z_hidden_test, output_hidden_test, z_out_test, ntest, n_hidden, p);

    argmax(z_out_test, test_predictions, ntest, n_out);

    test_correct = count_correct(test_predictions, d->test_labels, ntest);

    results->test[i] = 100 * test_correct / (double)ntest;

    return 0;
}


int main(int argc, char** argv)
{
    //path = "/mnt/c/Users/Michael/Desktop/Research/Data/mnist/";
    char path[1000];
    strcpy(path, "/mnt/c/Users/Michael/Desktop/Research/Data/mnist/");

    int n_hidden=50;

    double** W1;
    double b1[n_hidden];
    double** W2;
    double b2[n_out];
 
    double** grad_W1;
    double grad_b1[n_hidden];
    double** grad_W2;
    double grad_b2[n_out];
    

    double** train_images;
    int train_labels[ntrain];
    double** test_images;
    int test_labels[ntest];
    int** train_labels_onehot;
    
    struct params p;
    struct data d;
    struct params grad;
    struct accuracy results;
    
    int batch_size=200;
    int n_epochs=2;
    double learning_rate=0.02;
    
    if(argc != 5)
    {
	printf("\n\nUsage: ./mlp [n_hidden] [batch_size] [learning_rate] [n_epochs], try ./mlp 50 200 0.02 4\n\n");
	return 1;
    }
    n_hidden = atoi(argv[1]);
    batch_size = atoi(argv[2]);
    learning_rate = atof(argv[3]);
    n_epochs = atoi(argv[4]);
    
    double scale = learning_rate/batch_size;

    printf("\nThis program trains a two layer fully connected neural network to recognize \
handwritten digits (MNIST)\n\nNetwork Size: %d, Minibatch Size: %d, Learning Rate: %.2f, \
Training for %d epochs.\n\n", n_hidden, batch_size, learning_rate, n_epochs);

    int i, j;

    W1 = malloc(img_size * sizeof(double*));
    for(i=0; i<img_size; i++)
	W1[i] = malloc(n_hidden * sizeof(double));

    W2 = malloc(n_hidden * sizeof(double*));
    for(i=0; i<n_hidden; i++)
	W2[i] = malloc(n_out * sizeof(double));

    grad_W1 = malloc(img_size * sizeof(double*));
    for(i=0; i<img_size; i++)
	grad_W1[i] = malloc(n_hidden * sizeof(double));

    grad_W2 = malloc(n_hidden * sizeof(double*));
    for(i=0; i<n_hidden; i++)
	grad_W2[i] = malloc(n_out * sizeof(double));

    printf("\nInitializing weights...\n");

    initialize_weights(W1, W2, b1, b2, n_hidden);

    train_images = malloc(ntrain * sizeof(double*));
    for(i=0; i<ntrain; i++)
	train_images[i] = malloc(img_size * sizeof(double));

    test_images = malloc(ntest * sizeof(double*));
    for(i=0; i<ntest; i++)
	test_images[i] = malloc(img_size * sizeof(double));

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
    load_mnist(&d, path);  //first arg is a pointer to struct data

    for(i=0; i<ntrain; i++)
	onehot(train_labels[i], train_labels_onehot[i]);

    d.train_labels_onehot = train_labels_onehot;

    double** z_hidden;  //(batch_size, n_hidden)
    double** output_hidden; //same
    double** z_out;      //(batch_size, n_out)
    double** output_hidden_transposed;  //do we really need it?

    z_hidden = malloc(batch_size * sizeof(double*));
    for(i=0; i<batch_size; i++)
	z_hidden[i] = malloc(n_hidden * sizeof(double));

    output_hidden = malloc(batch_size * sizeof(double*));
    for(i=0; i<batch_size; i++)
	output_hidden[i] = malloc(n_hidden * sizeof(double));

    output_hidden_transposed = malloc(n_hidden * sizeof(double*));
    for(i=0; i<n_hidden; i++)
	output_hidden_transposed[i] = malloc(batch_size * sizeof(double));

    z_out = malloc(batch_size * sizeof(double*));
    for(i=0; i<batch_size; i++)
	z_out[i] = malloc(n_out * sizeof(double));

    double** error_out;  //output layer errors
    double** error_hidden; //hidden layer errors

    error_out = malloc(batch_size * sizeof(double*));
    for(i=0; i<batch_size; i++)
	error_out[i] = malloc(n_out * sizeof(double));

    error_hidden = malloc(batch_size * sizeof(double*));
    for(i=0; i<batch_size; i++)
	error_hidden[i] = malloc(n_hidden * sizeof(double));



    double*** batches;    //3D array: (nbatches, batch_size, img_size)

    int nbatches = ntrain/batch_size;

    batches = malloc((nbatches) * sizeof(double*));
    for(i=0; i<nbatches; i++)
    {
	batches[i] = malloc(batch_size * sizeof(double*));
	for(j=0; j<batch_size; j++)
	{
	    batches[i][j] = malloc(img_size * sizeof(double));
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

    double** W2_transposed;
    double** batch_transposed;

    W2_transposed = malloc(n_out * sizeof(double*));
    for(i=0; i<n_out; i++)
	W2_transposed[i] = malloc(n_hidden * sizeof(double));

    batch_transposed = malloc(img_size * sizeof(double*));
    for(i=0; i<img_size; i++)
	batch_transposed[i] = malloc(batch_size * sizeof(double));

    double* training_results;
    double* test_results;

    training_results = malloc(n_epochs * sizeof(double));
    test_results = malloc(n_epochs * sizeof(double));
	
    results.training = training_results;
    results.test = test_results;


    double** z_hidden_train;  //(60k, n_hidden)
    double** output_hidden_train; //same
    double** z_out_train;      //(60k, n_out)

    z_hidden_train = malloc(ntrain * sizeof(double*));
    for(i=0; i<ntrain; i++)
	z_hidden_train[i] = malloc(n_hidden * sizeof(double));

    output_hidden_train = malloc(ntrain * sizeof(double*));
    for(i=0; i<ntrain; i++)
	output_hidden_train[i] = malloc(n_hidden * sizeof(double));

    z_out_train = malloc(ntrain * sizeof(double*));
    for(i=0; i<ntrain; i++)
	z_out_train[i] = malloc(n_out * sizeof(double));

    double** z_hidden_test;  //(10k, n_hidden)
    double** output_hidden_test; //same
    double** z_out_test;      //(10k, n_out)

    z_hidden_test = malloc(ntest * sizeof(double*));
    for(i=0; i<ntest; i++)
	z_hidden_test[i] = malloc(n_hidden * sizeof(double));

    output_hidden_test = malloc(ntest * sizeof(double*));
    for(i=0; i<ntest; i++)
	output_hidden_test[i] = malloc(n_hidden * sizeof(double));

    z_out_test = malloc(ntest * sizeof(double*));
    for(i=0; i<ntest; i++)
	z_out_test[i] = malloc(n_out * sizeof(double));


    //***** Training Starts Here ******
    printf("\nTraining network...\n");
    
    clock_t begin = clock();

    for(i=0; i<n_epochs; i++)
    {
	for(j=0; j<nbatches; j++)
	{
	    //should we pass by value, or pass by reference? for example, p.W1 vs &p.W1
	    feedforward(batches[j], z_hidden, output_hidden, z_out, batch_size, n_hidden, &p);

	    substract(z_out, batch_labels[j], error_out, batch_size, n_out);

	    backprop(error_out, batches[j], z_hidden, output_hidden, z_out, p.W2, &grad, batch_size, 
		    n_hidden, error_hidden, output_hidden_transposed, W2_transposed, batch_transposed);

	    update_parameters(&p, &grad, scale, n_hidden);
	}

	test_accuracy(&d, &p, &results, z_hidden_train, output_hidden_train, z_out_train, 
			z_hidden_test, output_hidden_test, z_out_test, n_hidden, i);


	printf("\nEpoch %d: training dataset accuracy: %.2f, test dataset accuracy: %.2f", 
		i, results.training[i], results.test[i]); 

    }

    clock_t end = clock();

    double train_best = max(results.training, n_epochs);
    double test_best = max(results.test, n_epochs);

    printf("\n\nBest Accuracy: %.2f (training dataset), %.2f (test dataset)\n\n", train_best, test_best);
    printf("\n---- Program ran for %.1f seconds ----\n\n", (double)(end - begin)/CLOCKS_PER_SEC);

    return 0;
}
