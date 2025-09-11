#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include<math.h>

//Matrix structure
typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

//Thread data structure
typedef struct {
    Matrix *A;
    Matrix *B;
    Matrix *C;
    int row_start;
    int row_end;
} ThreadData;

//Allocate matrix
Matrix* create_matrix(int rows, int cols, int init_random) {
    Matrix *m = malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m->data[i] = malloc(cols * sizeof(double));
        for (int j = 0; j < cols; j++) {
            m->data[i][j] = init_random ? ((double)rand() / RAND_MAX) * 10.0 : 0.0;
        }
    }
    return m;
}

//Free memory for matrix
void free_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        free(m->data[i]);
    }
    free(m->data);
    free(m);
}

void print_matrix(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            printf("%6.2f ", m->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Get current time in seconds
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Single-threaded matrix multiplication
Matrix* multiply_single(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Error: Incompatible dimensions (%dx%d * %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols);
        return NULL;
    }
    
    Matrix *C = create_matrix(A->rows, B->cols, 0);
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i][k] * B->data[k][j];
            }
            C->data[i][j] = sum;
        }
    }
    return C;
}

//Thread worker function
void* thread_worker(void *arg) {
    ThreadData *td = (ThreadData*) arg;
    for (int i = td->row_start; i < td->row_end; i++) {
        for (int j = 0; j < td->B->cols; j++) {
            double sum = 0;
            for (int k = 0; k < td->A->cols; k++) {
                sum += td->A->data[i][k] * td->B->data[k][j];
            }
            td->C->data[i][j] = sum;
        }
    }
    return NULL;
}

// Multi-threaded matrix multiplication
Matrix* multiply_multi(Matrix *A, Matrix *B, int num_threads) {
    if (A->cols != B->rows) {
        printf("Error: Incompatible dimensions (%dx%d * %dx%d)\n",
               A->rows, A->cols, B->rows, B->cols);
        return NULL;
    }

    Matrix *C = create_matrix(A->rows, B->cols, 0);
    pthread_t threads[num_threads];
    ThreadData td[num_threads];

    int rows_per_thread = A->rows / num_threads;
    int extra = A->rows % num_threads;
    int start = 0;

    for (int t = 0; t < num_threads; t++) {
        int end = start + rows_per_thread + (t < extra ? 1 : 0);
        td[t].A = A;
        td[t].B = B;
        td[t].C = C;
        td[t].row_start = start;
        td[t].row_end = end;
        pthread_create(&threads[t], NULL, thread_worker, &td[t]);
        start = end;
    }

    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    return C;
}

// Verify that two matrices are equal (within tolerance)
int matrices_equal(Matrix *A, Matrix *B, double tolerance) {
    if (A->rows != B->rows || A->cols != B->cols) return 0;
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            if (fabs(A->data[i][j] - B->data[i][j]) > tolerance) {
                return 0;
            }
        }
    }
    return 1;
}

void run_correctness_tests() {
    printf("=== Running Correctness Tests ===\n\n");

    // Test 1: 1x1 * 1x1
    printf("Test 1: 1x1 * 1x1\n");
    Matrix *A = create_matrix(1, 1, 0);
    Matrix *B = create_matrix(1, 1, 0);
    A->data[0][0] = 2; B->data[0][0] = 3;
    
    Matrix *C_single = multiply_single(A, B);
    Matrix *C_multi = multiply_multi(A, B, 2);
    
    printf("Expected: 6.00\n");
    printf("Single-threaded: %.2f\n", C_single->data[0][0]);
    printf("Multi-threaded:  %.2f\n", C_multi->data[0][0]);
    printf("Results match: %s\n\n", matrices_equal(C_single, C_multi, 1e-10) ? "YES" : "NO");
    
    free_matrix(A); free_matrix(B); free_matrix(C_single); free_matrix(C_multi);

    // Test 2: 1x1 * 1x5
    printf("Test 2: 1x1 * 1x5\n");
    A = create_matrix(1, 1, 0);
    B = create_matrix(1, 5, 0);
    A->data[0][0] = 2;
    for (int j = 0; j < 5; j++) B->data[0][j] = j+1;
    
    C_single = multiply_single(A, B);
    C_multi = multiply_multi(A, B, 2);
    
    printf("Expected: [2.00 4.00 6.00 8.00 10.00]\n");
    printf("Single-threaded: ");
    for (int j = 0; j < 5; j++) printf("%.2f ", C_single->data[0][j]);
    printf("\nMulti-threaded:  ");
    for (int j = 0; j < 5; j++) printf("%.2f ", C_multi->data[0][j]);
    printf("\nResults match: %s\n\n", matrices_equal(C_single, C_multi, 1e-10) ? "YES" : "NO");
    
    free_matrix(A); free_matrix(B); free_matrix(C_single); free_matrix(C_multi);

    // Test 3: 2x1 * 1x3
    printf("Test 3: 2x1 * 1x3\n");
    A = create_matrix(2, 1, 0);
    B = create_matrix(1, 3, 0);
    A->data[0][0] = 1; A->data[1][0] = 2;
    B->data[0][0] = 3; B->data[0][1] = 4; B->data[0][2] = 5;
    
    C_single = multiply_single(A, B);
    C_multi = multiply_multi(A, B, 2);
    
    printf("Expected:\n[3.00 4.00 5.00]\n[6.00 8.00 10.00]\n");
    printf("Single-threaded:\n");
    print_matrix(C_single);
    printf("Multi-threaded:\n");
    print_matrix(C_multi);
    printf("Results match: %s\n\n", matrices_equal(C_single, C_multi, 1e-10) ? "YES" : "NO");
    
    free_matrix(A); free_matrix(B); free_matrix(C_single); free_matrix(C_multi);

    // Test 4: 2x2 * 2x2
    printf("Test 4: 2x2 * 2x2\n");
    A = create_matrix(2, 2, 0);
    B = create_matrix(2, 2, 0);
    A->data[0][0] = 1; A->data[0][1] = 2;
    A->data[1][0] = 3; A->data[1][1] = 4;
    B->data[0][0] = 5; B->data[0][1] = 6;
    B->data[1][0] = 7; B->data[1][1] = 8;
    
    C_single = multiply_single(A, B);
    C_multi = multiply_multi(A, B, 2);
    
    printf("Expected:\n[19.00 22.00]\n[43.00 50.00]\n");
    printf("Single-threaded:\n");
    print_matrix(C_single);
    printf("Multi-threaded:\n");
    print_matrix(C_multi);
    printf("Results match: %s\n\n", matrices_equal(C_single, C_multi, 1e-10) ? "YES" : "NO");
    
    free_matrix(A); free_matrix(B); free_matrix(C_single); free_matrix(C_multi);

    // Test 5: 3x4 * 4x3 (additional test)
    printf("Test 5: 3x4 * 4x3\n");
    A = create_matrix(3, 4, 1);
    B = create_matrix(4, 3, 1);
    
    C_single = multiply_single(A, B);
    C_multi = multiply_multi(A, B, 3);
    
    printf("Random matrices - checking consistency between single and multi-threaded\n");
    printf("Results match: %s\n\n", matrices_equal(C_single, C_multi, 1e-10) ? "YES" : "NO");
    
    free_matrix(A); free_matrix(B); free_matrix(C_single); free_matrix(C_multi);

    
    // Test 6: 1x3 * 3x1  -> 1x1
    printf("Test 6: 1x3 * 3x1\n");
    Matrix *A6 = create_matrix(1, 3, 0);
    Matrix *B6 = create_matrix(3, 1, 0);
    // A6 = [1 2 3], B6 = [4; 5; 6], expected = [1*4 + 2*5 + 3*6] = [32]
    A6->data[0][0] = 1; A6->data[0][1] = 2; A6->data[0][2] = 3;
    B6->data[0][0] = 4; B6->data[1][0] = 5; B6->data[2][0] = 6;

    Matrix *C6_single = multiply_single(A6, B6);
    Matrix *C6_multi  = multiply_multi(A6, B6, 3);

    printf("Expected: 32.00\n");
    printf("Single-threaded: %.2f\n", C6_single->data[0][0]);
    printf("Multi-threaded:  %.2f\n", C6_multi->data[0][0]);
    printf("Results match: %s\n\n", matrices_equal(C6_single, C6_multi, 1e-10) ? "YES" : "NO");

    free_matrix(A6); free_matrix(B6); free_matrix(C6_single); free_matrix(C6_multi);

    // Test 7: 3x2 * 2x4  -> 3x4 (non-square x non-square)
    printf("Test 7: 3x2 * 2x4\n");
    Matrix *A7 = create_matrix(3, 2, 0);
    Matrix *B7 = create_matrix(2, 4, 0);
    // A7 = [[1 2],[3 4],[5 6]]
    A7->data[0][0] = 1; A7->data[0][1] = 2;
    A7->data[1][0] = 3; A7->data[1][1] = 4;
    A7->data[2][0] = 5; A7->data[2][1] = 6;
    // B7 = [[7 8 9 10],[11 12 13 14]]
    B7->data[0][0] = 7;  B7->data[0][1] = 8;  B7->data[0][2] = 9;  B7->data[0][3] = 10;
    B7->data[1][0] = 11; B7->data[1][1] = 12; B7->data[1][2] = 13; B7->data[1][3] = 14;

    Matrix *C7_single = multiply_single(A7, B7);
    Matrix *C7_multi  = multiply_multi(A7, B7, 2);

    printf("Expected:\n[29.00 32.00 35.00 38.00]\n[65.00 72.00 79.00 86.00]\n[101.00 112.00 123.00 134.00]\n");
    printf("Single-threaded:\n"); print_matrix(C7_single);
    printf("Multi-threaded:\n");   print_matrix(C7_multi);
    printf("Results match: %s\n\n", matrices_equal(C7_single, C7_multi, 1e-10) ? "YES" : "NO");

    free_matrix(A7); free_matrix(B7); free_matrix(C7_single); free_matrix(C7_multi);
    printf("=== All Correctness Tests Complete ===\n\n");
}

void run_performance_tests() {
    printf("=== Running Performance Tests ===\n\n");
    
    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    int num_thread_counts = 6;
    
    // Test different matrix sizes
    int sizes[] = {100, 300, 500, 800, 1000};
    int num_sizes = 5;
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("Matrix size: %dx%d * %dx%d\n", size, size, size, size);
        printf("Creating large random matrices...\n");
        
        Matrix *A = create_matrix(size, size, 1);
        Matrix *B = create_matrix(size, size, 1);
        
        double single_time = 0;
        printf("\nThread Count | Time (s) | Speedup | Efficiency\n");
        printf("-------------|----------|---------|----------\n");
        
        for (int t = 0; t < num_thread_counts; t++) {
            int num_threads = thread_counts[t];
            
            double start_time = get_time();
            Matrix *C;
            
            if (num_threads == 1) {
                C = multiply_single(A, B);
                single_time = get_time() - start_time;
                printf("%12d | %8.3f |    1.00 |     1.00\n", 
                       num_threads, single_time);
            } else {
                C = multiply_multi(A, B, num_threads);
                double multi_time = get_time() - start_time;
                double speedup = single_time / multi_time;
                double efficiency = speedup / num_threads;
                printf("%12d | %8.3f | %7.2f | %8.2f\n", 
                       num_threads, multi_time, speedup, efficiency);
            }
            
            free_matrix(C);
        }
        
        free_matrix(A);
        free_matrix(B);
        printf("\n");
        
        //For very large matrices, only test the first few sizes to avoid excessive runtime
        if (size >= 500) break;
    }
    
    printf("Testing Complete");
}

int main() {
    srand(time(NULL));
    
    //first make sure everything works correctly
    run_correctness_tests();
    
    //then see how fast we can go
    run_performance_tests();
    
    return 0;
}