#include <iostream>
#include <stdio.h>
#include <chrono>
#include "funcs.h"
#include "funcs_cuda.cuh"
#include <cmath>
#include <thread>
#include <cassert>
#include <cublas_v2.h>
const int N = 10000000;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/* C = A*B */
void matmul_CPU(double *C, const double *A, const double *B, int hA, int wA,  int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

void generate_random_array(double *a, int size_a) {
    for (int i {0}; i < size_a; ++i) {
        a[i] = (double) 10.0*rand()/RAND_MAX;
    }
}

void compare_two_matrices(const double *C_cpu, const double *C_cublas, const int rows, const int cols) {
    double epsilon {0.001};
    for (int i {0}; i < rows; ++i) {
        for (int j {0}; j < cols; ++j) {
            assert(abs((C_cpu[i*cols + j] - C_cublas[j*rows + i]) < epsilon));
        }
    }
}
int main() {
    /* declare matrices dimensions */
    /*A(m,n), B(n,k)-> C(m,k) */
    int m {9};
    int n {9};
    int k {9};

    double *A, *B, *C_1, *C_2;
    A = new double [m*n];
    B = new double [n*k];
    C_1 = new double [m*k];
    C_2 = new double [m*k];

    generate_random_array(A, m*n);
    generate_random_array(B, n*k);

    std::cout << "A and B are defined" << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    matmul_CPU(C_1, A, B, m, n, k);
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on CPU!" << std::endl;

    MatricesClass mat_class(m,n,k);
    mat_class.initialize_A_B(A, B, m, n, k);
    begin = std::chrono::high_resolution_clock::now();
    mat_class.matmul_GPU();
    mat_class.retrieve_C(C_2);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on GPU!" << std::endl;
    std::this_thread::sleep_for (std::chrono::seconds(1));

    std::cout << "C_cpu \t \t C_cublas" << std::endl;
    std::cout << "We'll check if the answers of two methods are similar" << std::endl;
    for (int i {0}; i < m*k; ++i) {
        std::cout << C_1[i] << "    " << C_2[i] << std::endl;
    }
    compare_two_matrices(C_1, C_2, m, k);
    delete [] A;
    delete [] B;
    delete [] C_1;
    delete [] C_2;
    return 0;
}
