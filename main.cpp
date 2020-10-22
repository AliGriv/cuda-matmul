#include <iostream>
#include <stdio.h>
#include <chrono>
#include "funcs.h"
#include "funcs_cuda.cuh"
#include <cmath>
#include <thread>
#include <cassert>
#include <cublas_v2.h>
#include <eigen3/Eigen/Dense>
const int N = 1000;
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
/* C = A*B */
void matmul_CPU(double *C, const double *A, const double *B, int M, int N, int K)
{
    // Dimensions for our matrices
    // MxK * KxN = MxN
    // For every row...
    for (int row = 0; row < M; row++) {
        // For every column
        for (int col = 0; col < N; col++) {
            // For every element in the row-col pair...
            float temp = 0;
            for (int i = 0; i < K; i++) {
                temp += A[row + M * i] * B[col * K + i];
            }
            C[col * M + row] = temp;
//            // Check to see if the difference falls within our tolerance
//            assert(fabs(c[col * M + row] - temp) <= epsilon);
        }
    }
}

void generate_random_array(double *a, int size_a) {
    for (int i {0}; i < size_a; ++i) {
        a[i] = (double) 10.0*rand()/RAND_MAX;
    }
}
void generate_random_array(double *a, int size_a, Eigen::MatrixXd &a_e) {
    assert(size_a == (a_e.rows() * a_e.cols()));
    for (int i {0}; i < size_a; ++i) {
        a[i] = (double) 10.0*rand()/RAND_MAX;
        a_e(i) = a[i];
    }
}
void compare_two_matrices(const double *C_cpu, const double *C_cublas, const int rows, const int cols) {
    double epsilon {0.1};
    for (int i {0}; i < rows; ++i) {
        for (int j {0}; j < cols; ++j) {
            assert(abs((C_cpu[j*rows + i] - C_cublas[j*rows + i]) < epsilon));
        }
    }
}
int main() {
    /* declare matrices dimensions */
    /*A(m,n), B(n,k)-> C(m,k) */
    int m {900};
    int n {900};
    int k {900};

    double *A, *B, *C_1, *C_2;
    A = new double [m*n];
    B = new double [n*k];
    C_1 = new double [m*k];
    C_2 = new double [m*k];

    Eigen::MatrixXd A_e(m,n);
    Eigen::MatrixXd B_e(n,k);
    Eigen::MatrixXd C_e1(m,k);
    Eigen::MatrixXd C_e2(m,k);
//    generate_random_array(A, m*n);
//    generate_random_array(B, n*k);
    generate_random_array(A, m*n, A_e);
    generate_random_array(B, n*k, B_e);

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
    for (int i {0}; i < 10; ++i) {
        std::cout << C_1[i] << "    " << C_2[i] << std::endl;
    }
    compare_two_matrices(C_1, C_2, m, k);


    std::cout << "Next we are going to see if we can call the eigen matrix in cuda files and read the data directly" << std::endl;
//    std::cout << "A_e is \n" << A_e << std::endl;
    begin = std::chrono::high_resolution_clock::now();
    C_e1 = A_e*B_e;
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on CPU with Eigen-lib!" << std::endl;

    MatricesClassEigen mat_class_eigen(A_e, B_e);
    mat_class_eigen.initialize_A_B(A_e, B_e);
    begin = std::chrono::high_resolution_clock::now();
    mat_class_eigen.matmul_GPU();
    mat_class_eigen.retrieve_C(C_e2);
    end = std::chrono::high_resolution_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(end - begin);
    std::cout << "It took " << elapsed.count() << " seconds to compute on GPU with passing eigen!" << std::endl;
    std::this_thread::sleep_for (std::chrono::seconds(1));
    std::cout << "C_eigen \t \t C_cublas" << std::endl;
    std::cout << "We'll check if the answers of two methods are similar" << std::endl;
    for (int i {0}; i < 10; ++i) {
        std::cout << C_e1(i) << "    " << C_e2(i) << std::endl;
    }
    delete [] A;
    delete [] B;
    delete [] C_1;
    delete [] C_2;
    return 0;
}
