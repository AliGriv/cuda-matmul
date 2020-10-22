//
// Created by AliGriv on 2020-10-19.
//
#ifndef _FUNCS_CUDA_H
#define _FUNCS_CUDA_H
/* Gpu code for adding two vecs */
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <cmath>
#include "cublas_v2.h"
// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_string.h>
#include <eigen3/Eigen/Dense>
__global__ void VectorAdd_Kernel(const double *a, const double *b, double *c, const int n);
void VectorAdd_GPU(const double *h_a, const double *h_b, double *h_c, const int n);

class VectorsClass {
private:
    double *d_a;
    double *d_b;
    double *d_c;
    size_t bytes;
public:
    VectorsClass(int N) {
        bytes = N*sizeof(double);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
    }
    VectorsClass(const double *h_a, const double *h_b, int N) {
        bytes = N*sizeof(double);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cudaMemcpy( this->d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy( this->d_b, h_b, bytes, cudaMemcpyHostToDevice);
    }
    void VectorAdd_GPU_InClass(const double *h_a, const double *h_b, double *h_c, const int n);
    void VectorAdd_GPU_InClass(double *h_c, const int n);
    ~VectorsClass() {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
};

class MatricesClass {
private:
    double *d_A; /*m-by-n*/
    double *d_B; /*n-by-k*/
    double *d_C; /*m-by-k*/
    size_t bytes_A;
    size_t bytes_B;
    size_t bytes_C;
    int m;
    int n;
    int k;
    cublasHandle_t handle;
public:
    MatricesClass(int m, int n, int k):
    bytes_A(m*n*sizeof(double)),
    bytes_B(n*k*sizeof(double)),
    bytes_C(m*k*sizeof(double)),
    m(m), n(n), k(k){
        cudaMalloc(&d_A, bytes_A);
        cudaMalloc(&d_B, bytes_B);
        cudaMalloc(&d_C, bytes_C);
    }
    void initialize_A_B(const double *h_A, const double *h_B, const int m, const int n, const int k);
    void matmul_GPU();
    void retrieve_C(double *h_C);
    ~MatricesClass() {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }
};

class MatricesClassEigen {
private:
    double *d_A; /*m-by-n*/
    double *d_B; /*n-by-k*/
    double *d_C; /*m-by-k*/
    size_t bytes_A;
    size_t bytes_B;
    size_t bytes_C;
    int m;
    int n;
    int k;
    cublasHandle_t handle;
public:
    MatricesClassEigen(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B):
            bytes_A(A.rows()*A.cols()*sizeof(double)),
            bytes_B(B.rows()*B.cols()*sizeof(double)),
            bytes_C(A.rows()*B.cols()*sizeof(double)),
            m(A.rows()), n(B.rows()), k(B.cols()){
        cudaMalloc(&d_A, bytes_A);
        cudaMalloc(&d_B, bytes_B);
        cudaMalloc(&d_C, bytes_C);
    }
    void initialize_A_B(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);
    void matmul_GPU();
    void retrieve_C(Eigen::MatrixXd &C);
    ~MatricesClassEigen() {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }
};

#endif