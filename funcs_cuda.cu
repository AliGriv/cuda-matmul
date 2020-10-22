
#include "funcs_cuda.cuh"
// Number of threads in each thread block
const int blockSize = 16384;


__global__ void VectorAdd_Kernel(const double *a, const double *b, double *c, const int n) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

void VectorAdd_GPU(const double *h_a, const double *h_b, double *h_c, const int n) {

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
    // Device input vectors
    double *dd_a;
    double *dd_b;
    //Device output vector
    double *dd_c;
    // Allocate memory for each vector on GPU
    cudaMalloc(&dd_a, bytes);
    cudaMalloc(&dd_b, bytes);
    cudaMalloc(&dd_c, bytes);
    // Copy host vectors to device
    cudaMemcpy( dd_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( dd_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
//    int blockSize = 10000;

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(dd_a, dd_b, dd_c, n);
    cudaMemcpy( h_c, dd_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory
    cudaFree(dd_a);
    cudaFree(dd_b);
    cudaFree(dd_c);
}

void VectorsClass::VectorAdd_GPU_InClass(const double *h_a, const double *h_b, double *h_c, const int n) {
    // Copy host vectors to device
    cudaMemcpy( this->d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( this->d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads in each thread block
//    int blockSize = 10000;

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(this->d_a, this->d_b, this->d_c, n);
    cudaMemcpy( h_c, this->d_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory

}
void VectorsClass::VectorAdd_GPU_InClass(double *h_c, const int n) {

    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    VectorAdd_Kernel<<<gridSize, blockSize>>>(this->d_a, this->d_b, this->d_c, n);
    cudaMemcpy( h_c, this->d_c, bytes, cudaMemcpyDeviceToHost );
    // Release device memory

}

void MatricesClass::initialize_A_B(const double *h_A, const double *h_B, const int m, const int n, const int k) {
    checkCudaErrors(cudaMemcpy( this->d_A, h_A, m*n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( this->d_B, h_B, n*k*sizeof(double), cudaMemcpyHostToDevice));

}

void MatricesClass::matmul_GPU() {
    checkCudaErrors(cublasCreate(&this->handle));
    // Calculate: c = (alpha*a) * b + (beta*c)
    // MxN = MxK * KxN
    // Signature: handle, operation, operation, M, N, K, alpha, A, lda, B, ldb,
    // beta, C, ldc
    double alpha {1.0};
    double beta {0.0};
    checkCudaErrors(cublasDgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m, this->k, this->n, &alpha, this->d_A, this->m, this->d_B, this->n,
                    &beta, this->d_C, this->m));
    cudaThreadSynchronize();
}
void MatricesClass::retrieve_C(double *h_C) {
//    std::cout << "inside retrive_C" << std::endl;
    checkCudaErrors(cudaMemcpy( h_C, this->d_C, this->bytes_C, cudaMemcpyDeviceToHost ));
//    for (int i {0}; i < m*k; ++i) {
//        std::cout << h_C[i] << std::endl;
//    }
}
void MatricesClassEigen::initialize_A_B(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
    checkCudaErrors(cudaMemcpy( this->d_A, A.data(), A.rows()*A.cols()*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( this->d_B, B.data(), B.rows()*B.cols()*sizeof(double), cudaMemcpyHostToDevice));

}

void MatricesClassEigen::matmul_GPU() {
    checkCudaErrors(cublasCreate(&this->handle));
    // Calculate: c = (alpha*a) * b + (beta*c)
    // MxN = MxK * KxN
    // Signature: handle, operation, operation, M, N, K, alpha, A, lda, B, ldb,
    // beta, C, ldc
    double alpha {1.0};
    double beta {0.0};
    checkCudaErrors(cublasDgemm(this->handle, CUBLAS_OP_N, CUBLAS_OP_N, this->m, this->k, this->n, &alpha, this->d_A, this->m, this->d_B, this->n,
                                &beta, this->d_C, this->m));
    cudaThreadSynchronize();
}
void MatricesClassEigen::retrieve_C(Eigen::MatrixXd &C) {
//    std::cout << "inside retrive_C" << std::endl;
    checkCudaErrors(cudaMemcpy( C.data(), this->d_C, this->bytes_C, cudaMemcpyDeviceToHost ));
//    for (int i {0}; i < m*k; ++i) {
//        std::cout << h_C[i] << std::endl;
//    }
}