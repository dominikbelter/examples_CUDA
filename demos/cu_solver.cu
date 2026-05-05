#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const double*A, int lda)
{
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            printf("%f ", A[i + j*lda]);
        }
        printf("\n");
    }
}

int main()
{
    const int m = 4;
    const int n = 4;
    const int lda = m;
    const int nrhs = 1;

    //A = [1 5 3 2; 4 5 6 5; 2 4 1 4; 3 2 5 8];
    double A[16] = {
        1,4,2,3,
        5,5,4,2,
        3,6,1,5,
        2,5,4,8
    };

    // B=[6,15,4,7]'
    double B[4] = {6,15,4,7};
    double X[4];

    double *d_A, *d_B, *d_tau, *d_work;
    int *devInfo;
    int lwork = 0;

    cusolverDnHandle_t solver;
    cublasHandle_t cublas;

    cusolverDnCreate(&solver);
    cublasCreate(&cublas);

    cudaMalloc(&d_A, sizeof(A));
    cudaMalloc(&d_B, sizeof(B));
    cudaMalloc(&d_tau, sizeof(double)*n);
    cudaMalloc(&devInfo, sizeof(int));

    cudaMemcpy(d_A, A, sizeof(A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(B), cudaMemcpyHostToDevice);

    // workspace
    cusolverDnDgeqrf_bufferSize(solver, m, n, d_A, lda, &lwork);
    cudaMalloc(&d_work, sizeof(double)*lwork);

    // QR
    // [Q,R]=qr(A)
    cusolverDnDgeqrf(solver, m, n, d_A, lda, d_tau, d_work, lwork, devInfo);

    // y = Q' * B;
    cusolverDnDormqr(solver,
                     CUBLAS_SIDE_LEFT,
                     CUBLAS_OP_T,
                     m, nrhs, n,
                     d_A, lda,
                     d_tau,
                     d_B, m,
                     d_work, lwork,
                     devInfo);

    // Solve R x = Q^T B
    //x = R \ y
    double one = 1.0;
    cublasDtrsm(cublas,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT,
                n, nrhs,
                &one,
                d_A, lda,
                d_B, m);

    cudaMemcpy(X, d_B, sizeof(X), cudaMemcpyDeviceToHost);

    printf("Solution:\n");
    printMatrix(n, nrhs, X, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(devInfo);

    cublasDestroy(cublas);
    cusolverDnDestroy(solver);

    return 0;
}
