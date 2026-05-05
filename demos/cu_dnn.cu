#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cudnn.h>

#define CUDA_CALL(x) do { \
cudaError_t err = (x); \
    if(err != cudaSuccess){ \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
} \
} while(0)

#define CUDNN_CALL(x) do { \
    cudnnStatus_t err = (x); \
    if(err != CUDNN_STATUS_SUCCESS){ \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(err) << std::endl; \
        exit(1); \
} \
} while(0)

    __global__ void dev_const(float *px, float k, int size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) px[tid] = k;
    }

__global__ void dev_iota(float *px, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) px[tid] = (float)tid;
}

void print(const float *data, int n, int c, int h, int w) {
    int size = n*c*h*w;
    std::vector<float> buffer(size);

    CUDA_CALL(cudaMemcpy(buffer.data(), data,
                         size*sizeof(float),
                         cudaMemcpyDeviceToHost));

    int idx = 0;
    for(int ni=0; ni<n; ni++){
        for(int ci=0; ci<c; ci++){
            std::cout << "n=" << ni << ", c=" << ci << "\n";
            for(int hi=0; hi<h; hi++){
                for(int wi=0; wi<w; wi++){
                    std::cout << std::setw(6) << buffer[idx++];
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << std::endl;
}

int main() {

    cudnnHandle_t cudnn;
    CUDNN_CALL(cudnnCreate(&cudnn));

    // INPUT
    const int N=1, C=1, H=5, W=5;
    int in_size = N*C*H*W;

    cudnnTensorDescriptor_t in_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C, H, W));

    float *in_data;
    CUDA_CALL(cudaMalloc(&in_data, in_size*sizeof(float)));

    // FILTER
    const int K=1, FH=2, FW=2;
    int filt_size = K*C*FH*FW;

    cudnnFilterDescriptor_t filt_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        K, C, FH, FW));

    float *filt_data;
    CUDA_CALL(cudaMalloc(&filt_data, filt_size*sizeof(float)));

    // CONV
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        1,1,   // padding
        1,1,   // stride
        1,1,   // dilation
        CUDNN_CONVOLUTION,
        CUDNN_DATA_FLOAT));

    // OUTPUT
    int ON, OC, OH, OW;
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &ON, &OC, &OH, &OW));

    int out_size = ON*OC*OH*OW;

    cudnnTensorDescriptor_t out_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        ON, OC, OH, OW));

    float *out_data;
    CUDA_CALL(cudaMalloc(&out_data, out_size*sizeof(float)));

    // ALGO
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults;

    CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        in_desc,
        filt_desc,
        conv_desc,
        out_desc,
        1,  // request 1 algorithm
        &returnedAlgoCount,
        &perfResults
        ));

    cudnnConvolutionFwdAlgo_t algo = perfResults.algo;

    // WORKSPACE
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc,
        algo, &ws_size));

    void *ws_data = nullptr;
    CUDA_CALL(cudaMalloc(&ws_data, ws_size));

    // INIT DATA
    int block = 256;
    int grid_in = (in_size + block - 1)/block;
    int grid_f = (filt_size + block - 1)/block;

    dev_iota<<<grid_in, block>>>(in_data, in_size);
    dev_const<<<grid_f, block>>>(filt_data, 1.0f, filt_size);

    CUDA_CALL(cudaDeviceSynchronize());

    // CONVOLUTION
    float alpha=1.0f, beta=0.0f;
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data,
        filt_desc, filt_data,
        conv_desc,
        algo,
        ws_data, ws_size,
        &beta,
        out_desc, out_data));

    CUDA_CALL(cudaDeviceSynchronize());

    // PRINT
    std::cout << "Input:\n"; print(in_data, N,C,H,W);
    std::cout << "Filter:\n"; print(filt_data, K,C,FH,FW);
    std::cout << "Output:\n"; print(out_data, ON,OC,OH,OW);

    // CLEANUP
    cudaFree(ws_data);
    cudaFree(out_data);
    cudaFree(filt_data);
    cudaFree(in_data);

    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return 0;
}
