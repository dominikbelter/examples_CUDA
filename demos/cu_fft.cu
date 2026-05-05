#include <curand.h>
#include <iostream>

using namespace std;

void GPU_fill_rand(double *A, int nr_rows_A, int nr_cols_A)
{
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);

    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    //curandGenerateUniformDouble(prng, A, nr_rows_A * nr_cols_A);
    curandGenerateNormalDouble(prng, A, nr_rows_A * nr_cols_A, 0.0, 1.0);
}


int main(void)
{
    double   *hst_Mat , *dev_Mat;

    int Height = 4 ;
    int Width  = 4 ;
    int vSize = Height*Width ;
    int mSize = sizeof(double)*vSize ;

    hst_Mat = (double *)malloc(mSize) ;
    cudaMalloc((void**)&dev_Mat, mSize) ;

    memset(hst_Mat, 0, mSize) ;
    cudaMemset(dev_Mat, 0, mSize) ;

    GPU_fill_rand(dev_Mat, Height, Width) ;

    cudaMemcpy(hst_Mat, dev_Mat, mSize, cudaMemcpyDeviceToHost) ;

    cout << " * Result matrix : " << endl << "     " ;
    for(int i=0 ;i<Height ; i++)
    {
        for(int j=0 ; j<Width ; j++)
            cout << "   " << hst_Mat[i*Width+j] ;
        cout << endl << "     " ;
    }
    cout << endl << endl ;

    free(hst_Mat) ;
    cudaFree(dev_Mat) ;

    return 0;
}
