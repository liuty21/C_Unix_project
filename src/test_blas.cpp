#include "cblas.h"
#include <stdio.h>

int main(int argc, char const *argv[])
{
	const enum CBLAS_ORDER order = CblasRowMajor;
	const enum CBLAS_TRANSPOSE trans = CblasNoTrans;
	const int M = 4;
	const int N = 2;
	const int K = 3;
	const float alpha = 1;
	const float beta = 0;
	const int lda = K;
	const int ldb = 4;
	const int ldc = N;
	const float A[4*3]={1,2,3,4,5,6,6,5,4,3,2,1};
	const float B[4*4]={0,0,0,0,0,1,4,0,0,2,5,0,0,3,6,0};
	float C[4*2];

	cblas_sgemm(order, trans, trans, M, N, K, alpha, A, lda, B+5, ldb, beta,C,ldc);

	int i,j;
	for (i = 0; i < M; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			printf("%f ", C[i*N+j]);
		}
		printf("\n");
	}
	return 0;
}