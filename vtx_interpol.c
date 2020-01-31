#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <blas/blas.h>
#include <blas/lapack.h>
#include "vtx_interpol.h"

void get_covar_chd_noise(double *lknxx, const double *kxx, const double *var, unsigned long nx)
{
	int INFO, N, LDA, INCX, INCY;
	unsigned char UPLO;
	unsigned long i;

	N = nx * nx;
	INCX = 1;
	INCY = 1;

	dcopy_(&N, kxx, &INCX, lknxx, &INCY);

	if (var) {
		for (i = 0; i < nx; i++) {
			lknxx[i * nx + i] += var[i] + 1E-7;
		}
	} else {
		for (i = 0; i < nx; i++) {
			lknxx[i * nx + i] += 1E-7;
		}
	}

	N = nx;
	LDA = nx;
	UPLO = 'L';

	dpotrf_(&UPLO, &N, lknxx, &LDA, &INFO);

	assert(INFO == 0);
}

void get_gma_weight(double *wt_gma, const double *lknxx_gma, const double *gma, unsigned long nke)
{

	int N, LDA, LDB, NRHS, INCX, INCY, INFO;
	unsigned char UPLO;

	N = nke;
	INCX = 1;
	INCY = 1;

	dcopy_(&N, gma, &INCX, wt_gma, &INCY);

	UPLO = 'L';
	N = nke;
	NRHS = 1;
	LDA = nke;
	LDB = nke;

	dpotrs_(&UPLO, &N, &NRHS, lknxx_gma, &LDA, wt_gma, &LDB, &INFO);

	assert(INFO == 0);
}

void interpolate_gma(double *gma, const double *wt_gma, const double *Aeq, const double *Bes,
		     const double *Csq, unsigned long nq, unsigned long nke)

{
	double *wtb, *wbc;
	unsigned long i;
	int N, M, K, LDA, LDB, LDC, INCX, INCY;
	unsigned char UPLO, TA, TB;
	double ALPHA, BETA;

	wtb = malloc(nke * nke * sizeof(double));
	assert(wtb);
	wbc = malloc(nq * nke * sizeof(double));
	assert(wbc);

	N = nke;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	for (i = 0; i < nke; i++) {
		dsbmv_(&UPLO, &N, &K, &ALPHA, wt_gma, &LDA, &Bes[nke * i], &INCX, &BETA,
		       &wtb[nke * i], &INCY);
	}

	TA = 'N';
	TB = 'N';
	M = nq;
	K = nke;
	N = nke;
	LDA = nq;
	LDB = nke;
	LDC = nq;

	dgemm_(&TA, &TB, &M, &N, &K, &ALPHA, Csq, &LDA, wtb, &LDB, &BETA, wbc, &LDC);

	N = nke * nq;
	K = 0;
	LDA = 1;

	dsbmv_(&UPLO, &N, &K, &ALPHA, Aeq, &LDA, wbc, &INCX, &BETA, gma, &INCY);

	free(wtb);
	free(wbc);
}

void add_mean(double *gma, const double *gma_mn, unsigned long ngma)
{

	double ALPHA;
	int N, INCX, INCY;

	N = ngma;
	ALPHA = 1.0;
	INCX = 1;
	INCY = 1;

	daxpy_(&N, &ALPHA, gma_mn, &INCX, gma, &INCY);
}

void subtract_mean(double *gma, const double *gma_mn, unsigned long ngma)
{

	double ALPHA;
	int N, INCX, INCY;

	N = ngma;
	ALPHA = -1.0;
	INCX = 1;
	INCY = 1;

	daxpy_(&N, &ALPHA, gma_mn, &INCX, gma, &INCY);
}
