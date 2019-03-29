#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <atlas/blas.h>
#include <atlas/lapack.h>
#include "lib_flow.h"

void get_zs_fq_weights(double *wt_fq, const double *lkqq, const double *wt_gma, const double *A1,
		       const double *B1, const double *A2, const double *B2, const double *C,
		       const double *var_gma12, unsigned int ke_flag, unsigned long nq, unsigned long nke)
{
	double *wt_fq_0, *var_fq, *var_wt, ALPHA, BETA;
	unsigned long i;
	int N, NRHS, LDA, LDB, INFO, K, INCX, INCY;
	unsigned char UPLO;

	wt_fq_0 = malloc(nq * nke * sizeof(double));
	assert(wt_fq_0);
	var_fq = malloc(nq * sizeof(double));
	assert(var_fq);

	/* WT^0 */

	get_zs_fq_samples(wt_fq_0, var_fq, wt_gma, A1, B1, A2, B2, C, var_gma12, ke_flag, nq, nke);

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lkqq, &LDA, wt_fq_0, &LDB, &INFO);

	assert(INFO == 0);

	/* DIAG(V_F) x WT^0 */

	N = nq;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	for (i = 0; i < nke; i++) {
		dsbmv_(&UPLO, &N, &K, &ALPHA, var_fq, &LDA, &wt_fq_0[i * nq], &INCX, &BETA, &wt_fq[i * nq],
		       &INCY);
	}

	/* K^-1 DIAG(V_F) WT^0 */

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lkqq, &LDA, wt_fq, &LDB, &INFO);

	assert(INFO == 0);

	/* WT = WT^0 + K^-1 DIAG(V_F) WT^0  */

	N = nq * nke;
	INCX = 1;
	INCY = 1;
	ALPHA = 1.0;

	daxpy_(&N, &ALPHA, wt_fq_0, &INCX, wt_fq, &INCY);

	free(wt_fq_0);
}

void get_gma_gpr_mean(double *gma_gpr, const double *Ifq, const double *wfq, unsigned long nke,
		      unsigned long nq)
{
	int N, INCX, INCY;
	unsigned long i;

	N = nq;
	INCX = 1;
	INCY = 1;

	for (i = 0; i < nke; i++) {
		gma_gpr[i] = ddot_(&N, &wfq[i * nq], &INCX, &Ifq[i * nq], &INCY);
	}
}

void get_gma_gpr_var(double *var_gma_gpr, const double *var_fq, const double *II, const double *Iqe,
		     const double *lkqq, unsigned long nq, unsigned long nke)
{
	double *I_0, *I_cv, ALPHA, BETA, tmp;
	unsigned long i;
	int N, NRHS, LDA, LDB, INFO, K, INCX, INCY;
	unsigned char UPLO;

	I_0 = malloc(nq * nke * sizeof(double));
	assert(I_0);
	I_cv = malloc(nq * nke * sizeof(double));
	assert(I_cv);

	for (i = 0; i < nq * nke; i++) {
		I_0[i] = Iqe[i];
	}

	/* I^0 = K^-1 x I */

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lkqq, &LDA, I_0, &LDB, &INFO);

	assert(INFO == 0);

	/* I_CV = DIAG(V_F) x I^0 */

	N = nq;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	for (i = 0; i < nke; i++) {
		dsbmv_(&UPLO, &N, &K, &ALPHA, var_fq, &LDA, &I_0[i * nq], &INCX, &BETA, &I_cv[i * nq], &INCY);
	}

	/* I_CV = K^-1 x DIAG(V_F) x I^0 */

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lkqq, &LDA, I_cv, &LDB, &INFO);

	assert(INFO == 0);

	/* I_CV = I^0 + K^-1 DIAG(V_F) I^0  */

	N = nq * nke;
	INCX = 1;
	INCY = 1;
	ALPHA = 1.0;

	daxpy_(&N, &ALPHA, I_0, &INCX, I_cv, &INCY);

	/* V_GMA = II - DIAG(I^T I_CV) */

	for (i = 0; i < nke; i++) {

		tmp = ddot_(&N, &Iqe[i * nq], &INCX, &I_cv[i * nq], &INCY);

		var_gma_gpr[i] = II[i] - tmp;
	}

	free(I_0);
	free(I_cv);
}
