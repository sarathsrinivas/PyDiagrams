#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <atlas/blas.h>
#include <atlas/lapack.h>
#include "lib_flow.h"

void get_noise_covar_chd(double *lknxx, const double *kxx, const double *var, unsigned long nx)
{
	int INFO, N, LDA, INCX, INCY;
	unsigned char UPLO;
	unsigned long i;

	N = nx * nx;
	INCX = 1;
	INCY = 1;

	dcopy_(&N, kxx, &INCX, lknxx, &INCY);

	for (i = 0; i < nx; i++) {
		lknxx[i * nx + i] += var[i];
	}

	N = nx;
	LDA = nx;
	UPLO = 'L';

	dpotrf_(&UPLO, &N, lknxx, &LDA, &INFO);

	assert(INFO == 0);
}

void get_zs_fq_weights(double *wt_fq, const double *lknqq, unsigned long nq, unsigned long nke)
{
	/* wt_fq SHOULD HAVE FQ_SAMPLES ON INPUT */

	int N, NRHS, LDA, LDB, INFO, K, INCX, INCY;
	unsigned char UPLO;

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lknqq, &LDA, wt_fq, &LDB, &INFO);

	assert(INFO == 0);
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

void get_gma_gpr_var(double *var_gma_gpr, const double *II, const double *Iqe, const double *lknqq,
		     unsigned long nq, unsigned long nke)
{
	double *Icv, ALPHA, BETA, tmp;
	unsigned long i;
	int N, NRHS, LDA, LDB, INFO, K, INCX, INCY;
	unsigned char UPLO;

	Icv = malloc(nq * nke * sizeof(double));
	assert(Icv);

	N = nq * nke;
	INCX = 1;
	INCY = 1;

	dcopy_(&N, Iqe, &INCX, Icv, &INCY);

	/* ICV = K^-1 x I */

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lknqq, &LDA, Icv, &LDB, &INFO);

	assert(INFO == 0);

	/* V_GMA = II - DIAG(I^T I_CV) */

	N = nq;
	INCX = 1;
	INCY = 1;

	for (i = 0; i < nke; i++) {

		tmp = ddot_(&N, &Iqe[i * nq], &INCX, &Icv[i * nq], &INCY);

		var_gma_gpr[i] = II[i] - tmp;
	}

	free(Icv);
}

void get_gma_weight(double *wt_gma, const double *lknxx_gma, const double *gma, unsigned long nke)
{

	int N, LDA, LDB, NRHS, INCX, INCY, INFO;
	unsigned char UPLO;

	N = nke * nke;
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

void get_noisy_inverse(double *wt, const double *lkqq, const double *var, const double *y, unsigned long nq,
		       unsigned long nke)
{

	double *wt0, ALPHA, BETA;
	int N, K, LDA, LDB, NRHS, INFO, INCX, INCY;
	unsigned long i, j;
	unsigned char UPLO;

	wt0 = malloc(nq * nke * sizeof(double));
	assert(wt0);

	/* WT^0 */

	N = nq * nke;
	INCX = 1;
	INCY = 1;

	dcopy_(&N, y, &INCX, wt0, &INCY);

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lkqq, &LDA, wt0, &LDB, &INFO);

	assert(INFO == 0);

	/* -1.0 * DIAG(V_F) x WT^0 */

	N = nq;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = -1.0;
	BETA = 0.0;

	for (i = 0; i < nke; i++) {
		dsbmv_(&UPLO, &N, &K, &ALPHA, var, &LDA, &wt0[i * nq], &INCX, &BETA, &wt[i * nq], &INCY);
	}

	/* -1.0 * K^-1 DIAG(V_F) WT^0 */

	UPLO = 'L';
	N = nq;
	NRHS = nke;
	LDA = nq;
	LDB = nq;

	dpotrs_(&UPLO, &N, &NRHS, lkqq, &LDA, wt, &LDB, &INFO);

	assert(INFO == 0);

	/* WT = WT^0 - K^-1 DIAG(V_F) WT^0  */

	N = nq * nke;
	INCX = 1;
	INCY = 1;
	ALPHA = 1.0;

	daxpy_(&N, &ALPHA, wt0, &INCX, wt, &INCY);

	free(wt0);
}

double test_noisy_inverse(unsigned long nq, unsigned long nke, double sigma2, int seed)
{
	double *q, *kqq, *knqq, *lkqq, *lknqq, *y, *var_y, *wt, *wtn, p[4], err;
	unsigned int dimq;
	unsigned long np, i, j;
	dsfmt_t drng;
	int N, NRHS, LDA, LDB, INFO;
	unsigned char UPLO;

	fprintf(stderr, "test_noisy_inverse: \n");

	dimq = 3;
	np = 4;

	q = malloc(nq * dimq * sizeof(double));
	assert(q);

	kqq = malloc(nq * nq * sizeof(double));
	assert(kqq);
	lkqq = malloc(nq * nq * sizeof(double));
	assert(lkqq);
	knqq = malloc(nq * nq * sizeof(double));
	assert(knqq);
	lknqq = malloc(nq * nq * sizeof(double));
	assert(lknqq);

	y = calloc(nq * nke, sizeof(double));
	assert(y);
	var_y = malloc(nq * sizeof(double));
	assert(var_y);

	wt = malloc(nq * nke * sizeof(double));
	assert(wt);
	wtn = malloc(nq * nke * sizeof(double));
	assert(wtn);

	fill_ext_momenta_ball(q, nq, 0, 2.5, seed);

	for (i = 0; i < np; i++) {
		p[i] = 1.0;
	}

	dsfmt_init_gen_rand(&drng, seed + 21);

	/*
	for (i = 0; i < nq * nke; i++) {
		y[i] = -2.0 + 2.0 * dsfmt_genrand_close_open(&drng);
	}
	*/
	for (i = 0; i < nq; i++) {
		y[i * nq + i] = 1.0;
	}

	/* EXACT (K+SIGMA2)^-1 */

	get_krn_se_ard(knqq, q, q, nq, nq, dimq, p, np);

	dsfmt_init_gen_rand(&drng, seed + 434);

	for (i = 0; i < nq; i++) {
		var_y[i] = sigma2 * dsfmt_genrand_close_open(&drng);
	}

	for (i = 0; i < nq; i++) {
		knqq[i * nq + i] += var_y[i];
	}

	for (i = 0; i < nq * nke; i++) {
		wtn[i] = y[i];
	}

	UPLO = 'L';
	N = nq;
	LDA = nq;
	LDB = nq;
	NRHS = nke;

	dposv_(&UPLO, &N, &NRHS, knqq, &LDA, wtn, &LDB, &INFO);

	assert(INFO == 0);

	/* APPROX: K^-1 + K^-1 SIGMA2 K^-1 */

	get_krn_se_ard(kqq, q, q, nq, nq, dimq, p, np);

	/* JUST TO GET lkqq */
	get_gpr_weights(wt, lkqq, kqq, nq, dimq, y);

	get_noisy_inverse(wt, lkqq, var_y, y, nq, nke);

	err = 0;
	for (i = 0; i < nke * nq; i++) {
		err += fabs(wtn[i] - wt[i]);
		printf("%+.15E %+.15E %+.15E\n", wtn[i], wt[i], fabs(wtn[i] - wt[i]));
	}

	free(wtn);
	free(wt);
	free(var_y);
	free(y);
	free(lknqq);
	free(knqq);
	free(lkqq);
	free(kqq);
	free(q);

	return err;
}
