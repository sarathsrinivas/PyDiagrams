#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <atlas/blas.h>
#include <atlas/lapack.h>
#include "lib_flow.h"

void get_zs_fq_samples(double *fq, const double *wt_gma, const double *A1eq, const double *B1es,
		       const double *A2eq, const double *B2es, const double *Csq, unsigned long nq,
		       unsigned long nke)

{
	double *wtb, *gma1, *gma2, *wbc;
	unsigned long i;
	int N, M, K, LDA, LDB, LDC, INCX, INCY;
	unsigned char UPLO, TA, TB;
	double ALPHA, BETA;

	wtb = malloc(nke * nke * sizeof(double));
	assert(wtb);
	gma1 = malloc(nq * nke * sizeof(double));
	assert(gma1);
	gma2 = malloc(nq * nke * sizeof(double));
	assert(gma2);
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
		dsbmv_(&UPLO, &N, &K, &ALPHA, wt_gma, &LDA, &B1es[nke * i], &INCX, &BETA, &wtb[nke * i],
		       &INCY);
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

	dsbmv_(&UPLO, &N, &K, &ALPHA, A1eq, &LDA, wbc, &INCX, &BETA, gma1, &INCY);

	N = nke;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	for (i = 0; i < nke; i++) {
		dsbmv_(&UPLO, &N, &K, &ALPHA, wt_gma, &LDA, &B2es[nke * i], &INCX, &BETA, &wtb[nke * i],
		       &INCY);
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

	dsbmv_(&UPLO, &N, &K, &ALPHA, A2eq, &LDA, wbc, &INCX, &BETA, gma2, &INCY);

	dsbmv_(&UPLO, &N, &K, &ALPHA, gma1, &LDA, gma2, &INCX, &BETA, fq, &INCY);

	free(wbc);
	free(gma2);
	free(gma1);
	free(wtb);
}

void get_zs_fq_weights(double *wt_fq, const double *lkxx_gma, const double *wt_gma, const double *A1,
		       const double *B1, const double *A2, const double *B2, const double *C,
		       unsigned long nq, unsigned long nke)
{
	int N, NRHS, LDA, LDB, INFO;
	unsigned char UPLO;

	get_zs_fq_samples(wt_fq, wt_gma, A1, B1, A2, B2, C, nq, nke);

	UPLO = 'L';
	N = nke;
	NRHS = nq;
	LDA = nke;
	LDB = nke;

	dpotrs_(&UPLO, &N, &NRHS, lkxx_gma, &LDA, wt_fq, &LDB, &INFO);

	assert(INFO == 0);
}

double test_get_zs_fq_samples(unsigned long nke, unsigned long nq, int seed)
{
	double *kl1_ct, *kl2_ct, *ke_ct, *q, *q_ct, *wt_gma, *ktx1, *ktx2, pke[8], gma1, gma2, *fq, *fq2,
	    kmax, st[3], en[3], tmp1, tmp2, *A1, *A2, *B1, *B2, *C, err;
	unsigned int dimke, dimq;
	unsigned long np, i, j, l;
	dsfmt_t drng;

	dimke = 7;
	dimq = 3;
	np = dimke + 1;

	kl1_ct = malloc(dimke * nq * sizeof(double));
	assert(kl1_ct);
	kl2_ct = malloc(dimke * nq * sizeof(double));
	assert(kl2_ct);

	ke_ct = malloc(dimke * nke * sizeof(double));
	assert(ke_ct);
	q_ct = malloc(dimq * nq * sizeof(double));
	assert(q_ct);
	q = malloc(dimq * nq * sizeof(double));
	assert(q);

	wt_gma = malloc(nke * sizeof(double));
	assert(wt_gma);

	ktx1 = malloc(nke * nq * sizeof(double));
	assert(ktx1);
	ktx2 = malloc(nke * nq * sizeof(double));
	assert(ktx2);

	fq = malloc(nq * nke * sizeof(double));
	assert(fq);
	fq2 = malloc(nq * nke * sizeof(double));
	assert(fq2);

	A1 = malloc(nke * nq * sizeof(double));
	assert(A1);
	A2 = malloc(nke * nq * sizeof(double));
	assert(A2);
	B1 = malloc(nke * nke * sizeof(double));
	assert(B1);
	B2 = malloc(nke * nke * sizeof(double));
	assert(B2);
	C = malloc(nke * nq * sizeof(double));
	assert(C);

	kmax = 2.0;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ext_momenta_3ball_7d_ct(ke_ct, nke, st, en, seed);

	fill_ext_momenta_ball(q, nq, st[0], en[0], seed + 443);

	sph_ct_mom_ball(q_ct, q, dimq, nq);

	dsfmt_init_gen_rand(&drng, seed + 95);

	for (i = 0; i < np; i++) {
		pke[i] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);
	}

	dsfmt_init_gen_rand(&drng, seed + 9123);

	for (i = 0; i < nke; i++) {
		wt_gma[i] = 2.0 * dsfmt_genrand_close_open(&drng);
	}

	for (i = 0; i < nke; i++) {

		get_zs_loop_mom_7d_ct(kl1_ct, kl2_ct, &ke_ct[dimke * i], dimke, q_ct, nq, dimq);

		get_krn_se_ard(ktx1, kl1_ct, ke_ct, nq, nke, dimke, pke, np);
		get_krn_se_ard(ktx2, kl2_ct, ke_ct, nq, nke, dimke, pke, np);

		for (l = 0; l < nq; l++) {

			gma1 = 0;
			gma2 = 0;
			for (j = 0; j < nke; j++) {
				gma1 += wt_gma[j] * ktx1[l * nke + j];
				gma2 += wt_gma[j] * ktx2[l * nke + j];
			}

			fq[i * nq + l] = gma1 * gma2;
		}
	}

	get_zs_covar_Aeq(A1, A2, ke_ct, q_ct, nke, dimke, nq, dimq, pke, np);
	get_zs_covar_Bes(B1, B2, ke_ct, ke_ct, nke, dimke, pke, np);
	get_zs_covar_Cqs(C, ke_ct, q_ct, nke, dimke, nq, dimq, pke, np);

	get_zs_fq_samples(fq2, wt_gma, A1, B1, A2, B2, C, nq, nke);

	err = 0;
	for (i = 0; i < nq * nke; i++) {
		if (DEBUG) {
			printf("%+.15E %+.15E\n", fq[i], fq2[i]);
		}
		err += fabs(fq[i] - fq2[i]);
	}

	free(C);
	free(B2);
	free(B1);
	free(A2);
	free(A1);
	free(fq);
	free(fq2);
	free(ktx2);
	free(ktx1);
	free(wt_gma);
	free(q);
	free(q_ct);
	free(ke_ct);
	free(kl2_ct);
	free(kl1_ct);

	return err;
}
