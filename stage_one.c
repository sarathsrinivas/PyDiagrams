#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <atlas/blas.h>
#include <atlas/lapack.h>
#include "lib_flow.h"

void get_fq_samples(double *fq, double *var_fq, const double *wt_gma, const double *A1eq, const double *B1es,
		    const double *A2eq, const double *B2es, const double *Csq, const double *var_gma12,
		    unsigned int ke_flag, unsigned long nq, unsigned long nke)

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

	if (var_gma12 && var_fq) {
		get_var_fq(var_fq, &gma1[ke_flag * nq], &gma2[ke_flag * nq], var_gma12, nq);
	}

	free(wbc);
	free(gma2);
	free(gma1);
	free(wtb);
}

/*
void get_var_gma(double *var_gma12, const double *lkxx, const double *ktt12, const double *ktx12, const double
*kl12_ct, unsigned long nke, const double *q_ct, unsigned int dimq, unsigned long nq, const double *pke,
unsigned long npke, unsigned int ke_flag)
{

	double *ktt12, *ktx12, *kl12_ct;
	unsigned long i;

	ktt12 = malloc(4 * nq * nq * sizeof(double));
	assert(ktt12);
	ktx12 = malloc(2 * nq * nke * sizeof(double));
	assert(ktx12);
	kl12_ct = malloc(2 * nq * dimke * sizeof(double));
	assert(kl12_ct);

	get_zs_loop_mom_7d_ct(kl12_ct, &kl12_ct[nq * dimke], &ke_ct[dimke * ke_flag], dimke, q_ct, nq, dimq);

	get_krn_se_ard(ktx12, kl12_ct, ke_ct, 2 * nq, nke, dimke, pke, npke);

	get_krn_se_ard(ktt12, kl12_ct, kl12_ct, 2 * nq, 2 * nq, dimke, pke, npke);

	get_var_mat_chd(var_gma12, ktt12, ktx12, lkxx, 2 * nq, nke);

	free(ktt12);
	free(ktx12);
	free(kl12_ct);
}
*/

void get_var_fq(double *var_fq, const double *gma1, const double *gma2, const double *var_gma12,
		unsigned long nq)
{
	double *gma12, *var_gma12_diag, *var_fq_12, ALPHA, BETA, t1, t2;
	unsigned long i;
	int N, M, LDA, INCX, INCY, K;
	unsigned char UPLO;

	gma12 = malloc(2 * nq * sizeof(double));
	assert(gma12);
	var_gma12_diag = malloc(2 * nq * sizeof(double));
	assert(var_gma12_diag);
	var_fq_12 = malloc(2 * nq * sizeof(double));
	assert(var_fq_12);

	for (i = 0; i < nq; i++) {
		t1 = gma1[i];
		t2 = gma2[i];
		gma12[i] = t2 * t2;
		gma12[nq + i] = t1 * t1;
	}

	for (i = 0; i < 2 * nq; i++) {
		var_gma12_diag[i] = var_gma12[2 * nq * i + i];
	}

	N = 2 * nq;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	dsbmv_(&UPLO, &N, &K, &ALPHA, var_gma12_diag, &LDA, gma12, &INCX, &BETA, var_fq_12, &INCY);

	for (i = 0; i < nq; i++) {
		var_fq[i] = var_fq_12[i] + var_fq_12[nq + i] + var_gma12[(2 * nq) * i + (nq + i)];
	}

	free(var_fq_12);
	free(var_gma12_diag);
	free(gma12);
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

	sph_to_ct(q_ct, q, dimq, nq);

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

	get_fq_samples(fq2, NULL, wt_gma, A1, B1, A2, B2, C, NULL, 0, nq, nke);

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
