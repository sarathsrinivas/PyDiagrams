#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <atlas/blas.h>
#include <atlas/lapack.h>
#include "lib_flow.h"

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
}

void get_fq_samples(double *fq, double *var_fq, const double *wt_gma, const double *A1,
		    const double *B1, const double *A2, const double *B2, const double *C,
		    const double *var_gma12, unsigned int ke_flag, unsigned long nq,
		    unsigned long nke)

{
	double *gma1, *gma2;
	int N, K, LDA, INCX, INCY;
	unsigned char UPLO;
	double ALPHA, BETA;

	gma1 = malloc(nq * nke * sizeof(double));
	assert(gma1);
	gma2 = malloc(nq * nke * sizeof(double));
	assert(gma2);

	N = nke * nq;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	interpolate_gma(gma1, wt_gma, A1, B1, C, nq, nke);
	interpolate_gma(gma2, wt_gma, A2, B2, C, nq, nke);

	dsbmv_(&UPLO, &N, &K, &ALPHA, gma1, &LDA, gma2, &INCX, &BETA, fq, &INCY);

	if (var_gma12 && var_fq) {
		get_var_fq(var_fq, &gma1[ke_flag * nq], &gma2[ke_flag * nq], var_gma12, nq);
	}

	free(gma2);
	free(gma1);
}

void get_fq_samples_reg(double *fq_reg, double *var_fq, const double *wt_gma, const double *A1,
			const double *B1, const double *A2, const double *B2, const double *C,
			double *var_gma12, const double *reg12, const double *reg1x2,
			unsigned int ke_flag, unsigned long nq, unsigned long nke)

{
	double *gma1, *gma2, *fq, *var_gma12_reg;
	int N, K, LDA, INCX, INCY;
	unsigned char UPLO;
	double ALPHA, BETA;

	gma1 = malloc(nq * nke * sizeof(double));
	assert(gma1);
	gma2 = malloc(nq * nke * sizeof(double));
	assert(gma2);

	fq = malloc(nq * nke * sizeof(double));
	assert(fq);

	var_gma12_reg = malloc(4 * nq * nq * sizeof(double));
	assert(var_gma12_reg);

	N = nke * nq;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	interpolate_gma(gma1, wt_gma, A1, B1, C, nq, nke);
	interpolate_gma(gma2, wt_gma, A2, B2, C, nq, nke);

	dsbmv_(&UPLO, &N, &K, &ALPHA, gma1, &LDA, gma2, &INCX, &BETA, fq, &INCY);

	dsbmv_(&UPLO, &N, &K, &ALPHA, fq, &LDA, reg12, &INCX, &BETA, fq_reg, &INCY);

	if (var_gma12 && var_fq) {

		N = 4 * nq * nq;

		dsbmv_(&UPLO, &N, &K, &ALPHA, reg1x2, &LDA, var_gma12, &INCX, &BETA, var_gma12_reg,
		       &INCY);

		get_var_fq(var_fq, &gma1[ke_flag * nq], &gma2[ke_flag * nq], var_gma12_reg, nq);

		dcopy_(&N, var_gma12_reg, &INCX, var_gma12, &INCY);
	}

	free(gma2);
	free(gma1);
	free(fq);
	free(var_gma12_reg);
}

void get_fq_as_samples(double *fq, double *var_fq, const double *wt_gma_zs,
		       const double *wt_gma_zsp, const double *A1, const double *A2,
		       const double *B1, const double *B2, const double *C, const double *A1p,
		       const double *A2p, const double *B1p, const double *B2p, const double *Cp,
		       const double *var_gma12, unsigned int ke_flag, unsigned long nq,
		       unsigned long nke)
{
	double *gma1_zs, *gma2_zs, *gma1_zsp, *gma2_zsp;
	int N, K, LDA, INCX, INCY;
	unsigned char UPLO;
	double ALPHA, BETA;

	gma1_zs = malloc(nq * nke * sizeof(double));
	assert(gma1_zs);
	gma2_zs = malloc(nq * nke * sizeof(double));
	assert(gma2_zs);
	gma1_zsp = malloc(nq * nke * sizeof(double));
	assert(gma1_zsp);
	gma2_zsp = malloc(nq * nke * sizeof(double));
	assert(gma2_zsp);

	interpolate_gma(gma1_zs, wt_gma_zs, A1, B1, C, nq, nke);
	interpolate_gma(gma2_zs, wt_gma_zs, A2, B2, C, nq, nke);

	interpolate_gma(gma1_zsp, wt_gma_zsp, A1p, B1p, Cp, nq, nke);
	interpolate_gma(gma2_zsp, wt_gma_zsp, A2p, B2p, Cp, nq, nke);

	ALPHA = -1.0;
	N = nke * nq;
	INCX = 1;
	INCY = 1;

	/* gma1 = gma1_zs - gma1_zsp */

	daxpy_(&N, &ALPHA, gma1_zsp, &INCX, gma1_zs, &INCY);

	/* gma2 = gma2_zs - gma2_zsp */

	daxpy_(&N, &ALPHA, gma2_zsp, &INCX, gma2_zs, &INCY);

	N = nke * nq;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	/* fq = gma1 *  gma2 */

	dsbmv_(&UPLO, &N, &K, &ALPHA, gma1_zs, &LDA, gma2_zs, &INCX, &BETA, fq, &INCY);

	if (var_gma12 && var_fq) {
		get_var_fq(var_fq, &gma1_zs[ke_flag * nq], &gma2_zs[ke_flag * nq], var_gma12, nq);
	}

	free(gma1_zs);
	free(gma1_zsp);
	free(gma2_zs);
	free(gma2_zsp);
}

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
		var_fq[i]
		    = var_fq_12[i] + var_fq_12[nq + i] + fabs(var_gma12[(2 * nq) * i + (nq + i)]);
	}

	free(var_fq_12);
	free(var_gma12_diag);
	free(gma12);
}

double test_get_zs_fq_samples(unsigned long nke, unsigned long nq, int seed)
{
	double *kl1_ct, *kl2_ct, *ke_ct, *q, *q_ct, *wt_gma, *ktx1, *ktx2, pke[8], gma1, gma2, *fq,
	    *fq2, kmax, st[3], en[3], tmp1, tmp2, *A1, *A2, *B1, *B2, *C, err;
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

double test_get_zs_fq_as_samples(unsigned long nke, unsigned long nq, int seed)
{
	double *kl1_ct, *kl2_ct, *ke_ct, *kl1p_ct, *kl2p_ct, *kep_ct, *q, *q_ct, *wt_gma_zs,
	    *wt_gma_zsp, *ktx1_zs, *ktx2_zs, *ktx1_zsp, *ktx2_zsp, pke_zs[8], pke_zsp[8], gma1_zs,
	    gma2_zs, gma1_zsp, gma2_zsp, *fq, *fq2, kmax, st[3], en[3], tmp1, tmp2, *A1, *A2, *B1,
	    *B2, *C, *A1p, *A2p, *B1p, *B2p, *Cp, err;
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

	wt_gma_zs = malloc(nke * sizeof(double));
	assert(wt_gma_zs);
	wt_gma_zsp = malloc(nke * sizeof(double));
	assert(wt_gma_zsp);

	ktx1_zs = malloc(nke * nq * sizeof(double));
	assert(ktx1_zs);
	ktx2_zs = malloc(nke * nq * sizeof(double));
	assert(ktx2_zs);

	ktx1_zsp = malloc(nke * nq * sizeof(double));
	assert(ktx1_zsp);
	ktx2_zsp = malloc(nke * nq * sizeof(double));
	assert(ktx2_zsp);

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

	A1p = malloc(nke * nq * sizeof(double));
	assert(A1p);
	A2p = malloc(nke * nq * sizeof(double));
	assert(A2p);
	B1p = malloc(nke * nke * sizeof(double));
	assert(B1p);
	B2p = malloc(nke * nke * sizeof(double));
	assert(B2p);
	Cp = malloc(nke * nq * sizeof(double));
	assert(Cp);

	kmax = 2.0;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ke_sample_zs_ct(ke_ct, nke, st, en, seed);

	fill_ext_momenta_ball(q, nq, st[0], en[0], seed + 443);

	sph_to_ct(q_ct, q, dimq, nq);

	dsfmt_init_gen_rand(&drng, seed + 95);

	for (i = 0; i < np; i++) {
		pke_zs[i] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);
		pke_zsp[i] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);
	}

	dsfmt_init_gen_rand(&drng, seed + 9123);

	for (i = 0; i < nke; i++) {
		wt_gma_zs[i] = 2.0 * dsfmt_genrand_close_open(&drng);
		wt_gma_zsp[i] = 2.0 * dsfmt_genrand_close_open(&drng);
	}

	for (i = 0; i < nke; i++) {

		get_zs_loop_mom_7d_ct(kl1_ct, kl2_ct, &ke_ct[dimke * i], dimke, q_ct, nq, dimq);

		get_krn_se_ard(ktx1_zs, kl1_ct, ke_ct, nq, nke, dimke, pke_zs, np);
		get_krn_se_ard(ktx2_zs, kl2_ct, ke_ct, nq, nke, dimke, pke_zs, np);

		get_krn_se_ard(ktx1_zsp, kl1_ct, ke_ct, nq, nke, dimke, pke_zsp, np);
		get_krn_se_ard(ktx2_zsp, kl2_ct, ke_ct, nq, nke, dimke, pke_zsp, np);

		for (l = 0; l < nq; l++) {

			gma1_zs = 0;
			gma2_zs = 0;
			gma1_zsp = 0;
			gma2_zsp = 0;
			for (j = 0; j < nke; j++) {
				gma1_zs += wt_gma_zs[j] * ktx1_zs[l * nke + j];
				gma2_zs += wt_gma_zs[j] * ktx2_zs[l * nke + j];

				gma1_zsp += wt_gma_zsp[j] * ktx1_zsp[l * nke + j];
				gma2_zsp += wt_gma_zsp[j] * ktx2_zsp[l * nke + j];
			}

			fq[i * nq + l] = (gma1_zs - gma1_zsp) * (gma2_zs - gma2_zsp);
		}
	}

	get_zs_covar_Aeq(A1, A2, ke_ct, q_ct, nke, dimke, nq, dimq, pke_zs, np);
	get_zs_covar_Bes(B1, B2, ke_ct, ke_ct, nke, dimke, pke_zs, np);
	get_zs_covar_Cqs(C, ke_ct, q_ct, nke, dimke, nq, dimq, pke_zs, np);

	get_zs_covar_Aeq(A1p, A2p, ke_ct, q_ct, nke, dimke, nq, dimq, pke_zsp, np);
	get_zs_covar_Bes(B1p, B2p, ke_ct, ke_ct, nke, dimke, pke_zsp, np);
	get_zs_covar_Cqs(Cp, ke_ct, q_ct, nke, dimke, nq, dimq, pke_zsp, np);

	get_fq_as_samples(fq2, NULL, wt_gma_zs, wt_gma_zsp, A1, A2, B1, B2, C, A1p, A2p, B1p, B2p,
			  Cp, NULL, 0, nq, nke);

	err = 0;
	for (i = 0; i < nq * nke; i++) {
		if (DEBUG) {
			printf("%+.15E %+.15E\n", fq[i], fq2[i]);
		}
		err += fabs(fq[i] - fq2[i]);
	}

	free(Cp);
	free(B2p);
	free(B1p);
	free(A2p);
	free(A1p);
	free(C);
	free(B2);
	free(B1);
	free(A2);
	free(A1);
	free(fq2);
	free(fq);
	free(ktx2_zsp);
	free(ktx1_zsp);
	free(ktx2_zs);
	free(ktx1_zs);
	free(wt_gma_zsp);
	free(wt_gma_zs);
	free(q);
	free(q_ct);
	free(ke_ct);
	free(kl2_ct);
	free(kl1_ct);

	return err;
}
