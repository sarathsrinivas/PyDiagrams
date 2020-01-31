#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <lib_pots/lib_pots.h>
#include <blas/blas.h>
#include <blas/lapack.h>
#include "vtx_interpol.h"
#include "loop_vtx.h"
#include "split_covar.h"
#include "momentum_sample.h"

#define DIMKE (7)
#define DIMQ (3)

double test_interpolate_gma(unsigned long nke, unsigned long nq, int shft, int seed)
{
	double *ke, *kl1, *kl2, *q, *q_ct, *gma_smp, *gma_tst, *gma_gpr, pke[DIMKE + 1], kmax,
	    st[3], en[3], vparam[2], *work, *kxx, *lkxx, eps, *wt_gma, err;
	unsigned long npke, nwrk, i;
	struct split_covar *scv;
	dsfmt_t drng;

	printf("test_interpolate_gma(nke = %lu, nq = %lu, shft = %d):\n", nke, nq, shft);

	scv = malloc(sizeof(struct split_covar));
	assert(scv);

	npke = DIMKE + 1;

	ke = malloc(DIMKE * nke * sizeof(double));
	assert(ke);
	kl1 = malloc(DIMKE * nq * sizeof(double));
	assert(kl1);
	kl2 = malloc(DIMKE * nq * sizeof(double));
	assert(kl2);
	q = malloc(DIMQ * nq * sizeof(double));
	assert(q);
	q_ct = malloc(DIMQ * nq * sizeof(double));
	assert(q_ct);

	gma_smp = malloc(nke * sizeof(double));
	assert(gma_smp);
	gma_tst = malloc(2 * nq * nke * sizeof(double));
	assert(gma_tst);
	gma_gpr = malloc(2 * nq * nke * sizeof(double));
	assert(gma_gpr);

	kxx = malloc(nke * nke * sizeof(double));
	assert(kxx);
	lkxx = malloc(nke * nke * sizeof(double));
	assert(lkxx);

	wt_gma = malloc(nke * sizeof(double));
	assert(wt_gma);

	kmax = 2.0;
	eps = 0.4;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ph_sample_ct_box(ke, nke, st, en, seed);

	fill_q_sample_ball(q, nq, st[0], en[0], seed + 443);

	sph_to_ct(q_ct, q, DIMQ, nq);

	dsfmt_init_gen_rand(&drng, seed + 95);

	for (i = 0; i < npke; i++) {
		pke[i] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);
	}

	/* GET SAMPLE */

	vparam[0] = kmax;
	vparam[1] = eps;

	fill_pot_vexp_reg_ct(gma_smp, ke, nke, DIMKE, vparam);

	/* GET GPR PREDICTION FROM lib_gpr */

	for (i = 0; i < nke; i++) {

		get_ph_loop_mom(kl1, kl2, &ke[DIMKE * i], DIMKE, q_ct, nq, DIMQ, shft);

		gpr_interpolate(kl1, &gma_gpr[nq * i], nq, ke, gma_smp, nke, DIMKE, pke, npke, NULL,
				0);

		gpr_interpolate(kl2, &gma_gpr[nq * nke + nq * i], nq, ke, gma_smp, nke, DIMKE, pke,
				npke, NULL, 0);
	}

	/* GET PREDICTION FROM SPLIT COVARIANCE */

	nwrk = get_size_split_covar(nke, nq);

	work = malloc(nwrk * sizeof(double));
	assert(work);

	allocate_mem_split_covar(scv, work, nwrk, nke, nq);

	get_ph_split_covar(scv, ke, nke, DIMKE, q_ct, nq, DIMQ, pke, npke, shft);

	get_krn_se_ard(kxx, ke, ke, nke, nke, DIMKE, pke, npke);

	get_covar_chd_noise(lkxx, kxx, NULL, nke);

	get_gma_weight(wt_gma, lkxx, gma_smp, nke);

	interpolate_gma(gma_tst, wt_gma, scv->A1, scv->B1, scv->C, nq, nke);
	interpolate_gma(&gma_tst[nq * nke], wt_gma, scv->A2, scv->B2, scv->C, nq, nke);

	err = 0;
	for (i = 0; i < 2 * nq * nke; i++) {

		err += fabs(gma_tst[i] - gma_gpr[i]);
	}

	free(work);
	free(lkxx);
	free(kxx);
	free(wt_gma);
	free(gma_gpr);
	free(gma_tst);
	free(gma_smp);
	free(q_ct);
	free(q);
	free(kl2);
	free(kl1);
	free(ke);
	free(scv);

	return err;
}
