#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include "split_covar.h"
#include "momentum_sample.h"
#include "loop_vtx.h"

#define DIMKE (7)
#define DIMQ (3)

double test_zs_split_covar(unsigned long nke, unsigned long nq, int shft, int seed)
{
	double *kl1, *kl2, *kl1_ct, *kl2_ct, *q_ct, *ke, *ke_ct, *q, *A1, *A2, *B1, *B2, *C, kmax,
	    st[3], en[3], pke[DIMKE + 1], *kls1, *kls2, *kls1_spl, *kls2_spl, err, Px, Py, Pz, dlpx,
	    dlpy, dlpz, dlz, qx, qy, qz, Psx, Psy, Psz, dlpsx, dlpsy, dlpsz, dlsz, k1x, k1y, k1z,
	    *work;
	unsigned long np, i, j, s, e, l, nwrk;
	unsigned int dimke, dimq;
	struct split_covar *scv;
	dsfmt_t drng;

	printf("test_zs_split_covar(nke = %lu, nq = %lu, shft = %d):\n", nke, nq, shft);

	scv = malloc(sizeof(struct split_covar));
	assert(scv);

	dimke = DIMKE;
	dimq = DIMQ;
	np = DIMKE + 1;

	q = malloc(dimq * nq * sizeof(double));
	assert(q);

	ke_ct = malloc(dimke * nke * sizeof(double));
	assert(ke_ct);

	kl1_ct = malloc(dimke * nq * sizeof(double));
	assert(kl1_ct);
	kl2_ct = malloc(dimke * nq * sizeof(double));
	assert(kl2_ct);

	q_ct = malloc(dimke * nq * sizeof(double));
	assert(q_ct);

	kls1 = malloc(nke * nke * nq * sizeof(double));
	assert(kls1);
	kls2 = malloc(nke * nke * nq * sizeof(double));
	assert(kls2);
	kls1_spl = malloc(nke * nke * nq * sizeof(double));
	assert(kls1_spl);
	kls2_spl = malloc(nke * nke * nq * sizeof(double));
	assert(kls2_spl);

	nwrk = get_size_split_covar(nke, nq);

	work = malloc(nwrk * sizeof(double));
	assert(work);

	allocate_mem_split_covar(scv, work, nwrk, nke, nq);

	kmax = 2.0;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ph_sample_ct_box(ke_ct, nke, st, en, seed);

	fill_q_sample_ball(q, nq, st[0], en[0], seed + 443);

	sph_to_ct(q_ct, q, dimq, nq);

	dsfmt_init_gen_rand(&drng, seed + 95);

	for (i = 0; i < np; i++) {
		pke[i] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);
	}

	for (e = 0; e < nke; e++) {
		get_ph_loop_mom(kl1_ct, kl2_ct, &ke_ct[dimke * e], dimke, q_ct, nq, dimq, shft);

		get_krn_se_ard(&kls1[e * nq * nke], kl1_ct, ke_ct, nq, nke, dimke, pke, np);

		get_krn_se_ard(&kls2[e * nq * nke], kl2_ct, ke_ct, nq, nke, dimke, pke, np);
	}

	get_ph_split_covar(scv, ke_ct, nke, dimke, q_ct, nq, dimq, pke, np, shft);

	for (e = 0; e < nke; e++) {
		for (l = 0; l < nq; l++) {
			for (s = 0; s < nke; s++) {

				kls1_spl[e * nke * nq + l * nke + s] = scv->A1[e * nq + l]
								       * scv->B1[e * nke + s]
								       * scv->C[s * nq + l];

				kls2_spl[e * nke * nq + l * nke + s] = scv->A2[e * nq + l]
								       * scv->B2[e * nke + s]
								       * scv->C[s * nq + l];
			}
		}
	}

	err = 0;

	for (i = 0; i < nke * nke * nq; i++) {
		err += fabs(kls1[i] - kls1_spl[i]);
		err += fabs(kls2[i] - kls2_spl[i]);
	}

	free(work);
	free(kls2_spl);
	free(kls1_spl);
	free(kls2);
	free(kls1);
	free(q_ct);
	free(ke_ct);
	free(kl2_ct);
	free(kl1_ct);
	free(q);
	free(scv);

	return err;
}

double test_zsp_split_covar(unsigned long nke, unsigned long nq, int shft, int seed)
{
	double *kl1, *kl2, *kl1_ct, *kl2_ct, *q_ct, *ke, *ke_ct, *kep_ct, *q, *A1, *A2, *B1, *B2,
	    *C, kmax, st[3], en[3], pke[DIMKE + 1], *kls1, *kls2, *kls1_spl, *kls2_spl, err, Px, Py,
	    Pz, dlpx, dlpy, dlpz, dlz, qx, qy, qz, Psx, Psy, Psz, dlpsx, dlpsy, dlpsz, dlsz, k1x,
	    k1y, k1z;
	unsigned long np, i, j, s, e, l;
	unsigned int dimke, dimq;
	struct split_covar *scv;
	dsfmt_t drng;

	printf("test_zsp_split_covar(nke = %lu, nq = %lu, shft = %d):\n", nke, nq, shft);

	scv = malloc(sizeof(struct split_covar));
	assert(scv);

	dimke = DIMKE;
	dimq = DIMQ;
	np = DIMKE + 1;

	q = malloc(dimq * nq * sizeof(double));
	assert(q);

	ke_ct = malloc(dimke * nke * sizeof(double));
	assert(ke_ct);
	kep_ct = malloc(dimke * nke * sizeof(double));
	assert(kep_ct);

	kl1_ct = malloc(dimke * nq * sizeof(double));
	assert(kl1_ct);
	kl2_ct = malloc(dimke * nq * sizeof(double));
	assert(kl2_ct);

	q_ct = malloc(dimke * nq * sizeof(double));
	assert(q_ct);

	kls1 = malloc(nke * nke * nq * sizeof(double));
	assert(kls1);
	kls2 = malloc(nke * nke * nq * sizeof(double));
	assert(kls2);
	kls1_spl = malloc(nke * nke * nq * sizeof(double));
	assert(kls1_spl);
	kls2_spl = malloc(nke * nke * nq * sizeof(double));
	assert(kls2_spl);

	scv->A1 = malloc(nke * nq * sizeof(double));
	assert(scv->A1);
	scv->A2 = malloc(nke * nq * sizeof(double));
	assert(scv->A2);
	scv->B1 = malloc(nke * nke * sizeof(double));
	assert(scv->B1);
	scv->B2 = malloc(nke * nke * sizeof(double));
	assert(scv->B2);
	scv->C = malloc(nke * nq * sizeof(double));
	assert(scv->C);

	kmax = 2.0;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ph_sample_ct_box(ke_ct, nke, st, en, seed);

	get_ph_ex_sample(kep_ct, ke_ct, nke, dimke);

	fill_q_sample_ball(q, nq, st[0], en[0], seed + 443);

	sph_to_ct(q_ct, q, dimq, nq);

	dsfmt_init_gen_rand(&drng, seed + 95);

	for (i = 0; i < np; i++) {
		pke[i] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);
	}

	for (e = 0; e < nke; e++) {
		get_ph_loop_mom(kl1_ct, kl2_ct, &kep_ct[dimke * e], dimke, q_ct, nq, dimq, shft);

		get_krn_se_ard(&kls1[e * nq * nke], kl1_ct, kep_ct, nq, nke, dimke, pke, np);

		get_krn_se_ard(&kls2[e * nq * nke], kl2_ct, kep_ct, nq, nke, dimke, pke, np);
	}

	get_ph_split_covar(scv, kep_ct, nke, dimke, q_ct, nq, dimq, pke, np, shft);

	for (e = 0; e < nke; e++) {
		for (l = 0; l < nq; l++) {
			for (s = 0; s < nke; s++) {

				kls1_spl[e * nke * nq + l * nke + s] = scv->A1[e * nq + l]
								       * scv->B1[e * nke + s]
								       * scv->C[s * nq + l];

				kls2_spl[e * nke * nq + l * nke + s] = scv->A2[e * nq + l]
								       * scv->B2[e * nke + s]
								       * scv->C[s * nq + l];
			}
		}
	}

	err = 0;

	for (i = 0; i < nke * nke * nq; i++) {
		err += fabs(kls1[i] - kls1_spl[i]);
		err += fabs(kls2[i] - kls2_spl[i]);
	}

	free(scv->C);
	free(scv->B2);
	free(scv->B1);
	free(scv->A2);
	free(scv->A1);
	free(kls2_spl);
	free(kls1_spl);
	free(kls2);
	free(kls1);
	free(q_ct);
	free(kep_ct);
	free(ke_ct);
	free(kl2_ct);
	free(kl1_ct);
	free(q);
	free(scv);

	return err;
}
