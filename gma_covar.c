#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include "lib_flow.h"

#define DIMKE (6)
#define DIMQ (3)

void get_zs_loop_mom_ct_ct(double *kl1_ct, double *kl2_ct, const double *ke_ct, unsigned int dimke,
			   const double *q_ct, unsigned long nq, unsigned int dimq)
{
	double dl_ct[3], dlp_ct[3], P_ct[3], dlp_zs1[3], dlp_zs2[3], P_zs1[3], P_zs2[3];
	unsigned long i, l;

	dl_ct[0] = 0;
	dl_ct[1] = 0;
	dl_ct[2] = ke_ct[0];

	dlp_ct[0] = ke_ct[1];
	dlp_ct[1] = ke_ct[2];
	dlp_ct[2] = ke_ct[3];

	P_ct[0] = ke_ct[4];
	P_ct[1] = 0;
	P_ct[2] = ke_ct[5];

	for (l = 0; l < nq; l++) {

		for (i = 0; i < 3; i++) {

			dlp_zs1[i] = 0.5 * (P_ct[i] + dlp_ct[i] - q_ct[l * dimq + i]);

			P_zs1[i] = 0.5 * (P_ct[i] + dlp_ct[i] + q_ct[l * dimq + i]);

			dlp_zs2[i] = 0.5 * (P_ct[i] - dlp_ct[i] - q_ct[l * dimq + i]);

			P_zs2[i] = 0.5 * (P_ct[i] - dlp_ct[i] + q_ct[l * dimq + i]);
		}

		kl1_ct[l * dimke + 0] = dl_ct[2];

		kl1_ct[l * dimke + 1] = dlp_zs1[0];
		kl1_ct[l * dimke + 2] = dlp_zs1[1];
		kl1_ct[l * dimke + 3] = dlp_zs1[2];

		kl1_ct[l * dimke + 4] = P_zs1[0];
		kl1_ct[l * dimke + 5] = P_zs1[2];

		kl2_ct[l * dimke + 0] = dl_ct[2];

		kl2_ct[l * dimke + 1] = dlp_zs2[0];
		kl2_ct[l * dimke + 2] = dlp_zs2[1];
		kl2_ct[l * dimke + 3] = dlp_zs2[2];

		kl2_ct[l * dimke + 4] = P_zs2[0];
		kl2_ct[l * dimke + 5] = P_zs2[2];
	}
}

void get_zs_covar_Aeq(double *A1, double *A2, const double *ke_ct, const double *q_ct, unsigned long nke,
		      unsigned int dimke, unsigned long nq, unsigned int dimq, const double *pke,
		      unsigned long np)
{
	double dlz, dlpx, dlpy, dlpz, Px, Pz, lx2_dlp, ly2_dlp, lz2_dlp, lx2_P, lz2_P, qx, qy, qz, deq1, deq2,
	    k1x, k1y, k1z, k2x, k2y, k2z, Py, X1[DIMKE], X2[DIMKE], Q[DIMKE];
	unsigned long i, j, l;

	for (i = 0; i < nke; i++) {

		dlz = ke_ct[dimke * i + 0];
		dlpx = ke_ct[dimke * i + 1];
		dlpy = ke_ct[dimke * i + 2];
		dlpz = ke_ct[dimke * i + 3];
		Px = ke_ct[dimke * i + 4];
		Py = 0;
		Pz = ke_ct[dimke * i + 5];

		X1[0] = dlz;
		X1[1] = 0.5 * (Px + dlpx);
		X1[2] = 0.5 * (Py + dlpy);
		X1[3] = 0.5 * (Pz + dlpz);
		X1[4] = 0.5 * (Px + dlpx);
		X1[5] = 0.5 * (Pz + dlpz);

		X2[0] = dlz;
		X2[1] = 0.5 * (Px - dlpx);
		X2[2] = 0.5 * (Py - dlpy);
		X2[3] = 0.5 * (Pz - dlpz);
		X2[4] = 0.5 * (Px - dlpx);
		X2[5] = 0.5 * (Pz - dlpz);

		for (j = 0; j < nq; j++) {

			qx = q_ct[dimq * j + 0];
			qy = q_ct[dimq * j + 1];
			qz = q_ct[dimq * j + 2];

			Q[0] = 0;
			Q[1] = -0.5 * qx;
			Q[2] = -0.5 * qy;
			Q[3] = -0.5 * qz;
			Q[4] = 0.5 * qx;
			Q[5] = 0.5 * qz;

			deq1 = 0;
			deq2 = 0;
			for (l = 0; l < dimke; l++) {

				deq1 += (Q[l] * Q[l] + 2 * X1[l] * Q[l]) / (pke[l] * pke[l]);

				deq2 += (Q[l] * Q[l] + 2 * X2[l] * Q[l]) / (pke[l] * pke[l]);
			}

			A1[i * nq + j] = exp(-deq1);
			A2[i * nq + j] = exp(-deq2);
		}
	}
}

void get_zs_covar_Bes(double *B1, double *B2, const double *ke_ct, const double *ks_ct, unsigned long nke,
		      unsigned int dimke, const double *pke, unsigned long np)
{
	double dlpx, dlpy, dlpz, Px, Pz, dlpsx, dlpsy, dlpsz, Psx, Psz, lx2_dlp, ly2_dlp, lz2_dlp, lx2_P,
	    lz2_P, qx, qy, qz, des1, des2, k1x, k1y, k1z, k2x, k2y, k2z, dlz, dlsz, lz2_dl, X1[DIMKE],
	    X2[DIMKE], S[DIMKE], Py;
	unsigned long i, j, l;

	for (i = 0; i < nke; i++) {

		dlz = ke_ct[dimke * i + 0];
		dlpx = ke_ct[dimke * i + 1];
		dlpy = ke_ct[dimke * i + 2];
		dlpz = ke_ct[dimke * i + 3];
		Px = ke_ct[dimke * i + 4];
		Py = 0;
		Pz = ke_ct[dimke * i + 5];

		X1[0] = dlz;
		X1[1] = 0.5 * (Px + dlpx);
		X1[2] = 0.5 * (Py + dlpy);
		X1[3] = 0.5 * (Pz + dlpz);
		X1[4] = 0.5 * (Px + dlpx);
		X1[5] = 0.5 * (Pz + dlpz);

		X2[0] = dlz;
		X2[1] = 0.5 * (Px - dlpx);
		X2[2] = 0.5 * (Py - dlpy);
		X2[3] = 0.5 * (Pz - dlpz);
		X2[4] = 0.5 * (Px - dlpx);
		X2[5] = 0.5 * (Pz - dlpz);

		for (j = 0; j < nke; j++) {

			S[0] = ks_ct[dimke * j + 0];
			S[1] = ks_ct[dimke * j + 1];
			S[2] = ks_ct[dimke * j + 2];
			S[3] = ks_ct[dimke * j + 3];
			S[4] = ks_ct[dimke * j + 4];
			S[5] = ks_ct[dimke * j + 5];

			des1 = 0;
			des2 = 0;
			for (l = 0; l < dimke; l++) {

				des1 += (X1[l] - S[l]) * (X1[l] - S[l]) / (pke[l] * pke[l]);

				des2 += (X2[l] - S[l]) * (X2[l] - S[l]) / (pke[l] * pke[l]);
			}

			B1[i * nke + j] = exp(-des1);
			B2[i * nke + j] = exp(-des2);
		}
	}
}

void get_zs_covar_Cqs(double *C, const double *ke_ct, const double *q_ct, unsigned long nke,
		      unsigned int dimke, unsigned long nq, unsigned int dimq, const double *pke,
		      unsigned long np)
{
	double dlpsx, dlpsy, dlpsz, Psx, Psz, lx2_dlp, ly2_dlp, lz2_dlp, lx2_P, lz2_P, qx, qy, qz, dqs,
	    Q[DIMKE], S[DIMKE];

	unsigned long i, j, l;

	for (i = 0; i < nq; i++) {

		qx = q_ct[dimq * i + 0];
		qy = q_ct[dimq * i + 1];
		qz = q_ct[dimq * i + 2];

		Q[0] = 0;
		Q[1] = -0.5 * qx;
		Q[2] = -0.5 * qy;
		Q[3] = -0.5 * qz;
		Q[4] = 0.5 * qx;
		Q[5] = 0.5 * qz;

		for (j = 0; j < nke; j++) {

			S[0] = ke_ct[dimke * j + 0];
			S[1] = ke_ct[dimke * j + 1];
			S[2] = ke_ct[dimke * j + 2];
			S[3] = ke_ct[dimke * j + 3];
			S[4] = ke_ct[dimke * j + 4];
			S[5] = ke_ct[dimke * j + 5];

			dqs = 0;
			for (l = 0; l < dimke; l++) {

				dqs += (-2 * Q[l] * S[l]) / (pke[l] * pke[l]);
			}

			C[i * nke + j] = exp(-dqs);
		}
	}
}

double test_zs_gma_covar(unsigned long nke, unsigned long nq, int seed)
{
	double *kl1, *kl2, *kl1_ct, *kl2_ct, *q_ct, *ke, *ke_ct, *q, *A1, *A2, *B1, *B2, *C, kmax, st[3],
	    en[3], pke[DIMKE + 1], *kls1, *kls2, *kls1_spl, *kls2_spl, err, Px, Py, Pz, dlpx, dlpy, dlpz, dlz,
	    qx, qy, qz, Psx, Psy, Psz, dlpsx, dlpsy, dlpsz, dlsz, k1x, k1y, k1z;
	unsigned long np, i, j, s, e, l;
	unsigned int dimke, dimq;
	dsfmt_t drng;

	dimke = DIMKE;
	dimq = DIMQ;
	np = DIMKE + 1;

	ke = malloc(dimke * nke * sizeof(double));
	assert(ke);
	q = malloc(dimq * nq * sizeof(double));
	assert(q);

	kl1 = malloc(dimke * nq * sizeof(double));
	assert(kl1);
	kl2 = malloc(dimke * nq * sizeof(double));
	assert(kl2);

	ke_ct = malloc(dimke * nke * sizeof(double));
	assert(ke_ct);

	kl1_ct = malloc(dimke * nq * sizeof(double));
	assert(kl1);
	kl2_ct = malloc(dimke * nq * sizeof(double));
	assert(kl2);

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

	fill_ext_momenta_3ball(ke, nke, st, en, seed);

	fill_ext_momenta_ball(q, nq, st[0], en[0], seed + 443);

	sph_ct_mom_ball(q_ct, q, dimq, nq);
	sph_ct_mom6_zs(ke_ct, ke, dimke, nke);

	dsfmt_init_gen_rand(&drng, seed + 95);

	for (i = 0; i < np; i++) {
		pke[i] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);
	}

	for (e = 0; e < nke; e++) {
		get_zs_loop_mom_ct_ct(kl1_ct, kl2_ct, &ke_ct[dimke * e], dimke, q_ct, nq, dimq);

		get_krn_se_ard(&kls1[e * nq * nke], kl1_ct, ke_ct, nq, nke, dimke, pke, np);

		get_krn_se_ard(&kls2[e * nq * nke], kl2_ct, ke_ct, nq, nke, dimke, pke, np);
	}

	get_zs_covar_Aeq(A1, A2, ke_ct, q_ct, nke, dimke, nq, dimq, pke, np);
	get_zs_covar_Bes(B1, B2, ke_ct, ke_ct, nke, dimke, pke, np);
	get_zs_covar_Cqs(C, ke_ct, q_ct, nke, dimke, nq, dimq, pke, np);

	for (e = 0; e < nke; e++) {
		for (l = 0; l < nq; l++) {
			for (s = 0; s < nke; s++) {

				kls1_spl[e * nke * nq + l * nke + s] = pke[DIMKE] * pke[DIMKE]
								       * A1[e * nq + l] * B1[e * nke + s]
								       * C[l * nke + s];
				kls2_spl[e * nke * nq + l * nke + s] = pke[DIMKE] * pke[DIMKE]
								       * A2[e * nq + l] * B2[e * nke + s]
								       * C[l * nke + s];
			}
		}
	}

	err = 0;

	for (i = 0; i < nke * nke * nq; i++) {
		err += fabs(kls1[i] - kls1_spl[i]);
		err += fabs(kls2[i] - kls2_spl[i]);
	}

	return err;
}
