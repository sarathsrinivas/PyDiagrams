#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "split_covar.h"

#define DIMKE (7)
#define DIMQ (3)
#define CHUNK (100)

static void get_ph_covar_A(double *A1, double *A2, const double *ke_ct, unsigned long nke,
			   unsigned int dimke, const double *q_ct, unsigned long nq,
			   unsigned int dimq, const double *pke, unsigned long np, int shft)
{
	double dlz, dlpx, dlpy, dlpz, Px, Pz, lx2_dlp, ly2_dlp, lz2_dlp, lx2_P, lz2_P, qx, qy, qz,
	    deq1, deq2, k1x, k1y, k1z, k2x, k2y, k2z, Py, X1[DIMKE], X2[DIMKE], Q[DIMKE];
	unsigned long i, j, l;

#pragma omp parallel
	{
#pragma omp parallel for default(none)                                                             \
    shared(A1, A2, ke_ct, q_ct, pke, dimke, dimq, nq, nke, shft) private(                          \
	i, j, l, dlz, dlpx, dlpy, dlpz, Px, Py, Pz, X1, X2, qx, qy, qz, Q, deq1, deq2)             \
	schedule(dynamic, CHUNK)
		for (i = 0; i < nke; i++) {

			dlz = ke_ct[dimke * i + 0];
			dlpx = ke_ct[dimke * i + 1];
			dlpy = ke_ct[dimke * i + 2];
			dlpz = ke_ct[dimke * i + 3];
			Px = ke_ct[dimke * i + 4];
			Py = ke_ct[dimke * i + 5];
			Pz = ke_ct[dimke * i + 6];

			X1[0] = dlz;
			X1[1] = 0.5 * (Px + dlpx);
			X1[2] = 0.5 * (Py + dlpy);
			X1[3] = 0.5 * (Pz + dlpz - shft * dlz);
			X1[4] = 0.5 * (Px + dlpx);
			X1[5] = 0.5 * (Py + dlpy);
			X1[6] = 0.5 * (Pz + dlpz + shft * dlz);

			X2[0] = dlz;
			X2[1] = 0.5 * (Px - dlpx);
			X2[2] = 0.5 * (Py - dlpy);
			X2[3] = 0.5 * (Pz - dlpz - shft * dlz);
			X2[4] = 0.5 * (Px - dlpx);
			X2[5] = 0.5 * (Py - dlpy);
			X2[6] = 0.5 * (Pz - dlpz + shft * dlz);

			for (j = 0; j < nq; j++) {

				qx = q_ct[dimq * j + 0];
				qy = q_ct[dimq * j + 1];
				qz = q_ct[dimq * j + 2];

				Q[0] = 0;
				Q[1] = -0.5 * qx;
				Q[2] = -0.5 * qy;
				Q[3] = -0.5 * qz;
				Q[4] = 0.5 * qx;
				Q[5] = 0.5 * qy;
				Q[6] = 0.5 * qz;

				deq1 = 0;
				deq2 = 0;
				for (l = 0; l < dimke; l++) {

					deq1
					    += (Q[l] * Q[l] + 2 * X1[l] * Q[l]) / (pke[l] * pke[l]);

					deq2
					    += (Q[l] * Q[l] + 2 * X2[l] * Q[l]) / (pke[l] * pke[l]);
				}

				A1[i * nq + j] = exp(-deq1);
				A2[i * nq + j] = exp(-deq2);
			}
		}
	}
}

static void get_ph_covar_B(double *B1, double *B2, const double *ke_ct, const double *ks_ct,
			   unsigned long nke, unsigned int dimke, const double *pke,
			   unsigned long np, int shft)
{
	double dlpx, dlpy, dlpz, Px, Pz, dlpsx, dlpsy, dlpsz, Psx, Psz, lx2_dlp, ly2_dlp, lz2_dlp,
	    lx2_P, lz2_P, qx, qy, qz, des1, des2, k1x, k1y, k1z, k2x, k2y, k2z, dlz, dlsz, lz2_dl,
	    X1[DIMKE], X2[DIMKE], S[DIMKE], Py;
	unsigned long i, j, l;

#pragma parallel
	{
#pragma omp parallel for default(none)                                                             \
    shared(B1, B2, ke_ct, ks_ct, pke, dimke, nke, shft) private(i, j, l, dlz, dlpx, dlpy, dlpz,    \
								Px, Py, Pz, X1, X2, S, des1, des2) \
	schedule(dynamic, CHUNK)
		for (i = 0; i < nke; i++) {

			dlz = ke_ct[dimke * i + 0];
			dlpx = ke_ct[dimke * i + 1];
			dlpy = ke_ct[dimke * i + 2];
			dlpz = ke_ct[dimke * i + 3];
			Px = ke_ct[dimke * i + 4];
			Py = ke_ct[dimke * i + 5];
			Pz = ke_ct[dimke * i + 6];

			X1[0] = dlz;
			X1[1] = 0.5 * (Px + dlpx);
			X1[2] = 0.5 * (Py + dlpy);
			X1[3] = 0.5 * (Pz + dlpz - shft * dlz);
			X1[4] = 0.5 * (Px + dlpx);
			X1[5] = 0.5 * (Py + dlpy);
			X1[6] = 0.5 * (Pz + dlpz + shft * dlz);

			X2[0] = dlz;
			X2[1] = 0.5 * (Px - dlpx);
			X2[2] = 0.5 * (Py - dlpy);
			X2[3] = 0.5 * (Pz - dlpz - shft * dlz);
			X2[4] = 0.5 * (Px - dlpx);
			X2[5] = 0.5 * (Py - dlpy);
			X2[6] = 0.5 * (Pz - dlpz + shft * dlz);

			for (j = 0; j < nke; j++) {

				S[0] = ks_ct[dimke * j + 0];
				S[1] = ks_ct[dimke * j + 1];
				S[2] = ks_ct[dimke * j + 2];
				S[3] = ks_ct[dimke * j + 3];
				S[4] = ks_ct[dimke * j + 4];
				S[5] = ks_ct[dimke * j + 5];
				S[6] = ks_ct[dimke * j + 6];

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
}

static void get_ph_covar_C(double *C, const double *ke_ct, unsigned long nke, unsigned int dimke,
			   const double *q_ct, unsigned long nq, unsigned int dimq,
			   const double *pke, unsigned long np)
{
	double dlpsx, dlpsy, dlpsz, Psy, Psx, Psz, lx2_dlp, ly2_dlp, lz2_dlp, lx2_P, lz2_P, qx, qy,
	    qz, dqs, Q[DIMKE], S[DIMKE], sigma2;

	unsigned long i, j, l;

	sigma2 = pke[dimke] * pke[dimke];

#pragma parallel
	{
#pragma omp parallel for default(none) shared(C, ke_ct, q_ct, pke, dimke, dimq, nq, nke,           \
					      sigma2) private(i, j, l, qx, qy, qz, S, Q, dqs)      \
    schedule(dynamic, CHUNK)
		for (j = 0; j < nke; j++) {

			S[0] = ke_ct[dimke * j + 0];
			S[1] = ke_ct[dimke * j + 1];
			S[2] = ke_ct[dimke * j + 2];
			S[3] = ke_ct[dimke * j + 3];
			S[4] = ke_ct[dimke * j + 4];
			S[5] = ke_ct[dimke * j + 5];
			S[6] = ke_ct[dimke * j + 6];

			for (i = 0; i < nq; i++) {

				qx = q_ct[dimq * i + 0];
				qy = q_ct[dimq * i + 1];
				qz = q_ct[dimq * i + 2];

				Q[0] = 0;
				Q[1] = -0.5 * qx;
				Q[2] = -0.5 * qy;
				Q[3] = -0.5 * qz;
				Q[4] = 0.5 * qx;
				Q[5] = 0.5 * qy;
				Q[6] = 0.5 * qz;

				dqs = 0;
				for (l = 0; l < dimke; l++) {

					dqs += (-2 * Q[l] * S[l]) / (pke[l] * pke[l]);
				}

				C[j * nq + i] = sigma2 * exp(-dqs);
			}
		}
	}
}

void get_ph_split_covar(struct split_covar *scv, const double *ke_ct, unsigned long nke,
			unsigned int dimke, const double *q_ct, unsigned long nq, unsigned int dimq,
			const double *pke, unsigned long npke, int shft)
{
	get_ph_covar_A(scv->A1, scv->A2, ke_ct, nke, dimke, q_ct, nq, dimq, pke, npke, shft);
	get_ph_covar_B(scv->B1, scv->B2, ke_ct, ke_ct, nke, dimke, pke, npke, shft);
	get_ph_covar_C(scv->C, ke_ct, nke, dimke, q_ct, nq, dimq, pke, npke);
}
