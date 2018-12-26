#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include "lib_flow.h"

#define PI (3.1415926535897)
#define PREFAC (1 / (8 * PI * PI * PI))

extern double ddot_(const int *N, const double *X, const int *incx, const double *Y, const int *incy);

void get_I2q(double *I2q, const double *xq, unsigned long nq, unsigned int dim, double *q0, double *q1,
	     unsigned long nth, double lq)
{
	double lq2, sqrt_pi, qi, exp_q1, exp_q0, Erf_q1, Erf_q0, qi2;
	unsigned long i, j;

	lq2 = lq * lq;
	sqrt_pi = sqrt(PI);

	for (i = 0; i < nq; i++) {

		qi = xq[dim * i + 0];
		qi2 = qi * qi;

		for (j = 0; j < nth; j++) {

			exp_q1 = exp(-(q1[j] - qi) * (q1[j] - qi) / lq2);
			exp_q0 = exp(-(q0[j] - qi) * (q0[j] - qi) / lq2);

			Erf_q1 = Erf((q1[j] - qi) / lq);
			Erf_q0 = Erf((q0[j] - qi) / lq);

			I2q[i * nth + j] = -0.5 * lq2 * ((qi + q1[j]) * exp_q1 - (qi + q0[j]) * exp_q0)
					   + 0.25 * sqrt_pi * lq * (lq2 + 2 * qi2) * (Erf_q1 - Erf_q0);
		}
	}
}

void get_I3q(double *I3q, const double *xq, unsigned long nq, unsigned int dim, double *q0, double *q1,
	     unsigned long nth, double lq)
{
	double lq2, sqrt_pi, qi, exp_q1, exp_q0, Erf_q1, Erf_q0, qi2, q02, q12, tq0, tq1;
	unsigned long i, j;

	lq2 = lq * lq;
	sqrt_pi = sqrt(PI);

	for (i = 0; i < nq; i++) {

		qi = xq[dim * i + 0];
		qi2 = qi * qi;

		for (j = 0; j < nth; j++) {

			q02 = q0[j] * q0[j];
			q12 = q1[j] * q1[j];

			exp_q1 = exp(-(q1[j] - qi) * (q1[j] - qi) / lq2);
			exp_q0 = exp(-(q0[j] - qi) * (q0[j] - qi) / lq2);

			Erf_q1 = Erf((q1[j] - qi) / lq);
			Erf_q0 = Erf((q0[j] - qi) / lq);

			tq0 = lq2 + q02 + q0[j] * qi + qi2;
			tq1 = lq2 + q12 + q1[j] * qi + qi2;

			I3q[i * nth + j]
			    = -0.5 * lq2 * (tq1 * exp_q1 - tq0 * exp_q0)
			      + 0.25 * sqrt_pi * lq * qi * (3 * lq2 + 2 * qi2) * (Erf_q1 - Erf_q0);
		}
	}
}

void get_zs_Ifq(double *Ifq, const double *xq, unsigned long nq, const double *l, unsigned int dimq,
		const double *ke, unsigned long nke, unsigned int dimke, unsigned long nth, double fac,
		double kf)
{
	double *I_phi, *I2q, *I3q, sqrt_pi, phi_qi, lq, lth, lphi, *gth, *gwth, *th, *wth, exp_th_kj, *q0,
	    *q1, e_ext, dl, diff_th_kj, tmp, thj, I3_jk, I2_jk;
	unsigned long i, j, k, nth1;

	assert(nth % 4 == 0);
	nth1 = nth / 4;

	I_phi = malloc(nq * sizeof(double));
	assert(I_phi);
	I2q = malloc(nq * nth * sizeof(double));
	assert(I2q);
	I3q = malloc(nq * nth * sizeof(double));
	assert(I3q);
	gth = malloc(nth1 * sizeof(double));
	assert(gth);
	gwth = malloc(nth1 * sizeof(double));
	assert(gwth);
	th = malloc(nth * sizeof(double));
	assert(th);
	wth = malloc(nth * sizeof(double));
	assert(wth);
	q0 = malloc(nth * sizeof(double));
	assert(q0);
	q1 = malloc(nth * sizeof(double));
	assert(q1);

	gauss_grid_create(nth1, gth, gwth, -1, 1);

	sqrt_pi = sqrt(PI);

	lq = l[0];
	lth = l[1];
	lphi = l[2];

	for (i = 0; i < nq; i++) {
		phi_qi = xq[dimq * i + 2];
		I_phi[i] = 0.5 * sqrt_pi * lphi * (Erf(2.0 * PI - phi_qi) - Erf(0 - phi_qi));
	}

	for (i = 0; i < nke; i++) {
		e_ext = get_zs_energy(&ke[dimke * i], dimke);
		dl = ke[dimke * i + 0];

		get_zs_th_grid(th, wth, q0, q1, nth, gth, gwth, dl, kf, fac);

		get_I2q(I2q, xq, nq, dimq, q0, q1, nth, lq);
		get_I3q(I3q, xq, nq, dimq, q0, q1, nth, lq);

		for (j = 0; j < nq; j++) {

			thj = xq[dimq * j + 0];

			tmp = 0;
			for (k = 0; k < nth; k++) {

				diff_th_kj = (th[k] - thj) / lth;
				exp_th_kj = exp(-diff_th_kj * diff_th_kj);

				I3_jk = I3q[j * nth + k];
				I2_jk = I2q[j * nth + k];

				tmp += wth[k] * sin(th[k]) * exp_th_kj
				       * (-4 * cos(th[k]) * I3_jk + e_ext * I2_jk);
			}

			Ifq[i * nq + j] = PREFAC * I_phi[j] * tmp;
		}
	}

	free(I_phi);
	free(I2q);
	free(I3q);
	free(q0);
	free(q1);
	free(th);
	free(wth);
	free(gth);
	free(gwth);
}

void predict_zs_fq(double *zs, unsigned long nke, const double *wq, unsigned long nq, const double *Ifq)
{
	int N, INCX, INCY;
	unsigned long i;

	N = nq;
	INCX = 1;
	INCY = 1;

	for (i = 0; i < nke; i++) {

		zs[i] = ddot_(&N, &wq[i * nq], &INCX, &Ifq[i * nq], &INCY);
	}
}

double test_get_I2q(unsigned int tn, double q0, double q1, double lq)
{
	double *q, *I2q_exct, *I2q, *qg, *wt, err_norm, diff, qmin[1], qmax[1];
	unsigned long i, j, ng, nth;

	q = malloc(tn * sizeof(double));
	assert(q);
	I2q_exct = malloc(tn * sizeof(double));
	assert(I2q_exct);
	I2q = malloc(tn * sizeof(double));
	assert(I2q);

	for (i = 0; i < tn; i++) {
		q[i] = 3.0 * rand() / (1.0 + RAND_MAX);
	}

	nth = 1;
	qmin[0] = q0;
	qmax[0] = q1;

	get_I2q(I2q, q, tn, 1, qmin, qmax, nth, lq);

	ng = 20;
	qg = malloc(ng * sizeof(double));
	assert(qg);
	wt = malloc(ng * sizeof(double));
	assert(wt);

	gauss_grid_create(ng, qg, wt, q0, q1);

	for (i = 0; i < tn; i++) {

		I2q_exct[i] = 0;
		for (j = 0; j < ng; j++) {
			I2q_exct[i]
			    += wt[j] * qg[j] * qg[j] * exp(-(qg[j] - q[i]) * (qg[j] - q[i]) / (lq * lq));
		}
	}

	err_norm = 0;
	for (i = 0; i < tn; i++) {
		diff = I2q_exct[i] - I2q[i];
		err_norm += diff * diff;
	}

	err_norm = sqrt(err_norm);

	free(q);
	free(I2q);
	free(I2q_exct);
	free(qg);
	free(wt);

	return err_norm;
}

double test_get_I3q(unsigned int tn, double q0, double q1, double lq)
{
	double *q, *I3q_exct, *I3q, *qg, *wt, err_norm, diff, qmin[1], qmax[1];
	unsigned long i, j, ng, nth;

	q = malloc(tn * sizeof(double));
	assert(q);
	I3q_exct = malloc(tn * sizeof(double));
	assert(I3q_exct);
	I3q = malloc(tn * sizeof(double));
	assert(I3q);

	for (i = 0; i < tn; i++) {
		q[i] = 3.0 * rand() / (1.0 + RAND_MAX);
	}

	nth = 1;
	qmin[0] = q0;
	qmax[0] = q1;

	get_I3q(I3q, q, tn, 1, qmin, qmax, nth, lq);

	ng = 20;
	qg = malloc(ng * sizeof(double));
	assert(qg);
	wt = malloc(ng * sizeof(double));
	assert(wt);

	gauss_grid_create(ng, qg, wt, q0, q1);

	for (i = 0; i < tn; i++) {

		I3q_exct[i] = 0;
		for (j = 0; j < ng; j++) {
			I3q_exct[i] += wt[j] * qg[j] * qg[j] * qg[j]
				       * exp(-(qg[j] - q[i]) * (qg[j] - q[i]) / (lq * lq));
		}
	}

	err_norm = 0;
	for (i = 0; i < tn; i++) {
		diff = I3q_exct[i] - I3q[i];
		err_norm += diff * diff;
	}

	err_norm = sqrt(err_norm);

	free(q);
	free(I3q);
	free(I3q_exct);
	free(qg);
	free(wt);

	return err_norm;
}