#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include "lib_flow.h"

#define PI (3.1415926535897)
#define PREFAC (1 / (8 * PI * PI * PI))
#define DIMKE (6)

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

			thj = xq[dimq * j + 1];

			tmp = 0;
			for (k = 0; k < nth; k++) {

				diff_th_kj = (th[k] - thj) / lth;
				exp_th_kj = exp(-diff_th_kj * diff_th_kj);

				I3_jk = I3q[j * nth + k];
				I2_jk = I2q[j * nth + k];

				tmp += wth[k] * sin(th[k]) * exp_th_kj
				       * (-4 * dl * cos(th[k]) * I3_jk + e_ext * I2_jk);
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

void get_zs_num(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
		unsigned long nth, unsigned long nphi, double (*vfun)(double *, unsigned int, double *),
		double *param)
{
	double *xq, *wxq, ph_vol, q, th, phi, e_ext, kl1[DIMKE], kl2[DIMKE], v1, v2, tmp, eq, P, P_dl, dlp,
	    dl_dlp, dl, P_dlp, phi_dlp;
	unsigned long nxq, i, n;
	unsigned int dimq;

	nxq = nq * nth * nphi;
	dimq = 3;

	xq = malloc(dimq * nxq * sizeof(double));
	assert(xq);
	wxq = malloc(nxq * sizeof(double));
	assert(wxq);

	for (n = 0; n < ns; n++) {

		dl = ext_mom[dim * n + 0];
		P = ext_mom[dim * n + 2];
		dl_dlp = ext_mom[dim * n + 3];
		P_dl = ext_mom[dim * n + 4];
		P_dlp = ext_mom[dim * n + 5];

		get_ph_space_grid(xq, wxq, dimq, dl, kf, nq, nth, nphi);

		phi_dlp = acos((cos(P_dlp) - cos(P_dl) * cos(dl_dlp)) / (sin(P_dl) * sin(dl_dlp)));

		e_ext = get_zs_energy(&ext_mom[dim * n], dim);

		tmp = 0;
		for (i = 0; i < nxq; i++) {

			q = xq[dimq * i + 0];
			th = xq[dimq * i + 1];
			phi = xq[dimq * i + 2];

			eq = -4 * q * dl * cos(th) + e_ext;
			get_zs_loop_mom_ct(kl1, kl2, dim, &ext_mom[dim * n], phi_dlp, q, th, phi);

			v1 = (*vfun)(kl1, dim, param);
			v2 = (*vfun)(kl2, dim, param);

			tmp += wxq[i] * q * q * sin(th) * eq * v1 * v2;
		}

		zs[n] = PREFAC * tmp;
	}

	free(xq);
	free(wxq);
}

void get_zs_Ifq_num(double *Ifq_num, double *ke, unsigned long nke, unsigned int dimke, double kf,
		    unsigned long nq, unsigned long nth, unsigned long nphi, double *xqi, unsigned long nxqi,
		    unsigned int dimq, double *pq, double fac)
{
	double *xq, *wxq, dl, qexp, q, th_q, phi_q, qi, th_qi, phi_qi, tmp, eq, e_ext;
	unsigned int nxq, i, j, k;

	nxq = nth * nq * nphi;

	xq = malloc(dimq * nxq * sizeof(double));
	assert(xq);
	wxq = malloc(nxq * sizeof(double));
	assert(wxq);

	for (i = 0; i < nke; i++) {

		dl = ke[dimke * i + 0];

		get_ph_space_grid(xq, wxq, dimq, dl, kf, nq, nth, nphi);

		e_ext = get_zs_energy(&ke[dimke * i], dimke);

		for (j = 0; j < nxqi; j++) {

			qi = xqi[dimq * j + 0];
			th_qi = xqi[dimq * j + 1];
			phi_qi = xqi[dimq * j + 2];

			tmp = 0;
			for (k = 0; k < nxq; k++) {

				q = xq[dimq * k + 0];
				th_q = xq[dimq * k + 1];
				phi_q = xq[dimq * k + 2];

				eq = -4 * dl * q * cos(th_q) + e_ext;

				qexp = exp(-(qi - q) * (qi - q) / pq[0])
				       * exp(-(th_qi - th_q) * (th_qi - th_q) / pq[1])
				       * exp(-(phi_qi - phi_q) * (phi_qi - phi_q) / pq[2]);

				tmp += q * q * sin(th_q) * wxq[k] * eq * qexp;
			}

			Ifq_num[i * nxqi + j] = PREFAC * tmp;
		}
	}

	free(xq);
	free(wxq);
}
