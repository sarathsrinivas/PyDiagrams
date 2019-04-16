#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include <atlas/blas.h>
#include <atlas/lapack.h>
#include <lib_gpr/lib_gpr.h>
#include <lib_rng/lib_rng.h>
#include <lib_pots/lib_pots.h>
#include "lib_flow.h"

#define PI (3.1415926535897)
/*
#define PREFAC (1)
*/
#define PREFAC (1 / (8 * PI * PI * PI))
#define DIMKE (7)
#define DIMQ (3)

void get_zs_fq_mat_fun(double *fqke, const double *ke, unsigned long nke, unsigned int dimke,
		       const double *qi, unsigned long nqi, unsigned int dimq,
		       double v_fun(double *, unsigned int, double *), double *vpar)
{
	double *kl1, *kl2, v1, v2;
	unsigned long i, n;

	kl1 = malloc(dimke * nqi * sizeof(double));
	assert(kl1);
	kl2 = malloc(dimke * nqi * sizeof(double));
	assert(kl2);

	for (n = 0; n < nke; n++) {

		get_zs_loop_mom_7d_ct(kl1, kl2, &ke[dimke * n], dimke, qi, nqi, dimq);

		for (i = 0; i < nqi; i++) {

			v1 = v_fun(&kl1[dimke * i], dimke, vpar);
			v2 = v_fun(&kl2[dimke * i], dimke, vpar);

			fqke[n * nqi + i] = v1 * v2;
		}
	}

	free(kl1);
	free(kl2);
}

void get_fq_weight_mat(double *wtqke, double *lkqq, const double *kqq, const double *fqke, unsigned long nke,
		       unsigned long nqi)
{
	double eps;
	int N, N2, INCX, INCY, NRHS, LDA, LDB, info;
	unsigned long i;
	unsigned char UPLO;

	N2 = (int)(nqi * nqi);
	INCX = 1;
	INCY = 1;

	dcopy_(&N2, kqq, &INCX, lkqq, &INCY);

	eps = 1E-7;

	for (i = 0; i < nqi; i++) {
		lkqq[i * nqi + i] += eps;
	}

	N2 = (int)(nqi * nke);

	dcopy_(&N2, fqke, &INCX, wtqke, &INCY);

	UPLO = 'L';
	N = (int)nqi;
	LDA = (int)nqi;
	LDB = (int)nqi;
	NRHS = (int)nke;

	dposv_(&UPLO, &N, &NRHS, lkqq, &LDA, wtqke, &LDB, &info);
	assert(info == 0);
}

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
		const double *ke_ct, unsigned long nke, unsigned int dimke, unsigned long nth, double fac,
		double kf)
{
	double *I_phi, *I2q, *I3q, sqrt_pi, phi_qi, lq, lth, lphi, *gth, *gwth, *th, *wth, exp_th_kj, *q0,
	    *q1, e_ext, dl, diff_th_kj, tmp, thj, I3_jk, I2_jk, sigy2;
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
	sigy2 = l[3] * l[3];

	for (i = 0; i < nq; i++) {
		phi_qi = xq[dimq * i + 2];
		I_phi[i]
		    = 0.5 * sqrt_pi * lphi * (Erf((2.0 * PI - phi_qi) / lphi) - Erf((0 - phi_qi) / lphi));
	}

	for (i = 0; i < nke; i++) {
		e_ext = get_zs_energy_7d_ct(&ke_ct[dimke * i], dimke);
		dl = ke_ct[dimke * i + 0];

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

			Ifq[i * nq + j] = sigy2 * PREFAC * I_phi[j] * tmp;
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

void get_zs_Ifq_num(double *Ifq_num, double *ke_ct, unsigned long nke, unsigned int dimke, double kf,
		    unsigned long nq, unsigned long nth, unsigned long nphi, double *xqi, unsigned long nxqi,
		    unsigned int dimq, double *pq, double fac)
{
	double *xq, *wxq, dl, *qkrn, q, th_q, phi_q, qi, th_qi, phi_qi, tmp, eq, e_ext;
	unsigned int nxq, i, j, k, npq;

	nxq = nth * nq * nphi;
	npq = dimq + 1;

	xq = malloc(dimq * nxq * sizeof(double));
	assert(xq);
	wxq = malloc(nxq * sizeof(double));
	assert(wxq);
	qkrn = malloc(nxq * nxqi * sizeof(double));
	assert(qkrn);

	for (i = 0; i < nke; i++) {

		dl = ke_ct[dimke * i + 0];

		get_ph_space_grid(xq, wxq, dimq, dl, kf, nq, nth, nphi);

		get_krn_se_ard(qkrn, xqi, xq, nxqi, nxq, dimq, pq, npq);

		e_ext = get_zs_energy_7d_ct(&ke_ct[dimke * i], dimke);

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

				tmp += q * q * sin(th_q) * wxq[k] * eq * qkrn[j * nxq + k];
			}

			Ifq_num[i * nxqi + j] = PREFAC * tmp;
		}
	}

	free(xq);
	free(wxq);
}

double get_zs_contact(double g, double kf, double *ke_ct, unsigned int dimke)
{
	double dl, zs_ct;

	dl = ke_ct[0];

	zs_ct = -(16.0 / 3.0) * g * g * dl * dl * kf * kf * kf;

	return PREFAC * 2 * PI * zs_ct;
}

void get_zs_num(double *zs, double *ke_ct, unsigned long nke, unsigned int dimke, double kf, unsigned long nq,
		unsigned long nth, unsigned long nphi, double (*vfun)(double *, unsigned int, double *),
		double *param)
{
	double *xq, *wxq, ph_vol, q, th, phi, e_ext, kl1_ct[DIMKE], kl2_ct[DIMKE], q_ct[DIMQ], v1, v2, tmp,
	    eq, P, P_dl, dlp, dl_dlp, dl, P_dlp, phi_dlp;
	unsigned long nxq, i, n;
	unsigned int dimq;

	nxq = nq * nth * nphi;
	dimq = 3;

	xq = malloc(dimq * nxq * sizeof(double));
	assert(xq);
	wxq = malloc(nxq * sizeof(double));
	assert(wxq);

	for (n = 0; n < nke; n++) {

		dl = ke_ct[dimke * n + 0];

		get_ph_space_grid(xq, wxq, dimq, dl, kf, nq, nth, nphi);

		e_ext = get_zs_energy_7d_ct(&ke_ct[dimke * n], dimke);

		tmp = 0;
		for (i = 0; i < nxq; i++) {

			q = xq[dimq * i + 0];
			th = xq[dimq * i + 1];
			phi = xq[dimq * i + 2];

			eq = -4 * q * dl * cos(th) + e_ext;

			sph_ct_mom_ball(q_ct, &xq[dimq * i], dimq, 1);

			get_zs_loop_mom_7d_ct(kl1_ct, kl2_ct, &ke_ct[dimke * n], dimke, q_ct, 1, dimq);

			v1 = (*vfun)(kl1_ct, dimke, param);
			v2 = (*vfun)(kl2_ct, dimke, param);

			tmp += wxq[i] * q * q * sin(th) * eq * v1 * v2;
		}

		zs[n] = PREFAC * tmp;
	}

	free(xq);
	free(wxq);
}

/*   TESTS   */

double test_get_I2q(unsigned int tn, double q0, double q1, double lq)
{
	double *q, *I2q_exct, *I2q, *qg, *wt, err_norm, diff, qmin[1], qmax[1];
	unsigned long i, j, ng, nth;

	fprintf(stderr, "test_get_I2q() %s:%d\n", __FILE__, __LINE__);

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

	fprintf(stderr, "test_get_I3q() %s:%d\n", __FILE__, __LINE__);

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

double test_Ifq(unsigned long nke, unsigned long nqi, unsigned long nth, double fac, double kmax, double kf,
		int seed)
{
	double *ke_ct, *xqi, *Ikq, *Ikq_num, st[3], en[3], l[4], err_norm;
	unsigned long nq, nphi, i;
	dsfmt_t drng;

	fprintf(stderr, "test_Ifq() %s:%d\n", __FILE__, __LINE__);

	ke_ct = malloc(DIMKE * nke * sizeof(double));
	assert(ke_ct);
	xqi = malloc(DIMQ * nqi * sizeof(double));
	assert(xqi);

	Ikq_num = malloc(nke * nqi * sizeof(double));
	assert(Ikq_num);
	Ikq = malloc(nke * nqi * sizeof(double));
	assert(Ikq);

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ext_momenta_3ball_7d_ct(ke_ct, nke, st, en, seed);
	fill_ext_momenta_ball(xqi, nqi, st[0], en[0], seed + 4545);

	dsfmt_init_gen_rand(&drng, seed + 3443);

	l[0] = 0.5 + 1.5 * dsfmt_genrand_close_open(&drng);
	l[1] = 0.5 + 1.5 * dsfmt_genrand_close_open(&drng);
	l[2] = 0.5 + 1.5 * dsfmt_genrand_close_open(&drng);
	l[3] = 0.1 + 1.5 * dsfmt_genrand_close_open(&drng);

	get_zs_Ifq(Ikq, xqi, nqi, l, DIMQ, ke_ct, nke, DIMKE, nth, fac, kf);

	nq = 50;
	nphi = 50;

	get_zs_Ifq_num(Ikq_num, ke_ct, nke, DIMKE, kf, nq, nth, nphi, xqi, nqi, DIMQ, l, fac);

	err_norm = 0;
	for (i = 0; i < nke * nqi; i++) {
		err_norm += fabs(Ikq[i] - Ikq_num[i]) / fabs(Ikq[i] + Ikq_num[i]);
	}

	free(ke_ct);
	free(xqi);
	free(Ikq);
	free(Ikq_num);

	return err_norm;
}

double test_get_zs_num(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi, int seed)
{
	double *ke_ct, *zs_ct, *zs_ct_comp, kf, g, st[3], en[3], vpar[1], kmax, err;
	unsigned long i;
	unsigned int dimke;

	dimke = 7;

	ke_ct = malloc(nke * dimke * sizeof(double));
	assert(ke_ct);

	zs_ct = malloc(nke * sizeof(double));
	assert(zs_ct);
	zs_ct_comp = malloc(nke * sizeof(double));
	assert(zs_ct_comp);

	g = 0.1;
	kf = 1.3;
	kmax = 2.5;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ext_momenta_3ball_7d_ct(ke_ct, nke, st, en, seed);

	for (i = 0; i < nke; i++) {
		zs_ct_comp[i] = get_zs_contact(g, kf, &ke_ct[dimke * i], dimke);
	}

	vpar[0] = g;

	get_zs_num(zs_ct, ke_ct, nke, dimke, kf, nq, nth, nphi, &v_contact, vpar);

	err = 0;
	for (i = 0; i < nke; i++) {
		if (DEBUG) {
			printf("%+.15E %+.15E\n", zs_ct_comp[i], zs_ct[i]);
		}
		err += fabs(zs_ct_comp[i] - zs_ct[i]);
	}

	free(ke_ct);
	free(zs_ct);
	free(zs_ct_comp);

	return err;
}
