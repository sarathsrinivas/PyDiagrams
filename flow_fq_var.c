#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include <atlas/lapack.h>
#include <atlas/blas.h>
#include <lib_rng/lib_rng.h>
#include "lib_flow.h"

#define PI (3.1415926535897)
#define PREFAC (1 / (8 * PI * PI * PI))

double get_I22(double q0, double q1, double qi0, double qi1, double lq)
{
	double I22, q02, q03, q04, q05, q12, q13, q14, q15, l, l2, l3, l4, l5, l6, qi02, qi03, qi04, qi05,
	    qi12, qi13, qi14, qi15, sqpi, qdiff00, qdiff01, qdiff10, qdiff11;

	sqpi = sqrt(PI);

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;

	q02 = q0 * q0;
	q03 = q02 * q0;
	q04 = q03 * q0;
	q05 = q04 * q0;

	q12 = q1 * q1;
	q13 = q12 * q1;
	q14 = q13 * q1;
	q15 = q14 * q1;

	qi02 = qi0 * qi0;
	qi03 = qi02 * qi0;
	qi04 = qi03 * qi0;
	qi05 = qi04 * qi0;

	qi12 = qi1 * qi1;
	qi13 = qi12 * qi1;
	qi14 = qi13 * qi1;
	qi15 = qi14 * qi1;

	qdiff00 = (q0 - qi0) * (q0 - qi0);
	qdiff01 = (q0 - qi1) * (q0 - qi1);
	qdiff10 = (q1 - qi0) * (q1 - qi0);
	qdiff11 = (q1 - qi1) * (q1 - qi1);

	I22 = (l
	       * (2 * l
		      * ((-l4 - l2 * qdiff00 - 3 * (q04 + q03 * qi0 + q02 * qi02 + q0 * qi03 + qi04))
			     * exp(-qdiff00 / l2)
			 + (l4 + l2 * qdiff10 + 3 * (q14 + q13 * qi0 + q12 * qi02 + q1 * qi03 + qi04))
			       * exp(-qdiff10 / l2)
			 + (l4 + l2 * qdiff01 + 3 * (q04 + q03 * qi1 + q02 * qi12 + q0 * qi13 + qi14))
			       * exp(-qdiff01 / l2)
			 + (-l4 - l2 * qdiff11 - 3 * (q14 + q13 * qi1 + q12 * qi12 + q1 * qi13 + qi14))
			       * exp(-qdiff11 / l2))
		  + sqpi * (-6 * q05 + 6 * qi05 + 5 * l2 * (-q03 + qi03)) * Erf((q0 - qi0) / l)
		  + sqpi * (5 * l2 * (q13 - qi03) + 6 * (q15 - qi05)) * Erf((q1 - qi0) / l)
		  + sqpi
			* ((5 * l2 * (q03 - qi13) + 6 * (q05 - qi15)) * Erf((q0 - qi1) / l)
			   + qi13 * (5 * l2 + 6 * qi12) * Erf((q1 - qi1) / l)
			   + q13 * (5 * l2 + 6 * q12) * Erf((-q1 + qi1) / l))))
	      / 60.0;

	return I22;
}

double get_I23(double q0, double q1, double qi0, double qi1, double lq)
{
	double I23, q02, q03, q04, q05, q06, q12, q13, q14, q15, q16, l, l2, l4, l3, l5, l6, qi02, qi03, qi04,
	    qi05, qi06, qi12, qi13, qi14, qi15, qi16, sqpi, qdiff00, qdiff01, qdiff10, qdiff11;

	sqpi = sqrt(PI);

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;
	l5 = l4 * l;
	l6 = l5 * l;

	q02 = q0 * q0;
	q03 = q02 * q0;
	q04 = q03 * q0;
	q05 = q04 * q0;
	q06 = q05 * q0;

	q12 = q1 * q1;
	q13 = q12 * q1;
	q14 = q13 * q1;
	q15 = q14 * q1;
	q16 = q15 * q1;

	qi02 = qi0 * qi0;
	qi03 = qi02 * qi0;
	qi04 = qi03 * qi0;
	qi05 = qi04 * qi0;
	qi06 = qi05 * qi0;

	qi12 = qi1 * qi1;
	qi13 = qi12 * qi1;
	qi14 = qi13 * qi1;
	qi15 = qi14 * qi1;
	qi16 = qi15 * qi1;

	qdiff00 = (q0 - qi0) * (q0 - qi0);
	qdiff01 = (q0 - qi1) * (q0 - qi1);
	qdiff10 = (q1 - qi0) * (q1 - qi0);
	qdiff11 = (q1 - qi1) * (q1 - qi1);

	I23 = (l
	       * ((-2 * l
		   * (3 * l4 * (q0 - qi0) + 2 * l2 * (q03 - 3 * q02 * qi0 + 3 * q0 * qi02 + 7 * qi03)
		      + 8 * (q05 + q04 * qi0 + q03 * qi02 + q02 * qi03 + q0 * qi04 + qi05)))
		      * exp(-qdiff00 / l2)
		  + (2 * l
		     * (3 * l4 * (q1 - qi0) + 2 * l2 * (q13 - 3 * q12 * qi0 + 3 * q1 * qi02 + 7 * qi03)
			+ 8 * (q15 + q14 * qi0 + q13 * qi02 + q12 * qi03 + q1 * qi04 + qi05)))
			* exp(-qdiff10 / l2)
		  + (2 * l
		     * (3 * l4 * (q0 - qi1) + 2 * l2 * (q03 - 3 * q02 * qi1 + 3 * q0 * qi12 + 7 * qi13)
			+ 8 * (q05 + q04 * qi1 + q03 * qi12 + q02 * qi13 + q0 * qi14 + qi15)))
			* exp(-qdiff01 / l2)
		  - (2 * l
		     * (3 * l4 * (q1 - qi1) + 2 * l2 * (q13 - 3 * q12 * qi1 + 3 * q1 * qi12 + 7 * qi13)
			+ 8 * (q15 + q14 * qi1 + q13 * qi12 + q12 * qi13 + q1 * qi14 + qi15)))
			* exp(-qdiff11 / l2)
		  + sqpi * (3 * l6 - 12 * l2 * (q04 - 3 * qi04) + 16 * (-q06 + qi06)) * Erf((q0 - qi0) / l)
		  + sqpi * (-3 * l6 + 12 * l2 * (q14 - 3 * qi04) + 16 * (q16 - qi06)) * Erf((q1 - qi0) / l)
		  + 4 * sqpi * (3 * l2 * (q04 - 3 * qi14) + 4 * (q06 - qi16)) * Erf((q0 - qi1) / l)
		  + sqpi * (3 * l6 + 36 * l2 * qi14 + 16 * qi16) * Erf((q1 - qi1) / l)
		  + 3 * l6 * sqpi * Erf((-q0 + qi1) / l) + 12 * l2 * sqpi * q14 * Erf((-q1 + qi1) / l)
		  + 16 * sqpi * q16 * Erf((-q1 + qi1) / l)))
	      / 192.0;

	return I23;
}

double get_I33(double q0, double q1, double qi0, double qi1, double lq)
{
	double I33, q02, q03, q04, q05, q06, q07, q12, q13, q14, q15, q16, q17, l, l2, l4, l3, l5, l6, qi02,
	    qi03, qi04, qi05, qi06, qi07, qi12, qi13, qi14, qi15, qi16, qi17, sqpi, qdiff00, qdiff01, qdiff10,
	    qdiff11;

	sqpi = sqrt(PI);

	l = lq;
	l2 = l * l;
	l3 = l2 * l;
	l4 = l3 * l;
	l5 = l4 * l;
	l6 = l5 * l;

	q02 = q0 * q0;
	q03 = q02 * q0;
	q04 = q03 * q0;
	q05 = q04 * q0;
	q06 = q05 * q0;
	q07 = q06 * q0;

	q12 = q1 * q1;
	q13 = q12 * q1;
	q14 = q13 * q1;
	q15 = q14 * q1;
	q16 = q15 * q1;
	q17 = q16 * q1;

	qi02 = qi0 * qi0;
	qi03 = qi02 * qi0;
	qi04 = qi03 * qi0;
	qi05 = qi04 * qi0;
	qi06 = qi05 * qi0;
	qi07 = qi06 * qi0;

	qi12 = qi1 * qi1;
	qi13 = qi12 * qi1;
	qi14 = qi13 * qi1;
	qi15 = qi14 * qi1;
	qi16 = qi15 * qi1;
	qi17 = qi16 * qi1;

	qdiff00 = (q0 - qi0) * (q0 - qi0);
	qdiff01 = (q0 - qi1) * (q0 - qi1);
	qdiff10 = (q1 - qi0) * (q1 - qi0);
	qdiff11 = (q1 - qi1) * (q1 - qi1);

	I33 = (l
	       * ((l
		   * (3 * l6 + 3 * l4 * qdiff00
		      - l2 * (16 * q04 + 6 * q03 * qi0 - 9 * q02 * qi02 + 6 * q0 * qi03 + 16 * qi04)
		      - 10 * (q06 + q05 * qi0 + q04 * qi02 + q03 * qi03 + q02 * qi04 + q0 * qi05 + qi06)))
		      * exp(-qdiff00 / l2)
		  + (l
		     * (-3 * l6 - 3 * l4 * qdiff10
			+ l2 * (16 * q14 + 6 * q13 * qi0 - 9 * q12 * qi02 + 6 * q1 * qi03 + 16 * qi04)
			+ 10 * (q16 + q15 * qi0 + q14 * qi02 + q13 * qi03 + q12 * qi04 + q1 * qi05 + qi06)))
			* exp(-qdiff10 / l2)
		  + (l
		     * (-3 * l6 - 3 * l4 * qdiff01
			+ l2 * (16 * q04 + 6 * q03 * qi1 - 9 * q02 * qi12 + 6 * q0 * qi13 + 16 * qi14)
			+ 10 * (q06 + q05 * qi1 + q04 * qi12 + q03 * qi13 + q02 * qi14 + q0 * qi15 + qi16)))
			* exp(-qdiff01 / l2)
		  + (l
		     * (3 * l6 + 3 * l4 * qdiff11
			- l2 * (16 * q14 + 6 * q13 * qi1 - 9 * q12 * qi12 + 6 * q1 * qi13 + 16 * qi14)
			- 10 * (q16 + q15 * qi1 + q14 * qi12 + q13 * qi13 + q12 * qi14 + q1 * qi15 + qi16)))
			* exp(-qdiff11 / l2)
		  - sqpi * (21 * l2 * (q05 - qi05) + 10 * (q07 - qi07)) * Erf((q0 - qi0) / l)
		  + sqpi * (21 * l2 * (q15 - qi05) + 10 * (q17 - qi07)) * Erf((q1 - qi0) / l)
		  + sqpi * (21 * l2 * (q05 - qi15) + 10 * (q07 - qi17)) * Erf((q0 - qi1) / l)
		  + sqpi * qi15 * (21 * l2 + 10 * qi12) * Erf((q1 - qi1) / l)
		  + 21 * l2 * sqpi * q15 * Erf((-q1 + qi1) / l) + 10 * sqpi * q17 * Erf((-q1 + qi1) / l)))
	      / 140.;

	return I33;
}

void get_zs_II(double *II, const double *ke, unsigned long nke, unsigned int dimke, const double *lxq,
	       unsigned long nth, double fac, double kf)
{
	double *gth, *gwth, *th, *wth, *q0, *q1, dl, x, xi, wi, lq, lth, lphi, diff_th_kj, exp_th_kj, qmin,
	    qmax, qimin, qimax, I22, I33, I32, I23, IIphi, lphi2, dl2, e_ext, e_ext2, tmp, sin_thj, sigy2;
	unsigned long nth1, i, j, k;

	assert(nth % 4 == 0);
	nth1 = nth / 4;

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

	lq = lxq[0];
	lth = lxq[1];
	lphi = lxq[2];
	sigy2 = lxq[3] * lxq[3];

	lphi2 = lphi * lphi;

	IIphi = lphi2 * (exp(-(4 * PI * PI / lphi2)) - 1.0) + 2 * lphi * PI * sqrt(PI) * Erf(2 * PI / lphi);

	for (i = 0; i < nke; i++) {

		e_ext = get_zs_energy(&ke[dimke * i], dimke);
		dl = ke[dimke * i + 0];

		dl2 = dl * dl;
		e_ext2 = e_ext * e_ext;

		get_zs_th_grid(th, wth, q0, q1, nth, gth, gwth, dl, kf, fac);

		tmp = 0;
		for (j = 0; j < nth; j++) {

			xi = cos(th[j]);
			sin_thj = sin(th[j]);
			wi = wth[j];
			qimin = q0[j];
			qimax = q1[j];

			for (k = 0; k < nth; k++) {

				x = cos(th[k]);
				qmin = q0[k];
				qmax = q1[k];

				diff_th_kj = (th[k] - th[j]) / lth;
				exp_th_kj = exp(-diff_th_kj * diff_th_kj);

				I22 = get_I22(qmin, qmax, qimin, qimax, lq);
				I23 = get_I23(qmin, qmax, qimin, qimax, lq);
				I32 = get_I23(qimin, qimax, qmin, qmax, lq);
				I33 = get_I33(qmin, qmax, qimin, qimax, lq);

				tmp += wi * wth[k] * sin_thj * sin(th[k]) * exp_th_kj
				       * (16 * dl2 * xi * x * I33 - 4 * dl * e_ext * (xi * I32 + x * I23)
					  + e_ext2 * I22);
			}
		}
		II[i] = PREFAC * PREFAC * sigy2 * IIphi * tmp;
	}

	free(q1);
	free(q0);
	free(wth);
	free(th);
	free(gwth);
	free(gth);
}

/* Iq^T (K^-1) Iq */

double get_integ_covar(const double *Iq, const double *kqq_chlsk, unsigned long nxq, double *tmp_vec)
{

	double Icv, *vec, ALPHA, BETA;
	int N, M, INCX, INCY, LDA, LDB, NRHS, INFO;
	unsigned long i;
	unsigned char UPLO, TRANS, SIDE, DIAG;

	for (i = 0; i < nxq; i++) {
		tmp_vec[i] = Iq[i];
	}

	SIDE = 'L';
	UPLO = 'L';
	TRANS = 'N';
	DIAG = 'N';
	N = nxq;
	NRHS = 1;
	LDA = N;
	LDB = N;
	ALPHA = 1.0;
	dpotrs_(&UPLO, &N, &NRHS, kqq_chlsk, &LDA, tmp_vec, &LDB, &INFO);

	assert(INFO == 0);

	INCX = 1;
	INCY = 1;
	Icv = ddot_(&N, tmp_vec, &INCX, Iq, &INCY);

	return Icv;
}

void get_zs_II_num(double *II, const double *ke, unsigned long nke, unsigned int dimke, const double *lxq,
		   unsigned int dimq, unsigned long nq, unsigned long nth, unsigned long nphi, double fac,
		   double kf)
{

	double *gth, *gwth, *th, *wth, *q0, *q1, dl, x, xi, wi, lq, lth, lphi, diff_th_kj, exp_th_kj, qmin,
	    qmax, qimin, qimax, I22, I33, I32, I23, IIphi, lphi2, dl2, e_ext, e_ext2, tmp, *q, *wq, *gq, *gwq,
	    *phi, *wphi, xq[3], Ifq[1];
	unsigned long nth1, i, j, k, m;

	assert(nth % 4 == 0);
	nth1 = nth / 4;

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
	gq = malloc(nq * sizeof(double));
	assert(gq);
	gwq = malloc(nq * sizeof(double));
	assert(gwq);
	q = malloc(nq * sizeof(double));
	assert(q);
	wq = malloc(nq * sizeof(double));
	assert(wq);
	phi = malloc(nphi * sizeof(double));
	assert(phi);
	wphi = malloc(nphi * sizeof(double));
	assert(wphi);

	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);
	gauss_grid_create(nth1, gth, gwth, -1, 1);
	gauss_grid_create(nq, gq, gwq, -1, 1);

	for (i = 0; i < nke; i++) {

		e_ext = get_zs_energy(&ke[dimke * i], dimke);
		dl = ke[dimke * i + 0];

		get_zs_th_grid(th, wth, q0, q1, nth, gth, gwth, dl, kf, fac);

		tmp = 0;
		for (m = 0; m < nphi; m++) {
			for (j = 0; j < nth; j++) {

				gauss_grid_rescale(gq, gwq, nq, q, wq, q0[j], q1[j]);

				for (k = 0; k < nq; k++) {

					xq[0] = q[k];
					xq[1] = th[j];
					xq[2] = phi[m];

					get_zs_Ifq(Ifq, xq, 1, lxq, dimq, &ke[dimke * i], 1, dimke, nth, fac,
						   kf);

					tmp += wq[k] * wth[j] * wphi[m] * q[k] * q[k] * sin(th[j])
					       * (-4 * q[k] * dl * cos(th[j]) + e_ext) * Ifq[0];
				}
			}
		}

		II[i] = PREFAC * tmp;
	}

	free(wphi);
	free(phi);
	free(wq);
	free(q);
	free(gwq);
	free(gq);
	free(q1);
	free(q0);
	free(wth);
	free(th);
	free(gwth);
	free(gth);
}

/* TESTS */
double test_Imn(double qmin, double qmax, double qimin, double qimax, unsigned long nq)
{
	double I22, I22_num, I33, I33_num, I23, I23_num, I32, I32_num, *q, *wq, *qi, *wqi, rel_err, qi2, qi3,
	    q2, q3, Kqqi, lq;
	unsigned long i, j;

	fprintf(stderr, "test_get_zs_Imn() %s:%d\n", __FILE__, __LINE__);

	q = malloc(nq * sizeof(double));
	assert(q);
	wq = malloc(nq * sizeof(double));
	assert(wq);
	qi = malloc(nq * sizeof(double));
	assert(qi);
	wqi = malloc(nq * sizeof(double));
	assert(wqi);

	gauss_grid_create(nq, q, wq, qmin, qmax);
	gauss_grid_create(nq, qi, wqi, qimin, qimax);

	lq = 1.0;

	I22_num = 0;
	I23_num = 0;
	I32_num = 0;
	I33_num = 0;
	for (i = 0; i < nq; i++) {
		for (j = 0; j < nq; j++) {

			q2 = q[j] * q[j];
			q3 = q2 * q[j];
			qi2 = qi[i] * qi[i];
			qi3 = qi2 * qi[i];

			Kqqi = exp(-(qi[i] - q[j]) * (qi[i] - q[j]) / (lq * lq));

			I22_num += qi2 * q2 * Kqqi * wqi[i] * wq[j];
			I23_num += qi2 * q3 * Kqqi * wqi[i] * wq[j];
			I32_num += qi3 * q2 * Kqqi * wqi[i] * wq[j];
			I33_num += qi3 * q3 * Kqqi * wqi[i] * wq[j];
		}
	}

	I22 = get_I22(qmin, qmax, qimin, qimax, lq);
	I23 = get_I23(qmin, qmax, qimin, qimax, lq);
	I32 = get_I23(qimin, qimax, qmin, qmax, lq);
	I33 = get_I33(qmin, qmax, qimin, qimax, lq);

	rel_err = 0;
	rel_err += fabs(I22_num - I22) / fabs(I22_num + I22);
	rel_err += fabs(I23_num - I23) / fabs(I23_num + I23);
	rel_err += fabs(I32_num - I32) / fabs(I32_num + I32);
	rel_err += fabs(I33_num - I33) / fabs(I33_num + I33);

	free(q);
	free(wq);
	free(qi);
	free(wqi);

	return rel_err;
}

double test_get_zs_II(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi, double kmax,
		      double kf, int seed)
{
	double *ke, fac, *II_num, *II, st[3], en[3], l[4], abs_err;
	unsigned int dimke, dimq;
	unsigned long i;
	dsfmt_t drng;

	fprintf(stderr, "test_get_zs_II() %s:%d\n", __FILE__, __LINE__);

	dimke = 6;
	dimq = 3;

	ke = malloc(dimke * nke * sizeof(double));
	assert(ke);
	II_num = malloc(nke * sizeof(double));
	assert(II_num);
	II = malloc(nke * sizeof(double));
	assert(II);

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ext_momenta_3ball(ke, nke, st, en, seed);

	dsfmt_init_gen_rand(&drng, seed + 3443);

	l[0] = 1.2;
	l[1] = 0.8;
	l[2] = 0.5;
	l[3] = 0.3;

	fac = 0.96;

	get_zs_II(II, ke, nke, dimke, l, nth, fac, kf);

	get_zs_II_num(II_num, ke, nke, dimke, l, dimq, nq, nth, nphi, fac, kf);

	abs_err = 0;
	for (i = 0; i < nke; i++) {
		abs_err += fabs(II[i] - II_num[i]);
	}

	return abs_err;
}

double test_get_integ_covar(unsigned long n, int seed)
{

	double *Id, *vec, *tmp_vec, Icv, Icv_exct;
	int N, INCX, INCY;
	unsigned long i;
	dsfmt_t drng;

	fprintf(stderr, "test_get_integ_covar() %s:%d\n", __FILE__, __LINE__);

	Id = calloc(n * n, sizeof(double));
	assert(Id);
	vec = malloc(n * n * sizeof(double));
	assert(vec);
	tmp_vec = malloc(n * n * sizeof(double));
	assert(tmp_vec);

	dsfmt_init_gen_rand(&drng, seed);

	for (i = 0; i < n; i++) {
		vec[i] = 2.0 * dsfmt_genrand_close_open(&drng);
	}

	for (i = 0; i < n; i++) {
		Id[i * n + i] = 1.0;
	}

	Icv = get_integ_covar(vec, Id, n, tmp_vec);

	N = n;
	INCX = 1;
	INCY = 1;
	Icv_exct = ddot_(&N, vec, &INCX, vec, &INCY);

	free(Id);
	free(vec);
	free(tmp_vec);

	return fabs(Icv - Icv_exct);
}
