#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_gpr/lib_gpr.h>
#include "lib_flow.h"

#define EXP_HALF (0.6065306597126334) /* exp(-0.5) */

unsigned long get_work_sz_etd4rk_gpr(unsigned long n)
{
	unsigned long nwork;

	nwork = 7 * n;

	return nwork;
}

unsigned long get_work_sz_etd34rk_gpr(unsigned long n)
{
	unsigned long nwork;

	nwork = 9 * n;

	return nwork;
}

void get_expz_gpr(double *expz, double *J, double h, unsigned long n)
{
	unsigned long i;

	for (i = 0; i < n; i++) {
		expz[i] = exp(J[i] * h);
	}
}

void get_enf_gpr(double *enf, double *J, double h, unsigned long n)
{
	double z, y, z2, z3, z4;
	unsigned long i;

	for (i = 0; i < n; i++) {

		z = h * J[i];
		if (fabs(z) > 1E-10) {

			y = exp(z);
			enf[i] = (y - 1.0) / log(y);

		} else {
			z2 = z * z;
			z3 = z2 * z;
			z4 = z3 * z;

			enf[i] = 1.0 + z / 2.0 + z2 / 6.0 + z3 / 24.0 + z4 / 120.0;
		}
	}
}

void get_etd4rk_coeff_gpr(double *alp, double *bet, double *gam, double *J, double h,
			  unsigned long n)
{
	double z, z2, z3, z4, z5;
	unsigned long i;
	double ca[6] = {1.0 / 6.0, 1.0 / 6.0, 3.0 / 40.0, 1.0 / 45.0, 5.0 / 1008.0, 1.0 / 1120.0};
	double cb[6] = {1.0 / 6.0, 1.0 / 12.0, 1.0 / 40.0, 1.0 / 180.0, 1.0 / 1008.0, 1.0 / 6720.0};
	double cg[6] = {1.0 / 6.0, 0.0, -1.0 / 120.0, -1.0 / 360.0, -1.0 / 1680.0, 1.0 / 10080.0};

	for (i = 0; i < n; i++) {

		z = J[i] * h;
		z2 = z * z;
		z3 = z2 * z;

		if (fabs(z) <= 10E-4) {

			z4 = z3 * z;
			z5 = z4 * z;

			alp[i]
			    = ca[0] + ca[1] * z + ca[2] * z2 + ca[3] * z3 + ca[4] * z4 + ca[5] * z5;
			bet[i]
			    = cb[0] + cb[1] * z + cb[2] * z2 + cb[3] * z3 + cb[4] * z4 + cb[5] * z5;
			gam[i]
			    = cg[0] + cg[1] * z + cg[2] * z2 + cg[3] * z3 + cg[4] * z4 + cg[5] * z5;

		} else {
			alp[i] = (-4.0 - z + exp(z) * (4.0 - 3 * z + z2)) / z3;
			bet[i] = (2.0 + z + exp(z) * (z - 2)) / z3;
			gam[i] = (-4.0 - 3 * z - z2 + exp(z) * (4 - z)) / z3;
		}
	}
}

void etd4rk_vec_step_gpr(double t0, unsigned long m, double *y0, double h, double *J,
			 double *exp_hj2, double *enf_jh2, double *alph, double *bet, double *gam,
			 void fun(double *f, double t, double *u1, unsigned long mn, void *param),
			 void *param, double *y, double *work, unsigned long nwork)
{
	double *a, *b, *c, *f0, *fa, *fb, *fc;
	unsigned long i;

	assert(work);
	assert(nwork == get_work_sz_etd4rk(m));
	a = &work[0 * m];
	b = &work[1 * m];
	c = &work[2 * m];
	f0 = &work[3 * m];
	fa = &work[4 * m];
	fb = &work[5 * m];
	fc = &work[6 * m];

	fun(f0, t0, y0, m, param);

	for (i = 0; i < m; i++) {
		a[i] = y0[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * f0[i];
	}

	fun(fa, t0 + 0.5 * h, a, m, param);

	for (i = 0; i < m; i++) {
		b[i] = y0[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * fa[i];
	}

	fun(fb, t0 + 0.5 * h, b, m, param);

	for (i = 0; i < m; i++) {
		c[i] = a[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * (2 * fb[i] - f0[i]);
	}

	fun(fc, t0 + h, c, m, param);

	for (i = 0; i < m; i++) {
		y[i] = y0[i] * exp_hj2[i] * exp_hj2[i]
		       + h * (f0[i] * alph[i] + 2 * (fa[i] + fb[i]) * bet[i] + fc[i] * gam[i]);
	}
}

void etd4rk_vec_gpr(double t0, double tn, double h, double *y0, unsigned long n, double *J,
		    void fn(double *f, double t, double *u1, unsigned long mn, void *param),
		    void *param)
{
	double *dy, *work, *alp, *bet, *gam, *exp_jh2, *enf_jh2, t;
	unsigned long nwork, i;

	nwork = get_work_sz_etd4rk(n);

	work = malloc(nwork * sizeof(double));
	assert(work);
	dy = malloc(n * sizeof(double));
	assert(dy);
	exp_jh2 = malloc(n * sizeof(double));
	assert(exp_jh2);
	enf_jh2 = malloc(n * sizeof(double));
	assert(enf_jh2);
	alp = malloc(n * sizeof(double));
	assert(alp);
	bet = malloc(n * sizeof(double));
	assert(bet);
	gam = malloc(n * sizeof(double));
	assert(gam);

	get_expz(exp_jh2, J, 0.5 * h, n);
	get_enf(enf_jh2, J, 0.5 * h, n);
	get_etd4rk_coeff(alp, bet, gam, J, h, n);

	for (t = t0; t <= tn; t += h) {
		fprintf(stderr, "\r t = %+.15E  h = %.15E", t, h);

		etd4rk_vec_step(t, n, y0, h, J, exp_jh2, enf_jh2, alp, bet, gam, fn, param, dy,
				work, nwork);

		for (i = 0; i < n; i++) {
			y0[i] = dy[i];
		}
	}

	fprintf(stderr, "\n");

	free(work);
	free(dy);
	free(alp);
	free(bet);
	free(gam);
	free(exp_jh2);
	free(enf_jh2);
}

void etd34rk_vec_step_gpr(double t0, unsigned long m, double *y0, double h, double *J,
			  double *exp_hj2, double *enf_jh2, double *enf_jh, double *alph,
			  double *bet, double *gam, double *y0_smp_mn, double *y0_zs1_mn,
			  double *y0_zs2_mn, double *y0_zsp1_mn, double *y0_zsp2_mn,
			  double *exp_zs1, double *exp_zs2, double *exp_zsp1, double *exp_zsp2,
			  void fun(double *f, double t, double *u1, unsigned long mn, void *param),
			  void *param, double *y, double *eg, double *work, unsigned long nwork)
{
	double *a, *b, *b3, *c, *f0, *fa, *fb, *fc, *fb3;
	unsigned long i;

	assert(work);
	assert(nwork == get_work_sz_etd34rk(m));
	a = &work[0 * m];
	b = &work[1 * m];
	c = &work[2 * m];
	f0 = &work[3 * m];
	fa = &work[4 * m];
	fb = &work[5 * m];
	fc = &work[6 * m];
	b3 = &work[7 * m];
	fb3 = &work[8 * m];

	fun(f0, t0, y0, m, param);

	for (i = 0; i < m; i++) {
		a[i] = y0[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * f0[i];
	}

	fun(fa, t0 + 0.5 * h, a, m, param);

	for (i = 0; i < m; i++) {
		b[i] = y0[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * fa[i];
		b3[i] = y0[i] * exp_hj2[i] * exp_hj2[i] + h * enf_jh[i] * (2 * fa[i] - f0[i]);
	}

	fun(fb, t0 + 0.5 * h, b, m, param);
	fun(fb3, t0 + h, b3, m, param);

	for (i = 0; i < m; i++) {
		c[i] = a[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * (2 * fb[i] - f0[i]);
	}

	fun(fc, t0 + h, c, m, param);

	for (i = 0; i < m; i++) {
		y[i] = y0[i] * exp_hj2[i] * exp_hj2[i]
		       + h * (f0[i] * alph[i] + 2 * (fa[i] + fb[i]) * bet[i] + fc[i] * gam[i]);

		eg[i] += h * fabs(2 * bet[i] * (fa[i] - fb[i]) + gam[i] * (fc[i] - fb3[i]));
	}
}

void etd34rk_vec_gpr(double t0, double tn, double h, double *y0, unsigned long n, double *J,
		     void fn(double *f, double t, double *u1, unsigned long mn, void *param),
		     void *param, double *eg)
{
	double *dy, *work, *alp, *bet, *gam, *exp_jh2, *enf_jh2, *enf_jh, t;
	unsigned long nwork, i;

	nwork = get_work_sz_etd34rk(n);

	work = malloc(nwork * sizeof(double));
	assert(work);
	dy = malloc(n * sizeof(double));
	assert(dy);
	exp_jh2 = malloc(n * sizeof(double));
	assert(exp_jh2);
	enf_jh2 = malloc(n * sizeof(double));
	assert(enf_jh2);
	enf_jh = malloc(n * sizeof(double));
	assert(enf_jh);
	alp = malloc(n * sizeof(double));
	assert(alp);
	bet = malloc(n * sizeof(double));
	assert(bet);
	gam = malloc(n * sizeof(double));
	assert(gam);

	for (i = 0; i < n; i++) {
		eg[i] = 0;
	}

	get_expz(exp_jh2, J, 0.5 * h, n);
	get_enf(enf_jh2, J, 0.5 * h, n);
	get_enf(enf_jh, J, h, n);
	get_etd4rk_coeff(alp, bet, gam, J, h, n);

	for (t = t0; t <= tn; t += h) {
		fprintf(stderr, "\r t = %+.15E  h = %.15E", t, h);

		etd34rk_vec_step(t, n, y0, h, J, exp_jh2, enf_jh2, enf_jh, alp, bet, gam, fn, param,
				 dy, eg, work, nwork);

		for (i = 0; i < n; i++) {
			y0[i] = dy[i];
		}
	}

	fprintf(stderr, "\n");

	free(work);
	free(dy);
	free(alp);
	free(bet);
	free(gam);
	free(exp_jh2);
	free(enf_jh);
	free(enf_jh2);
}
