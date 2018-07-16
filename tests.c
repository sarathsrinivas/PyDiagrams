#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../lib_quadrature/lib_quadrature.h"
#include "../lib_arr/lib_arr.h"
#include "../lib_rng/lib_rng.h"
#include "../lib_pots/lib_pot.h"
#include "lib_flow.h"

#define PI (3.1415926535897)
#define DIM (6)

double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth,
			   unsigned long nphi)
{

	double *gq1, *wq1, *wq, *q, *phi, *wphi, *th, *wth, dl2, Is, Ith, Iq, sgn, th_max, vol_full, vol_cres,
	    vol_int, I_exact, a, b, q0, q1, cos_th, sin_th, pf_q, q2, d;
	unsigned long i, j, l;

	fprintf(stderr, "test_ph_phase_space(dl=%.1f) %s:%d\n", dl, __FILE__, __LINE__);

	gq1 = malloc(nq * sizeof(double));
	assert(gq1);
	wq1 = malloc(nq * sizeof(double));
	assert(wq1);

	q = malloc(nq * sizeof(double));
	assert(q);
	wq = malloc(nq * sizeof(double));
	assert(wq);

	th = malloc(4 * nth * sizeof(double));
	assert(th);
	wth = malloc(4 * nth * sizeof(double));
	assert(wth);

	phi = malloc(nphi * sizeof(double));
	assert(phi);
	wphi = malloc(nphi * sizeof(double));
	assert(wphi);

	gauss_grid_create(nq, gq1, wq1, -1, 1);
	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);

	if (dl < kf) {
		gauss_grid_create(2 * nth, th, wth, 0, 0.5 * PI);
		gauss_grid_create(2 * nth, &th[2 * nth], &wth[2 * nth], PI, 0.5 * PI);
		sgn = 1;
	} else {
		th_max = asin(kf / dl);

		gauss_grid_create(nth, th, wth, 0, fac * th_max);
		gauss_grid_create(nth, &th[nth], &wth[nth], fac * th_max, th_max);

		gauss_grid_create(nth, &th[2 * nth], &wth[2 * nth], PI - th_max, PI - fac * th_max);
		gauss_grid_create(nth, &th[3 * nth], &wth[3 * nth], PI - fac * th_max, PI);

		sgn = -1;
	}

	dl2 = dl * dl;

	Is = 0;
	for (i = 0; i < nphi; i++) {
		Ith = 0;
		for (j = 0; j < 4 * nth; j++) {

			cos_th = cos(th[j]);
			sin_th = sin(th[j]);

			a = dl * cos_th;
			b = sqrt(kf * kf - dl2 * sin_th * sin_th);

			q0 = sgn * (-a + b);
			q1 = a + b;

			gauss_grid_rescale(gq1, wq1, nq, q, wq, q0, q1);

			Iq = 0;
			for (l = 0; l < nq; l++) {
				q2 = q[l] * q[l];
				Iq += wq[l] * q2;
			}

			Ith += wth[j] * sin_th * Iq;
		}

		Is += wphi[i] * Ith;
	}

	vol_full = 4 * (PI / 3) * kf * kf * kf;
	d = 2 * dl;

	if (dl < kf) {
		vol_int = (1.0 / 12.0) * PI * (4 * kf + d) * (2 * kf - d) * (2 * kf - d);
	} else {
		vol_int = 0;
	}

	vol_cres = vol_full - vol_int;
	I_exact = 2 * vol_cres;

	free(wphi);
	free(phi);
	free(wth);
	free(th);
	free(wq);
	free(q);
	free(gq1);
	free(wq1);

	return fabs(I_exact - Is);
}

double test_zs_contact(double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi,
		       int seed)
{
	double *ke, st[6], en[6], *zs, zs_exact, *zs_diff, max_fabs, fac, g;
	unsigned long i;
	int ret;

	fprintf(stderr, "test_zs_contact() %s:%d\n", __FILE__, __LINE__);

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);

	zs = malloc(ns * sizeof(double));
	assert(zs);

	zs_diff = malloc(ns * sizeof(double));
	assert(zs_diff);

	st[0] = 0;
	en[0] = 3 * kf;
	st[1] = 0;
	en[1] = 3 * kf;
	st[2] = 0;
	en[2] = 3 * kf;
	st[3] = 0;
	en[3] = PI;
	st[4] = 0;
	en[4] = PI;
	st[5] = 0;
	en[5] = 2 * PI;

	fill_ext_momenta6(ke, ns, st, en, seed);

	fac = 0.96;

	ret = zs_flow(zs, ke, ns, DIM, kf, nq, nth, nphi, v_contact, NULL, fac);

	g = 0.5;

	for (i = 0; i < ns; i++) {
		zs_exact = zs_contact(&ke[DIM * i], DIM, kf, g);
		zs_diff[i] = fabs(zs_exact - zs[i]);
	}

	max_fabs = get_max_fabs(zs_diff, ns);

	free(zs);
	free(zs_diff);
	free(ke);

	return max_fabs;
}
