#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../lib_quadrature/lib_quadrature.h"
#include "../lib_arr/lib_arr.h"
#include "lib_flow.h"

#define PI (3.1415926535897)

double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth,
			   unsigned long nphi)
{
	double *gq1, *wq, *q, *phi, *wphi, *th, *wth, dl2, Is, Ith, Iq, sgn, th_max, vol_full, vol_cres,
	    vol_int, I_exact, a, b, q0, q1, cos_th, sin_th, pf_q, q2, d;
	unsigned long i, j, l;

	gq1 = malloc(nq * sizeof(double));
	assert(gq1);

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

	gauss_grid_create(nq, gq1, wq, -1, 1);
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

			gauss_grid_rescale(gq1, nq, q, q0, q1);
			pf_q = 0.5 * (q1 - q0);

			Iq = 0;
			for (l = 0; l < nq; l++) {
				q2 = q[l] * q[l];
				Iq += wq[l] * q2;
			}

			Ith += wth[j] * sin_th * Iq * pf_q;
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

	return fabs(I_exact - Is);
}
