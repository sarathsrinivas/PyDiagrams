#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include "lib_flow.h"

#define PI (3.14159265358979)

void get_zs_th_grid(double *th, double *wth, double *qmin, double *qmax, unsigned long nth, const double *gth,
		    const double *gwth, double dl, double kf, double fac)
{
	double th_max, th_brk, a, b, kf2, dl2, sin_th, cos_th;
	unsigned long i, nth1;

	assert(nth % 4 == 0 && "nth should be multiple of 4.");
	nth1 = nth / 4;

	if (dl < kf) {
		th_max = PI;

		gauss_grid_rescale(gth, gwth, nth1, th, wth, 0, 0.25 * th_max);
		gauss_grid_rescale(gth, gwth, nth1, &th[nth1], &wth[nth1], 0.25 * th_max, 0.5 * th_max);
		gauss_grid_rescale(gth, gwth, nth1, &th[2 * nth1], &wth[2 * nth1], 0.5 * th_max,
				   0.75 * th_max);
		gauss_grid_rescale(gth, gwth, nth1, &th[3 * nth1], &wth[3 * nth1], 0.75 * th_max, th_max);

	} else {
		th_max = asin(kf / dl);
		th_brk = fac * th_max;

		gauss_grid_rescale(gth, gwth, nth1, th, wth, 0, th_brk);
		gauss_grid_rescale(gth, gwth, nth1, &th[nth1], &wth[nth1], th_brk, th_max);
		gauss_grid_rescale(gth, gwth, nth1, &th[2 * nth1], &wth[2 * nth1], PI - th_max, PI - th_brk);
		gauss_grid_rescale(gth, gwth, nth1, &th[3 * nth1], &wth[3 * nth1], PI - th_brk, PI);
	}

	kf2 = kf * kf;
	dl2 = dl * dl;

	for (i = 0; i < nth; i++) {
		cos_th = cos(th[i]);
		sin_th = sin(th[i]);

		a = dl * cos_th;
		b = sqrt(kf2 - dl2 * sin_th * sin_th);

		qmin[i] = fabs(-a + b);
		qmax[i] = fabs(a + b);
	}
}

void get_ph_space_grid(double *xq, double *wxq, unsigned int dimq, double dl, double kf, unsigned long nq,
		       unsigned long nth, unsigned long nphi)
{
	double *q, *wq, *gq, *gwq, ph_vol, fac, *th, *wth, *gth, *gwth, *phi, *wphi, *q0, *q1;
	unsigned long i, j, k, nth1, m;

	assert(nth % 4 == 0);
	nth1 = nth / 4;

	q = malloc(nq * sizeof(double));
	assert(q);
	wq = malloc(nq * sizeof(double));
	assert(wq);
	gq = malloc(nq * sizeof(double));
	assert(gq);
	gwq = malloc(nq * sizeof(double));
	assert(gwq);
	phi = malloc(nphi * sizeof(double));
	assert(phi);
	wphi = malloc(nphi * sizeof(double));
	assert(wphi);
	th = malloc(nth * sizeof(double));
	assert(th);
	wth = malloc(nth * sizeof(double));
	assert(wth);
	gth = malloc(nth1 * sizeof(double));
	assert(gth);
	gwth = malloc(nth1 * sizeof(double));
	assert(gwth);
	q0 = malloc(nth * sizeof(double));
	assert(q0);
	q1 = malloc(nth * sizeof(double));
	assert(q1);

	gauss_grid_create(nq, gq, gwq, -1, 1);
	gauss_grid_create(nth1, gth, gwth, -1, 1);
	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);

	fac = 0.96;
	get_zs_th_grid(th, wth, q0, q1, nth, gth, gwth, dl, kf, fac);

	m = 0;
	for (i = 0; i < nphi; i++) {
		for (j = 0; j < nth; j++) {

			gauss_grid_rescale(gq, gwq, nq, q, wq, q0[j], q1[j]);

			for (k = 0; k < nq; k++) {

				xq[dimq * m + 0] = q[k];
				xq[dimq * m + 1] = th[j];
				xq[dimq * m + 2] = phi[i];

				wxq[m] = wq[k] * wth[j] * wphi[i];

				m++;
			}
		}
	}

	free(q);
	free(wq);
	free(gq);
	free(gwq);
	free(phi);
	free(wphi);
	free(th);
	free(wth);
	free(gth);
	free(gwth);
	free(q0);
	free(q1);
}

double get_ph_space_vol(double dl, double kf, unsigned long nq, unsigned long nth, unsigned long nphi)
{
	double *xq, *wxq, ph_vol, q, th, phi;
	unsigned long nxq, i;
	unsigned int dimq;

	nxq = nq * nth * nphi;
	dimq = 3;

	xq = malloc(dimq * nxq * sizeof(double));
	assert(xq);
	wxq = malloc(nxq * sizeof(double));
	assert(wxq);

	get_ph_space_grid(xq, wxq, dimq, dl, kf, nq, nth, nphi);

	ph_vol = 0;
	for (i = 0; i < nxq; i++) {
		q = xq[dimq * i + 0];
		th = xq[dimq * i + 1];
		phi = xq[dimq * i + 2];

		ph_vol += q * q * sin(th) * wxq[i];
	}

	free(xq);
	free(wxq);

	return ph_vol;
}

double fd(double x, double a, double eps)
{
	double reg;

	reg = 1.0 / (exp((x - a) / eps) + 1.0);

	return reg;
}

double get_ph_space_vol_fd(double dl, double kf, unsigned long nq, unsigned long nth, unsigned long nphi)
{
	double *q, *th, *phi, *wth, *wphi, *wq, qmin, qmax, th_max, vol_ph, q2, dl2, q_pl_dl2, q_mn_dl2, ph,
	    eq, n_pl, n_mn, a, b, q0, q1, sgn, fac, P_dl, t;
	unsigned long i, j, k, nth1;

	nth1 = nth / 4;

	q = malloc(nq * sizeof(double));
	assert(q);
	wq = malloc(nq * sizeof(double));
	assert(wq);
	th = malloc(nth * sizeof(double));
	assert(th);
	wth = malloc(nth * sizeof(double));
	assert(wth);
	phi = malloc(nphi * sizeof(double));
	assert(phi);
	wphi = malloc(nphi * sizeof(double));
	assert(wphi);

	P_dl = 5.0;
	t = 0.01;

	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);
	gauss_grid_create(nth, th, wth, 0, PI);
	gauss_grid_create(nq, q, wq, 0, dl + kf);

	dl2 = dl * dl;

	vol_ph = 0;
	for (i = 0; i < nphi; i++) {
		for (j = 0; j < nth; j++) {
			for (k = 0; k < nq; k++) {

				q2 = q[k] * q[k];

				q_pl_dl2 = q2 + dl2 + 2 * q[k] * dl * cos(th[j]);
				q_mn_dl2 = q2 + dl2 - 2 * q[k] * dl * cos(th[j]);

				ph = fd(q_mn_dl2, kf, t) - fd(q_pl_dl2, kf, t);

				eq = -4 * dl * q[k] * cos(th[j]) + 4 * dl * P_dl;

				vol_ph += ph * eq * q2 * sin(th[j]) * wq[k] * wth[j] * wphi[i];
			}
		}
	}

	free(q);
	free(wq);
	free(th);
	free(wth);
	free(phi);
	free(wphi);

	return vol_ph;
}

double get_ph_vol_exct(double dl, double kf)
{
	double vol_full, vol_exct, vol_int, d;

	vol_full = 4 * (PI / 3.0) * kf * kf * kf;
	d = 2 * dl;

	if (dl < kf) {
		vol_int = (1.0 / 12.0) * PI * (4 * kf + d) * (2 * kf - d) * (2 * kf - d);
	} else {
		vol_int = 0;
	}

	vol_exct = 2 * (vol_full - vol_int);

	return vol_exct;
}
