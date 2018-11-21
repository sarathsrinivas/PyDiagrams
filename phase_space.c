#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../lib_quadrature/lib_quadrature.h"
#include "lib_flow.h"

#define PI (3.14159265358979)

double fd(double x, double a, double eps)
{
	double reg;

	reg = 1.0 / (exp((x - a) / eps) + 1.0);

	return reg;
}

unsigned long get_ph_phase_space(double kf, double dl, unsigned long nq, unsigned long nth,
				 unsigned long nphi, double fac_th_brk, double *gl_q, double *wl_q,
				 double *gl_th, double *wl_th, double *gl_phi, double *wl_phi, double *qvec,
				 double *wt, double *work_mem, unsigned long nwork)
{

	double q0, q1, qmin, qmax, th_max, a, b, *th, *phi, *q, *wq, *wth, *wphi, sgn, th_brk, sin_th, w_th,
	    w_q, w_phi, q2, th_t, phi_t, t, q_pl_dl2, q_mn_dl2, ph, dl2, cos_th, kmax;
	unsigned long nth1, i, j, k, l, nl;

	nl = nq * nth * nphi;

	assert(nth % 4 == 0 && "nth should be multiple of 4.");
	assert(nwork == 2 * nl && "Not enough memory for *work_mem.");

	nth1 = nth / 4;

	if (work_mem) {
		q = work_mem;
		wq = &work_mem[nq];
		th = &work_mem[2 * nq];
		wth = &work_mem[2 * nq + nth];
		phi = &work_mem[2 * nq + 2 * nth];
		wphi = &work_mem[2 * nq + 2 * nth + nphi];
	} else {
		return 2 * nq * nth * nphi;
	}

	if (dl < kf) {
		th_max = PI;

		gauss_grid_rescale(gl_th, wl_th, nth1, th, wth, 0, 0.25 * th_max);
		gauss_grid_rescale(gl_th, wl_th, nth1, &th[nth1], &wth[nth1], 0.25 * th_max, 0.5 * th_max);
		gauss_grid_rescale(gl_th, wl_th, nth1, &th[2 * nth1], &wth[2 * nth1], 0.5 * th_max,
				   0.75 * th_max);
		gauss_grid_rescale(gl_th, wl_th, nth1, &th[3 * nth1], &wth[3 * nth1], 0.75 * th_max, th_max);
		sgn = 1;

	} else {
		th_max = asin(kf / dl);
		th_brk = fac_th_brk * th_max;

		gauss_grid_rescale(gl_th, wl_th, nth1, th, wth, 0, th_brk);
		gauss_grid_rescale(gl_th, wl_th, nth1, &th[nth1], &wth[nth1], th_brk, th_max);
		gauss_grid_rescale(gl_th, wl_th, nth1, &th[2 * nth1], &wth[2 * nth1], PI - th_max,
				   PI - th_brk);
		gauss_grid_rescale(gl_th, wl_th, nth1, &th[3 * nth1], &wth[3 * nth1], PI - th_brk, PI);
		sgn = -1;
	}

	gauss_grid_rescale(gl_phi, wl_phi, nphi, phi, wphi, 0, 2 * PI);

	dl2 = dl * dl;
	t = 0.1;
	kmax = 2;

	for (i = 0; i < nphi; i++) {
		w_phi = wphi[i];
		phi_t = phi[i];
		for (j = 0; j < nth; j++) {

			w_th = wth[j];
			th_t = th[j];

			sin_th = sin(th[j]);
			cos_th = cos(th[j]);

			a = (dl * cos_th);
			b = sqrt(kf * kf - dl * dl * sin_th * sin_th);

			q0 = fabs(-a + b);
			q1 = fabs(a + b);

			qmin = (q0 < q1) ? q0 : q1;
			qmax = (q0 > q1) ? q0 : q1;

			gauss_grid_rescale(gl_q, wl_q, nq, q, wq, q0, q1);

			for (k = 0; k < nq; k++) {

				l = nth * nq * i + nq * j + k;

				q2 = q[k] * q[k];

				qvec[3 * l + 0] = q[k];
				qvec[3 * l + 1] = th_t;
				qvec[3 * l + 2] = phi_t;

				wt[l] = q2 * sin(th_t) * wq[k] * w_th * w_phi;
			}
		}
	}

	return 0;
}

double get_ph_space_vol(double dl, double kf, unsigned long nq, unsigned long nth, unsigned long nphi)
{
	double *q, *th, *phi, *wth, *wphi, *wq, qmin, qmax, th_max, vol_ph, q2, dl2, q_pl_dl2, q_mn_dl2, ph,
	    eq, n_pl, n_mn, a, b, q0, q1, sgn, fac, P_dl;
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

	fac = 0.96;

	if (dl < kf) {
		th_max = PI;
		gauss_grid_create(nth, th, wth, 0, th_max);
	} else {
		th_max = asin(kf / dl);
		gauss_grid_create(nth1, th, wth, 0, fac * th_max);
		gauss_grid_create(nth1, &th[nth1], &wth[nth1], fac * th_max, th_max);
		gauss_grid_create(nth1, &th[2 * nth1], &wth[2 * nth1], PI - th_max, PI - fac * th_max);
		gauss_grid_create(nth1, &th[3 * nth1], &wth[3 * nth1], PI - fac * th_max, PI);
	}
	qmin = 0;
	qmax = dl + kf;
	P_dl = 0.5;

	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);

	dl2 = dl * dl;

	vol_ph = 0;
	for (i = 0; i < nphi; i++) {
		for (j = 0; j < nth; j++) {

			a = (dl * cos(th[j]));
			b = sqrt(kf * kf - dl2 * sin(th[j]) * sin(th[j]));

			q0 = fabs(-a + b);
			q1 = fabs(a + b);

			qmin = (q0 < q1) ? q0 : q1;
			qmax = (q0 > q1) ? q0 : q1;

			gauss_grid_create(nq, q, wq, q0, q1);

			for (k = 0; k < nq; k++) {

				q2 = q[k] * q[k];

				eq = -4 * dl * q[k] * cos(th[j]) + 4 * dl * P_dl;

				vol_ph += eq * q2 * sin(th[j]) * wq[k] * wth[j] * wphi[i];
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
