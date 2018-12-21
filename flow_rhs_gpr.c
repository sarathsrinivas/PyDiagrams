#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include <lib_gpr/lib_gpr.h>
#include "lib_flow.h"

#define PI (3.14159265358979)
#define PREFAC (1)
#define DIM (6)

int zs_flow_gpr(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, double kmax,
		unsigned long nq, unsigned long nth, unsigned long nphi, double fac, const double *p, int np,
		const double *wt, double *lkrxx)
{
	double dl, dl2, P_dl, dl_dlp, P_dlp, phi_dlp, e_ext, *q, *wq, *gth1, *wth1, *th, *wth, *phi, *wphi,
	    *w, cos_th, sin_th, a, b, q0, q1, e_q, q2, *gma1, *gma2, *kl1, *kl2, pf_th, pf_phi, pf_q, *gq1,
	    *wq1, sgn, th_max, wphi_i, wth_j, *krpx, *krpp, *var, lims[2], Q0, Q1, limsgn;
	unsigned long n, i, j, l, m, nm, k;

	sgn = 1;

	nm = nth * nphi * nq;

	w = malloc(nm * sizeof(double));
	assert(w);

	gma1 = malloc(nm * sizeof(double));
	assert(gma1);

	gma2 = malloc(nm * sizeof(double));
	assert(gma2);

	kl1 = calloc(DIM * nm, sizeof(double));
	assert(kl1);

	kl2 = calloc(DIM * nm, sizeof(double));
	assert(kl2);

	gq1 = malloc(nq * sizeof(double));
	assert(gq1);
	wq1 = malloc(nq * sizeof(double));
	assert(wq1);

	gth1 = malloc(nth * sizeof(double));
	assert(gth1);
	wth1 = malloc(nth * sizeof(double));
	assert(wth1);

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

	krpx = malloc(nm * ns * sizeof(double));
	assert(krpx);

	krpp = malloc(nm * nm * sizeof(double));
	assert(krpp);

	var = malloc(nm * nm * sizeof(double));
	assert(var);

	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);

	gauss_grid_create(nth, gth1, wth1, -1, 1);
	gauss_grid_create(nq, gq1, wq1, -1, 1);

	for (n = 0; n < ns; n++) {

		dl = ext_mom[dim * n + 0];
		dl_dlp = ext_mom[dim * n + 3];
		P_dl = ext_mom[dim * n + 4];
		P_dlp = ext_mom[dim * n + 5];

		if (dl < kf) {

			/*
			gauss_grid_rescale(gth1, wth1, nth, th, wth, 0, 0.25 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[nth], &wth[nth], 0.25 * PI, 0.5 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[2 * nth], &wth[2 * nth], PI, 0.75 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[3 * nth], &wth[3 * nth], 0.75 * PI, 0.5 * PI);
			*/

			gauss_grid_create(nth, th, wth, 0, PI);

		} else if (dl > kf) {
			th_max = asin(kf / dl);

			gauss_grid_rescale(gth1, wth1, nth, th, wth, 0, fac * th_max);
			gauss_grid_rescale(gth1, wth1, nth, &th[nth], &wth[nth], fac * th_max, th_max);
			gauss_grid_rescale(gth1, wth1, nth, &th[2 * nth], &wth[2 * nth], PI - th_max,
					   PI - fac * th_max);
			gauss_grid_rescale(gth1, wth1, nth, &th[3 * nth], &wth[3 * nth], PI - fac * th_max,
					   PI);

			sgn = -1;
		}

		dl2 = dl * dl;

		phi_dlp = acos((cos(P_dlp) - cos(P_dl) * cos(dl_dlp)) / (sin(P_dl) * sin(dl_dlp)));

		e_ext = get_zs_energy(&ext_mom[dim * n], dim);

		m = 0;
		for (i = 0; i < nphi; i++) {
			wphi_i = wphi[i];
			for (j = 0; j < 4 * nth; j++) {
				wth_j = wth[j];
				cos_th = cos(th[j]);
				sin_th = sin(th[j]);

				a = dl * cos_th;
				b = sqrt(kf * kf - dl2 * sin_th * sin_th);

				q0 = sgn * (-a + b);
				q1 = a + b;

				gauss_grid_rescale(gq1, wq1, nq, q, wq, q0, q1);

				for (l = 0; l < nq; l++) {

					q2 = q[l] * q[l];

					e_q = -2 * q[l] * dl * cos_th;

					get_zs_loop_mom_ct(&kl1[DIM * m], &kl2[DIM * m], dim,
							   &ext_mom[n * dim], phi_dlp, q[l], th[j], phi[i]);

					w[m] = wq[l] * wphi_i * wth_j * sin_th * q2 * 2 * (e_q + e_ext);

					m++;
				}
			}
		}

		get_krn_se_ard(krpx, kl1, ext_mom, nm, ns, dim, p, np);
		gpr_predict(gma1, wt, krpx, nm, ns);

		get_krn_se_ard(krpp, kl1, kl1, nm, nm, dim, p, np);
		get_var_mat_chd(var, krpp, krpx, lkrxx, nm, ns);

		get_krn_se_ard(krpx, kl2, ext_mom, nm, ns, dim, p, np);
		gpr_predict(gma2, wt, krpx, nm, ns);

		zs[n] = 0;
		for (m = 0; m < nm; m++) {
			zs[n] += w[m] * gma1[m] * gma2[m];
		}

		zs[n] *= PREFAC;
	}

	free(wphi);
	free(phi);
	free(wth);
	free(th);
	free(wq);
	free(q);
	free(wth1);
	free(gth1);
	free(wq1);
	free(gq1);
	free(gma2);
	free(gma1);
	free(w);
	free(krpx);
	free(krpp);
	free(var);
	free(kl1);
	free(kl2);

	return 0;
}

int zsp_flow_gpr(double *zsp, double *ext_mom, unsigned long ns, unsigned int dim, double kf,
		 unsigned long nq, unsigned long nth, unsigned long nphi, double fac, const double *p, int np,
		 const double *wt)

{
	double dlp, dlp2, P_dl, dl_dlp, P_dlp, phi_dl, e_ext, *q, *wq, *gth1, *wth1, *th, *wth, *phi, *wphi,
	    *w, cos_th, sin_th, a, b, q0, q1, e_q, q2, *gma1, *gma2, kl1[DIM], kl2[DIM], pf_th, pf_phi, pf_q,
	    *gq1, *wq1, sgn, th_max, wphi_i, wth_j, *krpx;
	unsigned long n, i, j, l, m, nm;

	sgn = 1;

	nm = 4 * nth * nphi * nq;

	w = malloc(nm * sizeof(double));
	assert(w);

	gma1 = malloc(nm * sizeof(double));
	assert(gma1);

	gma2 = malloc(nm * sizeof(double));
	assert(gma2);

	gq1 = malloc(nq * sizeof(double));
	assert(gq1);
	wq1 = malloc(nq * sizeof(double));
	assert(wq1);

	gth1 = malloc(nth * sizeof(double));
	assert(gth1);
	wth1 = malloc(nth * sizeof(double));
	assert(wth1);

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

	krpx = malloc(nm * ns * sizeof(double));
	assert(krpx);

	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);

	gauss_grid_create(nth, gth1, wth1, -1, 1);
	gauss_grid_create(nq, gq1, wq1, -1, 1);

	for (n = 0; n < ns; n++) {

		dlp = ext_mom[dim * n + 1];
		dl_dlp = ext_mom[dim * n + 3];
		P_dl = ext_mom[dim * n + 4];
		P_dlp = ext_mom[dim * n + 5];

		if (dlp < kf) {

			gauss_grid_rescale(gth1, wth1, nth, th, wth, 0, 0.25 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[nth], &wth[nth], 0.25 * PI, 0.5 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[2 * nth], &wth[2 * nth], PI, 0.75 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[3 * nth], &wth[3 * nth], 0.75 * PI, 0.5 * PI);

			sgn = 1;

		} else if (dlp > kf) {
			th_max = asin(kf / dlp);

			gauss_grid_rescale(gth1, wth1, nth, th, wth, 0, fac * th_max);
			gauss_grid_rescale(gth1, wth1, nth, &th[nth], &wth[nth], fac * th_max, th_max);
			gauss_grid_rescale(gth1, wth1, nth, &th[2 * nth], &wth[2 * nth], PI - th_max,
					   PI - fac * th_max);
			gauss_grid_rescale(gth1, wth1, nth, &th[3 * nth], &wth[3 * nth], PI - fac * th_max,
					   PI);

			sgn = -1;
		}

		dlp2 = dlp * dlp;

		phi_dl = acos((cos(P_dl) - cos(P_dlp) * cos(dl_dlp)) / (sin(P_dlp) * sin(dl_dlp)));

		e_ext = get_zsp_energy(&ext_mom[dim * n], dim);

		m = 0;
		for (i = 0; i < nphi; i++) {
			wphi_i = wphi[i];
			for (j = 0; j < 4 * nth; j++) {
				wth_j = wth[j];
				cos_th = cos(th[j]);
				sin_th = sin(th[j]);

				a = dlp * cos_th;
				b = sqrt(kf * kf - dlp2 * sin_th * sin_th);

				q0 = sgn * (-a + b);
				q1 = a + b;

				gauss_grid_rescale(gq1, wq1, nq, q, wq, q0, q1);

				for (l = 0; l < nq; l++) {

					q2 = q[l] * q[l];

					e_q = -2 * q[l] * dlp * cos_th;

					get_zsp_loop_mom_ct(kl1, kl2, dim, &ext_mom[n * dim], phi_dl, q[l],
							    th[j], phi[i]);

					w[m] = wq[l] * wphi_i * wth_j * sin_th * q2 * 2 * (e_q + e_ext);

					m++;
				}
			}
		}

		get_krn_se_ard(krpx, kl1, ext_mom, nm, ns, dim, p, np);
		gpr_predict(gma1, wt, krpx, nm, ns);

		get_krn_se_ard(krpx, kl2, ext_mom, nm, ns, dim, p, np);
		gpr_predict(gma2, wt, krpx, nm, ns);

		zsp[n] = 0;
		for (m = 0; m < nm; m++) {
			zsp[n] += w[m] * gma1[m] * gma2[m];
		}

		zsp[n] *= PREFAC;
	}

	free(wphi);
	free(phi);
	free(wth);
	free(th);
	free(wq);
	free(q);
	free(wth1);
	free(gth1);
	free(wq1);
	free(gq1);
	free(gma2);
	free(gma1);
	free(w);
	free(krpx);

	return 0;
}
