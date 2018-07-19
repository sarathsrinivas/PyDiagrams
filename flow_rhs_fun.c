#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../lib_quadrature/lib_quadrature.h"
#include "lib_flow.h"

#define PI (3.14159265358979)
#define PREFAC (1)
#define DIM (6)
/*
#define PREFAC (1 / (8 * PI * PI * PI))
*/

double zs_contact(const double *ke, unsigned int dim, double kf, double g)
{
	double dl, dlp, P, dl_dlp, P_dl, P_dlp, P2, dl2, dl4, cos_P_dl, zs_ct, kf2, kf3, g2;

	dl = ke[0];
	dlp = ke[1];
	P = ke[2];
	dl_dlp = ke[3];
	P_dl = ke[4];
	P_dlp = ke[5];

	P2 = P * P;
	dl2 = dl * dl;
	dl4 = dl2 * dl2;
	cos_P_dl = cos(P_dl);
	kf2 = kf * kf;
	kf3 = kf2 * kf;
	g2 = g * g;

	zs_ct = 0;

	if (dl < kf) {
		zs_ct = -(16.0 / 3) * g2 * (P * dl4 - 3 * P * dl2 * kf2) * cos_P_dl;
	} else if (dl > kf) {
		zs_ct = (16.0 / 3) * g2 * kf3 * (2 * P * dl * cos_P_dl - dl2);
	}

	return 2 * PI * zs_ct;
}

void get_zs_loop_mom(double *kl1, double *kl2, unsigned int dim, const double *ke, double phi_dlp, double q,
		     double q_th, double q_phi)
{
	double dl, dlp, P, dl_dlp, P_dl, P_dlp, P2, dlp2, dl2, cos_dl_dlp, cos_P_dl, cos_P_dlp, sin_dl_dlp,
	    sin_P_dl, sin_P_dlp, Podl, Podlp, dlodlp, cos_th, sin_th, phi_P, Poq, dlpoq, dloq, dl_zs1,
	    dlp_zs1, P_zs1, dl_dlp_zs1, P_dl_zs1, P_dlp_zs1, dl_zs2, dlp_zs2, P_zs2, dl_dlp_zs2, P_dl_zs2,
	    P_dlp_zs2, cos_P_q, cos_dlp_q, q2;

	dl = ke[0];
	dlp = ke[1];
	P = ke[2];
	dl_dlp = ke[3];
	P_dl = ke[4];
	P_dlp = ke[5];

	P2 = P * P;
	dlp2 = dlp * dlp;
	dl2 = dl * dl;

	cos_dl_dlp = cos(dl_dlp);
	cos_P_dl = cos(P_dl);
	cos_P_dlp = cos(P_dlp);

	sin_dl_dlp = sin(dl_dlp);
	sin_P_dl = sin(P_dl);
	sin_P_dlp = sin(P_dlp);

	Podl = P * dl * cos_P_dl;
	Podlp = P * dlp * cos_P_dlp;
	dlodlp = dl * dlp * cos_dl_dlp;

	q2 = q * q;
	cos_th = cos(q_th);
	sin_th = sin(q_th);

	phi_P = 0;

	cos_P_q = cos_P_dl * cos_th + sin_P_dl * sin_th * cos(q_phi - phi_P);
	cos_dlp_q = cos_dl_dlp * cos_th + sin_dl_dlp * sin_th * cos(q_phi - phi_dlp);

	Poq = P * q * cos_P_q;
	dlpoq = dlp * q * cos_dlp_q;
	dloq = dl * q * cos_th;

	dlp_zs1 = 0.5 * sqrt(P2 + q2 + dlp2 + 2 * (-Poq - dlpoq + Podlp));
	P_zs1 = sqrt(P2 + q2 + dlp2 + 2 * (Poq + dlpoq + Podlp));
	dl_dlp_zs1 = acos(0.5 * (Podl - dloq + dlodlp) / (dl * dlp_zs1));
	P_dl_zs1 = acos((Podl + dloq + dlodlp) / (P_zs1 * dl));
	P_dlp_zs1 = acos(0.5 * (P2 - q2 + dlp2 + 2 * Podlp) / (P_zs1 * dlp_zs1));

	dlp_zs2 = 0.5 * sqrt(P2 + q2 + dlp2 + 2 * (-Poq + dlpoq - Podlp));
	P_zs2 = sqrt(P2 + q2 + dlp2 + 2 * (Poq - dlpoq - Podlp));
	dl_dlp_zs2 = acos(0.5 * (-Podl + dloq + dlodlp) / (dl * dlp_zs2));
	P_dl_zs2 = acos((Podl + dloq - dlodlp) / (P_zs2 * dl));
	P_dlp_zs2 = acos(0.5 * (-P2 + q2 - dlp2 + 2 * Podlp) / (P_zs2 * dlp_zs2));

	kl1[0] = dl;
	kl1[1] = dlp_zs1;
	kl1[2] = P_zs1;
	kl1[3] = dl_dlp_zs1;
	kl1[4] = P_dl_zs1;
	kl1[5] = P_dlp_zs1;

	kl2[0] = dl;
	kl2[1] = dlp_zs2;
	kl2[2] = P_zs2;
	kl2[3] = dl_dlp_zs2;
	kl2[4] = P_dl_zs2;
	kl2[5] = P_dlp_zs2;
}

static double get_zs_energy(const double *ke, unsigned int dim)
{
	double dl, dlp, P, dl_dlp, P_dl, P_dlp, P2, dlp2, dl2, cos_dl_dlp, cos_P_dl, cos_P_dlp, sin_dl_dlp,
	    sin_P_dl, sin_P_dlp, Podl, Podlp, dlodlp, f[4], e_ext;

	dl = ke[0];
	dlp = ke[1];
	P = ke[2];
	dl_dlp = ke[3];
	P_dl = ke[4];
	P_dlp = ke[5];

	P2 = P * P;
	dlp2 = dlp * dlp;
	dl2 = dl * dl;

	cos_dl_dlp = cos(dl_dlp);
	cos_P_dl = cos(P_dl);
	cos_P_dlp = cos(P_dlp);

	sin_dl_dlp = sin(dl_dlp);
	sin_P_dl = sin(P_dl);
	sin_P_dlp = sin(P_dlp);

	Podl = P * dl * cos_P_dl;
	Podlp = P * dlp * cos_P_dlp;
	dlodlp = dl * dlp * cos_dl_dlp;

	f[0] = P2 + dl2 + dlp2 + 2 * (Podl + Podlp + dlodlp);
	f[1] = P2 + dl2 + dlp2 + 2 * (-Podl - Podlp + dlodlp);
	f[2] = P2 + dl2 + dlp2 + 2 * (-Podl + Podlp - dlodlp);
	f[3] = P2 + dl2 + dlp2 + 2 * (Podl - Podlp - dlodlp);
	e_ext = 0.5 * (f[0] - f[1] - f[2] + f[3]);

	return e_ext;
}

int zs_flow(double *zs, double *ext_mom, unsigned long ns, unsigned int dim, double kf, unsigned long nq,
	    unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *),
	    double *param, double fac)
{
	double dl, dl2, P_dl, dl_dlp, P_dlp, phi_dlp, e_ext, *q, *wq, *gth1, *wth1, *th, *wth, *phi, *wphi,
	    *w, cos_th, sin_th, a, b, q0, q1, e_q, q2, *gma1, *gma2, kl1[DIM], kl2[DIM], pf_th, pf_phi, pf_q,
	    *gq1, *wq1, sgn, th_max, wphi_i, wth_j;
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

	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);

	gauss_grid_create(nth, gth1, wth1, -1, 1);
	gauss_grid_create(nq, gq1, wq1, -1, 1);

	for (n = 0; n < ns; n++) {

		dl = ext_mom[dim * n + 0];
		dl_dlp = ext_mom[dim * n + 3];
		P_dl = ext_mom[dim * n + 4];
		P_dlp = ext_mom[dim * n + 5];

		if (dl < kf) {

			gauss_grid_rescale(gth1, wth1, nth, th, wth, 0, 0.25 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[nth], &wth[nth], 0.25 * PI, 0.5 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[2 * nth], &wth[2 * nth], PI, 0.75 * PI);
			gauss_grid_rescale(gth1, wth1, nth, &th[3 * nth], &wth[3 * nth], 0.75 * PI, 0.5 * PI);

			sgn = 1;

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

					get_zs_loop_mom(kl1, kl2, dim, ext_mom, phi_dlp, q[l], th[j], phi[i]);

					gma1[m] = vfun(kl1, dim, param);
					gma2[m] = vfun(kl2, dim, param);

					w[m] = wq[l] * wphi_i * wth_j * sin_th * q2 * 2 * (e_q + e_ext);

					m++;
				}
			}
		}

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

	return 0;
}
