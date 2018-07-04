#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../lib_quadrature/lib_quadrature.h"
#include "lib_flow.h"

#define PI (3.14159265358979)
#define PREFAC (1)
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

	if (dl < kf) {
		zs_ct = -(16.0 / 3) * g2 * (P * dl4 - 3 * P * dl2 * kf2) * cos_P_dl;
	} else if (dl > kf) {
		zs_ct = (16.0 / 3) * g2 * kf3 * (2 * P * dl * cos_P_dl - dl2);
	}

	return 2 * PI * zs_ct;
}

double *zs_flow(double *ext_mom, unsigned long ns, unsigned int dim1, double kf, unsigned long nq,
		unsigned long nth, unsigned long nphi, double vfun(double *, unsigned int, double *),
		unsigned int dim, double *param, double fac)
{
	double f[4], dl, dlp, P, dl_dlp, P_dl, P_dlp, P2, dlp2, dl2, cos_dl_dlp, cos_P_dl, cos_P_dlp,
	    sin_dl_dlp, sin_P_dl, sin_P_dlp, Podl, Podlp, dlodlp, phi_P, phi_dlp, e_ext, *q, *wq, *th, *wth,
	    *phi, *wphi, *zs, zsq, cos_th, sin_th, a, b, q0, q1, e_q, q2, cos_P_q, cos_dlp_q, Poq, dlpoq,
	    dlp_zs1, dlp_zs2, P_zs1, P_zs2, gma1, gma2, *kl, pf_th, pf_phi, pf_q, *gq1, sgn, th_max, zsth;
	unsigned long n, i, j, l;

	sgn = 1;

	zs = malloc(ns * sizeof(double));
	assert(zs);

	kl = malloc(dim * sizeof(double));
	assert(kl);

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

	gauss_grid_create(nphi, phi, wphi, 0, 2 * PI);

	gauss_grid_create(nq, gq1, wq, -1, 1);

	for (n = 0; n < ns; n++) {

		dl = ext_mom[dim1 * n + 0];
		dlp = ext_mom[dim1 * n + 1];
		P = ext_mom[dim1 * n + 2];
		dl_dlp = ext_mom[dim1 * n + 3];
		P_dl = ext_mom[dim1 * n + 4];
		P_dlp = ext_mom[dim1 * n + 5];

		if (dl < kf) {
			gauss_grid_create(2 * nth, th, wth, 0, 0.5 * PI);
			gauss_grid_create(2 * nth, &th[2 * nth], &wth[2 * nth], PI, 0.5 * PI);
			sgn = 1;
		} else {
			th_max = asin(kf / dl);
			/*
			fac = 0.96;
			*/

			gauss_grid_create(nth, th, wth, 0, fac * th_max);
			gauss_grid_create(nth, &th[nth], &wth[nth], fac * th_max, th_max);

			gauss_grid_create(nth, &th[2 * nth], &wth[2 * nth], PI - th_max, PI - fac * th_max);
			gauss_grid_create(nth, &th[3 * nth], &wth[3 * nth], PI - fac * th_max, PI);

			sgn = -1;
		}

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

		phi_P = 0;
		phi_dlp = acos((cos_P_dlp - cos_P_dl * cos_dl_dlp) / (sin_P_dl * sin_dl_dlp));

		f[0] = P2 + dl2 + dlp2 + 2 * (Podl + Podlp + dlodlp);
		f[1] = P2 + dl2 + dlp2 + 2 * (-Podl - Podlp + dlodlp);
		f[2] = P2 + dl2 + dlp2 + 2 * (-Podl + Podlp - dlodlp);
		f[3] = P2 + dl2 + dlp2 + 2 * (Podl - Podlp - dlodlp);
		e_ext = 0.5 * (f[0] - f[1] - f[2] + f[3]);

		zs[n] = 0;
		for (i = 0; i < nphi; i++) {
			zsth = 0;
			for (j = 0; j < 4 * nth; j++) {
				cos_th = cos(th[j]);
				sin_th = sin(th[j]);

				a = dl * cos_th;
				b = sqrt(kf * kf - dl2 * sin_th * sin_th);

				q0 = sgn * (-a + b);
				q1 = a + b;

				gauss_grid_rescale(gq1, nq, q, q0, q1);
				pf_q = 0.5 * (q1 - q0);

				zsq = 0;
				for (l = 0; l < nq; l++) {

					q2 = q[l] * q[l];

					e_q = -2 * q[l] * dl * cos_th;

					cos_P_q = cos_P_dl * cos_th + sin_P_dl * sin_th * cos(phi[i] - 0);
					cos_dlp_q = cos_dl_dlp * cos_th
						    + sin_dl_dlp * sin_th * cos(phi[i] - phi_dlp);

					Poq = P * q[l] * cos_P_q;
					dlpoq = dlp * q[l] * cos_dlp_q;

					dlp_zs1 = 0.5 * sqrt(P2 + q2 + dlp2 + 2 * (-Poq - dlpoq + Podlp));
					P_zs1 = sqrt(P2 + q2 + dlp2 + 2 * (Poq + dlpoq + Podlp));

					dlp_zs2 = 0.5 * sqrt(P2 + q2 + dlp2 + 2 * (-Poq + dlpoq - Podlp));
					P_zs2 = sqrt(P2 + q2 + dlp2 + 2 * (Poq - dlpoq - Podlp));

					kl[0] = dl;
					kl[1] = dlp_zs1;
					gma1 = vfun(kl, dim, param);

					kl[0] = dl;
					kl[1] = dlp_zs2;
					gma2 = vfun(kl, dim, param);

					zsq += wq[l] * q2 * gma1 * gma2 * 2 * (e_q + e_ext);
				}
				zsth += wth[j] * sin_th * zsq * pf_q;
			}

			zs[n] += wphi[i] * zsth;
		}
		zs[n] *= PREFAC;
	}

	free(gq1);
	free(q);
	free(wq);
	free(th);
	free(wth);
	free(phi);
	free(wphi);
	free(kl);

	return zs;
}
