#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include "lib_flow.h"

#define PI (3.14159265358979)
#define DIM (7)
#define PREFAC (1 / (8 * PI * PI * PI))

double get_diag_energy_7d_ct(const double *ke_ct, unsigned int dim)
{
	double dlz, dlpx, dlpy, dlpz, Px, Py, Pz, dl_dlp, P_dl, P_dlp, P2, dlp2, dl2, cos_dl_dlp,
	    cos_P_dl, cos_P_dlp, sin_dl_dlp, sin_P_dl, sin_P_dlp, Podl, Podlp, dlodlp, f[4], e_diag;

	dlz = ke_ct[0];
	dlpx = ke_ct[1];
	dlpy = ke_ct[2];
	dlpz = ke_ct[3];
	Px = ke_ct[4];
	Py = ke_ct[5];
	Pz = ke_ct[6];

	P2 = Px * Px + Py * Py + Pz * Pz;
	dlp2 = dlpx * dlpx + dlpy * dlpy + dlpz * dlpz;
	dl2 = dlz * dlz;

	Podl = Pz * dlz;
	Podlp = Px * dlpx + Py * dlpy + Pz * dlpz;
	dlodlp = dlpz * dlz;

	f[0] = P2 + dl2 + dlp2 + 2 * (Podl + Podlp + dlodlp);
	f[1] = P2 + dl2 + dlp2 + 2 * (-Podl - Podlp + dlodlp);
	f[2] = P2 + dl2 + dlp2 + 2 * (-Podl + Podlp - dlodlp);
	f[3] = P2 + dl2 + dlp2 + 2 * (Podl - Podlp - dlodlp);

	e_diag = 0.5 * (f[0] + f[1] - f[2] - f[3]);
	e_diag = -1.0 * e_diag * e_diag;

	return e_diag;
}

double get_zs_energy_7d_ct(const double *ke_ct, unsigned int dim)
{
	double dlz, dlpx, dlpy, dlpz, Px, Py, Pz, dl_dlp, P_dl, P_dlp, P2, dlp2, dl2, cos_dl_dlp,
	    cos_P_dl, cos_P_dlp, sin_dl_dlp, sin_P_dl, sin_P_dlp, Podl, Podlp, dlodlp, f[4], e_ext;

	dlz = ke_ct[0];
	dlpx = ke_ct[1];
	dlpy = ke_ct[2];
	dlpz = ke_ct[3];
	Px = ke_ct[4];
	Py = ke_ct[5];
	Pz = ke_ct[6];

	P2 = Px * Px + Py * Py + Pz * Pz;
	dlp2 = dlpx * dlpx + dlpy * dlpy + dlpz * dlpz;
	dl2 = dlz * dlz;

	Podl = Pz * dlz;
	Podlp = Px * dlpx + Py * dlpy + Pz * dlpz;
	dlodlp = dlpz * dlz;

	f[0] = P2 + dl2 + dlp2 + 2 * (Podl + Podlp + dlodlp);
	f[1] = P2 + dl2 + dlp2 + 2 * (-Podl - Podlp + dlodlp);
	f[2] = P2 + dl2 + dlp2 + 2 * (-Podl + Podlp - dlodlp);
	f[3] = P2 + dl2 + dlp2 + 2 * (Podl - Podlp - dlodlp);
	e_ext = 0.5 * (f[0] - f[1] - f[2] + f[3]);

	return e_ext;
}

void get_zs_loop_mom_7d_ct(double *kl1_ct, double *kl2_ct, const double *ke_ct, unsigned int dimke,
			   const double *q_ct, unsigned long nq, unsigned int dimq)
{
	double dl_ct[3], dlp_ct[3], P_ct[3], dlp_zs1[3], dlp_zs2[3], P_zs1[3], P_zs2[3];
	unsigned long i, l;

	dl_ct[0] = 0;
	dl_ct[1] = 0;
	dl_ct[2] = ke_ct[0];

	dlp_ct[0] = ke_ct[1];
	dlp_ct[1] = ke_ct[2];
	dlp_ct[2] = ke_ct[3];

	P_ct[0] = ke_ct[4];
	P_ct[1] = ke_ct[5];
	P_ct[2] = ke_ct[6];

	for (l = 0; l < nq; l++) {

		for (i = 0; i < 3; i++) {

			dlp_zs1[i] = 0.5 * (P_ct[i] + dlp_ct[i] - q_ct[l * dimq + i]);

			P_zs1[i] = 0.5 * (P_ct[i] + dlp_ct[i] + q_ct[l * dimq + i]);

			dlp_zs2[i] = 0.5 * (P_ct[i] - dlp_ct[i] - q_ct[l * dimq + i]);

			P_zs2[i] = 0.5 * (P_ct[i] - dlp_ct[i] + q_ct[l * dimq + i]);
		}

		kl1_ct[l * dimke + 0] = dl_ct[2];
		kl1_ct[l * dimke + 1] = dlp_zs1[0];
		kl1_ct[l * dimke + 2] = dlp_zs1[1];
		kl1_ct[l * dimke + 3] = dlp_zs1[2];
		kl1_ct[l * dimke + 4] = P_zs1[0];
		kl1_ct[l * dimke + 5] = P_zs1[1];
		kl1_ct[l * dimke + 6] = P_zs1[2];

		kl2_ct[l * dimke + 0] = dl_ct[2];
		kl2_ct[l * dimke + 1] = dlp_zs2[0];
		kl2_ct[l * dimke + 2] = dlp_zs2[1];
		kl2_ct[l * dimke + 3] = dlp_zs2[2];
		kl2_ct[l * dimke + 4] = P_zs2[0];
		kl2_ct[l * dimke + 5] = P_zs2[1];
		kl2_ct[l * dimke + 6] = P_zs2[2];
	}
}

void get_zs_num_7d_ct(double *zs_ct, double *ke_ct, unsigned long nke, unsigned int dimke,
		      double kf, unsigned long nq, unsigned long nth, unsigned long nphi,
		      double (*vfun)(double *, unsigned int, double *), double *param)
{
	double *xq, *wxq, ph_vol, q, th, phi, e_ext, *kl1_ct, *kl2_ct, v1, v2, tmp, eq, P, P_dl,
	    dlp, dl_dlp, dl, P_dlp, phi_dlp, *q_ct;
	unsigned long nxq, i, n;
	unsigned int dimq;

	nxq = nq * nth * nphi;
	dimq = 3;

	kl1_ct = malloc(dimke * nxq * sizeof(double));
	assert(kl1_ct);
	kl2_ct = malloc(dimke * nxq * sizeof(double));
	assert(kl2_ct);

	xq = malloc(dimq * nxq * sizeof(double));
	assert(xq);
	q_ct = malloc(dimq * nxq * sizeof(double));
	assert(q_ct);
	wxq = malloc(nxq * sizeof(double));
	assert(wxq);

	for (n = 0; n < nke; n++) {

		dl = ke_ct[dimke * n + 0];

		get_ph_space_grid(xq, wxq, dimq, dl, kf, nq, nth, nphi);

		sph_to_ct(q_ct, xq, dimq, nxq);

		get_zs_loop_mom_7d_ct(kl1_ct, kl2_ct, &ke_ct[dimke * n], dimke, q_ct, nxq, dimq);

		e_ext = get_zs_energy_7d_ct(&ke_ct[dimke * n], dimke);

		tmp = 0;

		for (i = 0; i < nxq; i++) {
			q = xq[dimq * i + 0];
			th = xq[dimq * i + 1];
			phi = xq[dimq * i + 2];

			eq = -4 * q * dl * cos(th) + e_ext;

			v1 = (*vfun)(&kl1_ct[dimke * i], dimke, param);
			v2 = (*vfun)(&kl2_ct[dimke * i], dimke, param);

			tmp += wxq[i] * q * q * sin(th) * eq * v1 * v2;
		}

		zs_ct[n] = PREFAC * tmp;
	}

	free(xq);
	free(wxq);
	free(kl1_ct);
	free(kl2_ct);
}
