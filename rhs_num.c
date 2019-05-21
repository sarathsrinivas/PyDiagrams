#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_pots/lib_pots.h>
#include "lib_flow.h"

#define PI (3.1415926535897)
#define PREFAC (1 / (8 * PI * PI * PI))
#define DIMKE (7)
#define DIMQ (3)

double get_zs_contact(double g, double kf, double *ke_ct, unsigned int dimke)
{
	double dl, zs_ct;

	dl = ke_ct[0];

	zs_ct = -(16.0 / 3.0) * g * g * dl * dl * kf * kf * kf;

	return PREFAC * 2 * PI * zs_ct;
}

void get_zs_num(double *zs, double *ke_ct, unsigned long nke, unsigned int dimke, double kf, unsigned long nq,
		unsigned long nth, unsigned long nphi, double (*vfun)(double *, unsigned int, double *),
		double *param)
{
	double *xq, *wxq, ph_vol, q, th, phi, e_ext, kl1_ct[DIMKE], kl2_ct[DIMKE], q_ct[DIMQ], v1, v2, tmp,
	    eq, P, P_dl, dlp, dl_dlp, dl, P_dlp, phi_dlp;
	unsigned long nxq, i, n;
	unsigned int dimq;

	nxq = nq * nth * nphi;
	dimq = 3;

	xq = malloc(dimq * nxq * sizeof(double));
	assert(xq);
	wxq = malloc(nxq * sizeof(double));
	assert(wxq);

	for (n = 0; n < nke; n++) {

		dl = ke_ct[dimke * n + 0];

		get_ph_space_grid(xq, wxq, dimq, dl, kf, nq, nth, nphi);

		e_ext = get_zs_energy_7d_ct(&ke_ct[dimke * n], dimke);

		tmp = 0;
		for (i = 0; i < nxq; i++) {

			q = xq[dimq * i + 0];
			th = xq[dimq * i + 1];
			phi = xq[dimq * i + 2];

			eq = -4 * q * dl * cos(th) + e_ext;

			sph_to_ct(q_ct, &xq[dimq * i], dimq, 1);

			get_zs_loop_mom_7d_ct(kl1_ct, kl2_ct, &ke_ct[dimke * n], dimke, q_ct, 1, dimq);

			v1 = (*vfun)(kl1_ct, dimke, param);
			v2 = (*vfun)(kl2_ct, dimke, param);

			tmp += wxq[i] * q * q * sin(th) * eq * v1 * v2;
		}

		zs[n] = PREFAC * tmp;
	}

	free(xq);
	free(wxq);
}

/* TESTS */

double test_get_zs_num(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi, int seed)
{
	double *ke_ct, *zs_ct, *zs_ct_comp, kf, g, st[3], en[3], vpar[1], kmax, err;
	unsigned long i;
	unsigned int dimke;

	dimke = 7;

	ke_ct = malloc(nke * dimke * sizeof(double));
	assert(ke_ct);

	zs_ct = malloc(nke * sizeof(double));
	assert(zs_ct);
	zs_ct_comp = malloc(nke * sizeof(double));
	assert(zs_ct_comp);

	g = 0.1;
	kf = 1.3;
	kmax = 2.5;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ke_sample_zs_ct(ke_ct, nke, st, en, seed);

	for (i = 0; i < nke; i++) {
		zs_ct_comp[i] = get_zs_contact(g, kf, &ke_ct[dimke * i], dimke);
	}

	vpar[0] = g;

	get_zs_num(zs_ct, ke_ct, nke, dimke, kf, nq, nth, nphi, &v_contact, vpar);

	err = 0;
	for (i = 0; i < nke; i++) {
		if (DEBUG) {
			printf("%+.15E %+.15E\n", zs_ct_comp[i], zs_ct[i]);
		}
		err += fabs(zs_ct_comp[i] - zs_ct[i]);
	}

	free(ke_ct);
	free(zs_ct);
	free(zs_ct_comp);

	return err;
}

double test_get_zsp_num(unsigned long nke, unsigned long nq, unsigned long nth, unsigned long nphi, int seed)
{
	double *ke_ct, *kep_ct, *zsp_ct, *zsp_ct_comp, kf, g, st[3], en[3], vpar[1], kmax, err;
	unsigned long i;
	unsigned int dimke;

	dimke = 7;

	ke_ct = malloc(nke * dimke * sizeof(double));
	assert(ke_ct);
	kep_ct = malloc(nke * dimke * sizeof(double));
	assert(kep_ct);

	zsp_ct = malloc(nke * sizeof(double));
	assert(zsp_ct);
	zsp_ct_comp = malloc(nke * sizeof(double));
	assert(zsp_ct_comp);

	g = 0.1;
	kf = 1.3;
	kmax = 2.5;

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ke_sample_zs_ct(ke_ct, nke, st, en, seed);

	get_kep_sample_zsp_ct(kep_ct, ke_ct, nke, dimke);

	for (i = 0; i < nke; i++) {
		zsp_ct_comp[i] = get_zs_contact(g, kf, &kep_ct[dimke * i], dimke);
	}

	vpar[0] = g;

	get_zs_num(zsp_ct, kep_ct, nke, dimke, kf, nq, nth, nphi, &v_contact, vpar);

	err = 0;
	for (i = 0; i < nke; i++) {
		if (DEBUG) {
			printf("%+.15E %+.15E\n", zsp_ct_comp[i], zsp_ct[i]);
		}
		err += fabs(zsp_ct_comp[i] - zsp_ct[i]);
	}

	free(ke_ct);
	free(kep_ct);
	free(zsp_ct);
	free(zsp_ct_comp);

	return err;
}
