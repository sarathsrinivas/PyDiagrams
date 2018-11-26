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

double test_ph_phase_space_vol(double dl, double kf, double fac_th_brk, unsigned long nq, unsigned long nth,
			       unsigned long nphi)
{
	double *gl_q, *wl_q, *gl_th, *wl_th, *gl_phi, *wl_phi, *qvec, *wt, vol, vol_full, vol_int, vol_exct,
	    q, th, phi, d, *work;
	unsigned long nl, i;

	gl_q = malloc(nq * sizeof(double));
	assert(gl_q);
	wl_q = malloc(nq * sizeof(double));
	assert(wl_q);

	gl_th = malloc(nth * sizeof(double));
	assert(gl_th);
	wl_th = malloc(nth * sizeof(double));
	assert(wl_th);

	gl_phi = malloc(nphi * sizeof(double));
	assert(gl_phi);
	wl_phi = malloc(nphi * sizeof(double));
	assert(wl_phi);

	nl = nq * nth * nphi;

	work = malloc(2 * nl * sizeof(double));
	assert(work);

	qvec = malloc(3 * nl * sizeof(double));
	assert(qvec);
	wt = malloc(nl * sizeof(double));
	assert(wt);

	gauss_grid_create(nq, gl_q, wl_q, -1, 1);
	gauss_grid_create(nth / 4, gl_th, wl_th, -1, 1);
	gauss_grid_create(nphi, gl_phi, wl_phi, -1, 1);

	get_ph_phase_space(kf, dl, nq, nth, nphi, fac_th_brk, gl_q, wl_q, gl_th, wl_th, gl_phi, wl_phi, qvec,
			   wt, work, 2 * nl);

	vol = 0;
	for (i = 0; i < nl; i++) {
		q = qvec[3 * i + 0];
		th = qvec[3 * i + 1];
		phi = qvec[3 * i + 2];

		vol += wt[i];
	}

	vol_full = 4 * (PI / 3) * kf * kf * kf;
	d = 2 * dl;

	if (dl < kf) {
		vol_int = (1.0 / 12.0) * PI * (4 * kf + d) * (2 * kf - d) * (2 * kf - d);
	} else {
		vol_int = 0;
	}

	vol_exct = 2 * (vol_full - vol_int);

	printf("%+.15E %+.15E\n", vol_exct, vol);

	free(wl_phi);
	free(gl_phi);
	free(wl_th);
	free(gl_th);
	free(wl_q);
	free(gl_q);
	free(qvec);
	free(wt);
	free(work);

	return fabs(vol_exct - vol);
}

double test_ph_phase_space(double dl, double kf, double fac, unsigned long nq, unsigned long nth,
			   unsigned long nphi)
{

	double *gq1, *wq1, *wq, *q, *phi, *wphi, *th, *wth, dl2, Is, Ith, Iq, sgn, th_max, vol_full, vol_cres,
	    vol_int, I_exact, a, b, q0, q1, cos_th, sin_th, pf_q, q2, d, q_min, q_max;
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
		gauss_grid_create(2 * nth, &th[2 * nth], &wth[2 * nth], 0.5 * PI, PI);
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

			q_min = (q0 < q1) ? q0 : q1;
			q_max = (q0 > q1) ? q0 : q1;

			gauss_grid_rescale(gq1, wq1, nq, q, wq, q_min, q_max);

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

double test_zs_contact(int isdl_kf, double kf, unsigned long ns, unsigned long nq, unsigned long nth,
		       unsigned long nphi, int seed)
{
	double *ke, st[6], en[6], *zs, zs_exact, *zs_diff, max_fabs, fac, g, norm;
	unsigned long i;
	int ret;

	fprintf(stderr, "test_zs_contact(dl>kf=%d) %s:%d\n", isdl_kf, __FILE__, __LINE__);

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);

	zs = malloc(ns * sizeof(double));
	assert(zs);

	zs_diff = malloc(ns * sizeof(double));
	assert(zs_diff);

	st[0] = 0;
	if (isdl_kf == 0) {
		en[0] = kf;
	} else {
		en[0] = 3 * kf;
	}
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
	norm = get_norm_real(zs_diff, ns);
	norm /= ns;

	free(zs);
	free(zs_diff);
	free(ke);

	return 0.5 * (max_fabs + norm);
}

double test_get_zs_loop_mom(unsigned long ns, double kf, int seed)
{
	double *ke, st[6], en[6], dl, dlp, P, P_dl, P_dlp, dl_dlp, phi_dlp, q, th_q, phi_q, q_ct[3], dl_ct[3],
	    dlp_ct[3], P_ct[3], kl1[6], kl2[6], dl_zs1_ct[3], dlp_zs1_ct[3], P_zs1_ct[3], dl_zs2_ct[3],
	    dlp_zs2_ct[3], P_zs2_ct[3], dl_zs1_cmp, dlp_zs1_cmp, P_zs1_cmp, dl_zs2_cmp, dlp_zs2_cmp,
	    P_zs2_cmp, P_dl_zs1_cmp, P_dlp_zs1_cmp, dl_dlp_zs1_cmp, P_dl_zs2_cmp, P_dlp_zs2_cmp,
	    dl_dlp_zs2_cmp, diff;
	unsigned long i, k;
	dsfmt_t drng;

	fprintf(stderr, "test_get_zs_loop_mom() %s:%d\n", __FILE__, __LINE__);

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);

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

	fill_ext_momenta_3ball(ke, ns, st, en, seed);
	dsfmt_init_gen_rand(&drng, seed + 7);

	for (k = 0; k < 100; k++) {

		q = dsfmt_genrand_close_open(&drng);
		th_q = PI * dsfmt_genrand_close_open(&drng);
		phi_q = 2 * PI * dsfmt_genrand_close_open(&drng);

		q_ct[0] = q * cos(phi_q) * sin(th_q);
		q_ct[1] = q * sin(phi_q) * sin(th_q);
		q_ct[2] = q * cos(th_q);

		diff = 0;

		for (i = 0; i < ns; i++) {

			dl = ke[i * DIM + 0];
			dlp = ke[i * DIM + 1];
			P = ke[i * DIM + 2];
			dl_dlp = ke[i * DIM + 3];
			P_dl = ke[i * DIM + 4];
			P_dlp = ke[i * DIM + 5];

			phi_dlp = acos((cos(P_dlp) - cos(P_dl) * cos(dl_dlp)) / (sin(P_dl) * sin(dl_dlp)));

			get_zs_loop_mom(kl1, kl2, DIM, &ke[i * DIM], phi_dlp, q, th_q, phi_q);

			printf("%+.15E %+.15E %+.15E %+.15E %+.15E %+.15E\n", kl1[3], kl1[4], kl1[5], kl2[3],
			       kl2[4], kl2[5]);

			dl_ct[0] = 0;
			dl_ct[1] = 0;
			dl_ct[2] = dl;

			dlp_ct[0] = dlp * cos(phi_dlp) * sin(dl_dlp);
			dlp_ct[1] = dlp * sin(phi_dlp) * sin(dl_dlp);
			dlp_ct[2] = dlp * cos(dl_dlp);

			P_ct[0] = P * cos(0) * sin(P_dl);
			P_ct[1] = P * sin(0) * sin(P_dl);
			P_ct[2] = P * cos(P_dl);

			dlp_zs1_ct[0] = 0.5 * (P_ct[0] - q_ct[0] + dlp_ct[0]);
			dlp_zs1_ct[1] = 0.5 * (P_ct[1] - q_ct[1] + dlp_ct[1]);
			dlp_zs1_ct[2] = 0.5 * (P_ct[2] - q_ct[2] + dlp_ct[2]);

			P_zs1_ct[0] = P_ct[0] + q_ct[0] + dlp_ct[0];
			P_zs1_ct[1] = P_ct[1] + q_ct[1] + dlp_ct[1];
			P_zs1_ct[2] = P_ct[2] + q_ct[2] + dlp_ct[2];

			dlp_zs2_ct[0] = 0.5 * (-P_ct[0] + q_ct[0] + dlp_ct[0]);
			dlp_zs2_ct[1] = 0.5 * (-P_ct[1] + q_ct[1] + dlp_ct[1]);
			dlp_zs2_ct[2] = 0.5 * (-P_ct[2] + q_ct[2] + dlp_ct[2]);

			P_zs2_ct[0] = P_ct[0] + q_ct[0] - dlp_ct[0];
			P_zs2_ct[1] = P_ct[1] + q_ct[1] - dlp_ct[1];
			P_zs2_ct[2] = P_ct[2] + q_ct[2] - dlp_ct[2];

			dl_zs1_cmp = dl_ct[2];
			dlp_zs1_cmp = sqrt(dlp_zs1_ct[0] * dlp_zs1_ct[0] + dlp_zs1_ct[1] * dlp_zs1_ct[1]
					   + dlp_zs1_ct[2] * dlp_zs1_ct[2]);
			P_zs1_cmp = sqrt(P_zs1_ct[0] * P_zs1_ct[0] + P_zs1_ct[1] * P_zs1_ct[1]
					 + P_zs1_ct[2] * P_zs1_ct[2]);

			dl_zs2_cmp = dl_ct[2];
			dlp_zs2_cmp = sqrt(dlp_zs2_ct[0] * dlp_zs2_ct[0] + dlp_zs2_ct[1] * dlp_zs2_ct[1]
					   + dlp_zs2_ct[2] * dlp_zs2_ct[2]);
			P_zs2_cmp = sqrt(P_zs2_ct[0] * P_zs2_ct[0] + P_zs2_ct[1] * P_zs2_ct[1]
					 + P_zs2_ct[2] * P_zs2_ct[2]);

			dl_dlp_zs1_cmp = acos(
			    (dl_ct[0] * dlp_zs1_ct[0] + dl_ct[1] * dlp_zs1_ct[1] + dl_ct[2] * dlp_zs1_ct[2])
			    / (dl * dlp_zs1_cmp));

			P_dl_zs1_cmp
			    = acos((dl_ct[0] * P_zs1_ct[0] + dl_ct[1] * P_zs1_ct[1] + dl_ct[2] * P_zs1_ct[2])
				   / (dl * P_zs1_cmp));

			P_dlp_zs1_cmp = acos((dlp_zs1_ct[0] * P_zs1_ct[0] + dlp_zs1_ct[1] * P_zs1_ct[1]
					      + dlp_zs1_ct[2] * P_zs1_ct[2])
					     / (dlp_zs1_cmp * P_zs1_cmp));

			dl_dlp_zs2_cmp = acos(
			    (dl_ct[0] * dlp_zs2_ct[0] + dl_ct[1] * dlp_zs2_ct[1] + dl_ct[2] * dlp_zs2_ct[2])
			    / (dl * dlp_zs2_cmp));

			P_dl_zs2_cmp
			    = acos((dl_ct[0] * P_zs2_ct[0] + dl_ct[1] * P_zs2_ct[1] + dl_ct[2] * P_zs2_ct[2])
				   / (dl * P_zs2_cmp));

			P_dlp_zs2_cmp = acos((dlp_zs2_ct[0] * P_zs2_ct[0] + dlp_zs2_ct[1] * P_zs2_ct[1]
					      + dlp_zs2_ct[2] * P_zs2_ct[2])
					     / (dlp_zs2_cmp * P_zs2_cmp));

			diff += fabs(dl_zs1_cmp - kl1[0]);
			diff += fabs(dlp_zs1_cmp - kl1[1]);
			diff += fabs(P_zs1_cmp - kl1[2]);

			diff += fabs(dl_zs2_cmp - kl2[0]);
			diff += fabs(dlp_zs2_cmp - kl2[1]);
			diff += fabs(P_zs2_cmp - kl2[2]);

			diff += fabs(dl_dlp_zs1_cmp - kl1[3]);
			diff += fabs(P_dl_zs1_cmp - kl1[4]);
			diff += fabs(P_dlp_zs1_cmp - kl1[5]);

			diff += fabs(dl_dlp_zs2_cmp - kl2[3]);
			diff += fabs(P_dl_zs2_cmp - kl2[4]);
			diff += fabs(P_dlp_zs2_cmp - kl2[5]);
		}
	}

	free(ke);

	return diff;
}

double test_zsp_contact(int isdlp_kf, double kf, unsigned long ns, unsigned long nq, unsigned long nth,
			unsigned long nphi, int seed)
{
	double *ke, st[6], en[6], *zsp, zsp_exact, *zsp_diff, max_fabs, fac, g, norm;
	unsigned long i;
	int ret;

	fprintf(stderr, "test_zsp_contact(dlp>kf=%d) %s:%d\n", isdlp_kf, __FILE__, __LINE__);

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);

	zsp = malloc(ns * sizeof(double));
	assert(zsp);

	zsp_diff = malloc(ns * sizeof(double));
	assert(zsp_diff);

	st[0] = 0;
	en[0] = 3 * kf;
	st[1] = 0;
	if (isdlp_kf == 0) {
		en[1] = kf;
	} else {
		en[1] = 3 * kf;
	}
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

	ret = zsp_flow(zsp, ke, ns, DIM, kf, nq, nth, nphi, v_contact, NULL, fac);

	g = 0.5;

	for (i = 0; i < ns; i++) {
		zsp_exact = zsp_contact(&ke[DIM * i], DIM, kf, g);
		zsp_diff[i] = fabs(zsp_exact - zsp[i]);
	}

	max_fabs = get_max_fabs(zsp_diff, ns);
	norm = get_norm_real(zsp_diff, ns);
	norm /= ns;

	free(zsp);
	free(zsp_diff);
	free(ke);

	return 0.5 * (max_fabs + norm);
}

double test_get_zsp_loop_mom(unsigned long ns, double kf, int seed)
{
	double *ke, st[6], en[6], dl, dlp, P, P_dl, P_dlp, dl_dlp, phi_dl, q, th_q, phi_q, q_ct[3], dl_ct[3],
	    dlp_ct[3], P_ct[3], kl1[6], kl2[6], dl_zs1_ct[3], dlp_zs1_ct[3], P_zs1_ct[3], dl_zs2_ct[3],
	    dlp_zs2_ct[3], P_zs2_ct[3], dl_zs1_cmp, dlp_zs1_cmp, P_zs1_cmp, dl_zs2_cmp, dlp_zs2_cmp,
	    P_zs2_cmp, P_dl_zs1_cmp, P_dlp_zs1_cmp, dl_dlp_zs1_cmp, P_dl_zs2_cmp, P_dlp_zs2_cmp,
	    dl_dlp_zs2_cmp, diff;
	unsigned long i;
	dsfmt_t drng;

	fprintf(stderr, "test_get_zsp_loop_mom() %s:%d\n", __FILE__, __LINE__);

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);

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

	dsfmt_init_gen_rand(&drng, seed + 7);

	q = dsfmt_genrand_close_open(&drng);
	th_q = PI * dsfmt_genrand_close_open(&drng);
	phi_q = 2 * PI * dsfmt_genrand_close_open(&drng);

	q_ct[0] = q * cos(phi_q) * sin(th_q);
	q_ct[1] = q * sin(phi_q) * sin(th_q);
	q_ct[2] = q * cos(th_q);

	diff = 0;

	for (i = 0; i < ns; i++) {

		dl = ke[i * DIM + 0];
		dlp = ke[i * DIM + 1];
		P = ke[i * DIM + 2];
		dl_dlp = ke[i * DIM + 3];
		P_dl = ke[i * DIM + 4];
		P_dlp = ke[i * DIM + 5];

		phi_dl = acos((cos(P_dl) - cos(P_dlp) * cos(dl_dlp)) / (sin(P_dlp) * sin(dl_dlp)));

		get_zsp_loop_mom(kl1, kl2, DIM, &ke[i * DIM], phi_dl, q, th_q, phi_q);

		dl_ct[0] = dl * cos(phi_dl) * sin(dl_dlp);
		dl_ct[1] = dl * sin(phi_dl) * sin(dl_dlp);
		dl_ct[2] = dl * cos(dl_dlp);

		dlp_ct[0] = 0;
		dlp_ct[1] = 0;
		dlp_ct[2] = dlp;

		P_ct[0] = P * cos(0) * sin(P_dlp);
		P_ct[1] = P * sin(0) * sin(P_dlp);
		P_ct[2] = P * cos(P_dlp);

		dl_zs1_ct[0] = 0.5 * (-P_ct[0] + q_ct[0] + dl_ct[0]);
		dl_zs1_ct[1] = 0.5 * (-P_ct[1] + q_ct[1] + dl_ct[1]);
		dl_zs1_ct[2] = 0.5 * (-P_ct[2] + q_ct[2] + dl_ct[2]);

		P_zs1_ct[0] = P_ct[0] + q_ct[0] - dl_ct[0];
		P_zs1_ct[1] = P_ct[1] + q_ct[1] - dl_ct[1];
		P_zs1_ct[2] = P_ct[2] + q_ct[2] - dl_ct[2];

		dl_zs2_ct[0] = 0.5 * (P_ct[0] - q_ct[0] + dl_ct[0]);
		dl_zs2_ct[1] = 0.5 * (P_ct[1] - q_ct[1] + dl_ct[1]);
		dl_zs2_ct[2] = 0.5 * (P_ct[2] - q_ct[2] + dl_ct[2]);

		P_zs2_ct[0] = P_ct[0] + q_ct[0] + dl_ct[0];
		P_zs2_ct[1] = P_ct[1] + q_ct[1] + dl_ct[1];
		P_zs2_ct[2] = P_ct[2] + q_ct[2] + dl_ct[2];

		dl_zs1_cmp = sqrt(dl_zs1_ct[0] * dl_zs1_ct[0] + dl_zs1_ct[1] * dl_zs1_ct[1]
				  + dl_zs1_ct[2] * dl_zs1_ct[2]);
		dlp_zs1_cmp = dlp_ct[2];
		P_zs1_cmp
		    = sqrt(P_zs1_ct[0] * P_zs1_ct[0] + P_zs1_ct[1] * P_zs1_ct[1] + P_zs1_ct[2] * P_zs1_ct[2]);

		dl_zs2_cmp = sqrt(dl_zs2_ct[0] * dl_zs2_ct[0] + dl_zs2_ct[1] * dl_zs2_ct[1]
				  + dl_zs2_ct[2] * dl_zs2_ct[2]);
		dlp_zs2_cmp = dlp_ct[2];
		P_zs2_cmp
		    = sqrt(P_zs2_ct[0] * P_zs2_ct[0] + P_zs2_ct[1] * P_zs2_ct[1] + P_zs2_ct[2] * P_zs2_ct[2]);

		dl_dlp_zs1_cmp
		    = acos((dlp_ct[0] * dl_zs1_ct[0] + dlp_ct[1] * dl_zs1_ct[1] + dlp_ct[2] * dl_zs1_ct[2])
			   / (dlp * dl_zs1_cmp));

		P_dlp_zs1_cmp
		    = acos((dlp_ct[0] * P_zs1_ct[0] + dlp_ct[1] * P_zs1_ct[1] + dlp_ct[2] * P_zs1_ct[2])
			   / (dlp * P_zs1_cmp));

		P_dl_zs1_cmp = acos(
		    (dl_zs1_ct[0] * P_zs1_ct[0] + dl_zs1_ct[1] * P_zs1_ct[1] + dl_zs1_ct[2] * P_zs1_ct[2])
		    / (dl_zs1_cmp * P_zs1_cmp));

		dl_dlp_zs2_cmp
		    = acos((dlp_ct[0] * dl_zs2_ct[0] + dlp_ct[1] * dl_zs2_ct[1] + dlp_ct[2] * dl_zs2_ct[2])
			   / (dlp * dl_zs2_cmp));

		P_dlp_zs2_cmp
		    = acos((dlp_ct[0] * P_zs2_ct[0] + dlp_ct[1] * P_zs2_ct[1] + dlp_ct[2] * P_zs2_ct[2])
			   / (dlp * P_zs2_cmp));

		P_dl_zs2_cmp = acos(
		    (dl_zs2_ct[0] * P_zs2_ct[0] + dl_zs2_ct[1] * P_zs2_ct[1] + dl_zs2_ct[2] * P_zs2_ct[2])
		    / (dl_zs2_cmp * P_zs2_cmp));

		diff += fabs(dl_zs1_cmp - kl1[0]);
		diff += fabs(dlp_zs1_cmp - kl1[1]);
		diff += fabs(P_zs1_cmp - kl1[2]);

		diff += fabs(dl_zs2_cmp - kl2[0]);
		diff += fabs(dlp_zs2_cmp - kl2[1]);
		diff += fabs(P_zs2_cmp - kl2[2]);

		diff += fabs(dl_dlp_zs1_cmp - kl1[3]);
		diff += fabs(P_dl_zs1_cmp - kl1[4]);
		diff += fabs(P_dlp_zs1_cmp - kl1[5]);

		diff += fabs(dl_dlp_zs2_cmp - kl2[3]);
		diff += fabs(P_dl_zs2_cmp - kl2[4]);
		diff += fabs(P_dlp_zs2_cmp - kl2[5]);
	}

	free(ke);

	return diff;
}

double test_antisymmetry(double kf, unsigned long ns, unsigned long nq, unsigned long nth, unsigned long nphi,
			 int seed)
{
	double *ke, st[6], en[6], *zs, *zsp, *rhs_dir, *rhs_exch, *diff, max_fabs, fac, g, tmp, norm;
	unsigned long i;
	int ret;

	fprintf(stderr, "test_antisymmetry() %s:%d\n", __FILE__, __LINE__);

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);

	zs = malloc(ns * sizeof(double));
	assert(zs);

	zsp = malloc(ns * sizeof(double));
	assert(zsp);

	rhs_dir = malloc(ns * sizeof(double));
	assert(rhs_dir);

	rhs_exch = malloc(ns * sizeof(double));
	assert(rhs_exch);

	diff = malloc(ns * sizeof(double));
	assert(diff);

	st[0] = 0;
	en[0] = kf;
	st[1] = 0;
	en[1] = kf;
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

	ret = zs_flow(zs, ke, ns, DIM, kf, nq, nth, nphi, v_yukawa, NULL, fac);
	ret = zsp_flow(zsp, ke, ns, DIM, kf, nq, nth, nphi, v_yukawa, NULL, fac);

	for (i = 0; i < ns; i++) {
		rhs_dir[i] = zs[i] - zsp[i];
	}

	for (i = 0; i < ns; i++) {
		/* dl <-> dlp */
		tmp = ke[DIM * i + 0];
		ke[DIM * i + 0] = ke[DIM * i + 1];
		ke[DIM * i + 1] = tmp;

		/* P_dl <-> P_dlp */
		tmp = ke[DIM * i + 4];
		ke[DIM * i + 4] = ke[DIM * i + 5];
		ke[DIM * i + 5] = tmp;
	}

	ret = zs_flow(zs, ke, ns, DIM, kf, nq, nth, nphi, v_yukawa, NULL, fac);
	ret = zsp_flow(zsp, ke, ns, DIM, kf, nq, nth, nphi, v_yukawa, NULL, fac);

	for (i = 0; i < ns; i++) {
		rhs_exch[i] = zs[i] - zsp[i];
	}

	for (i = 0; i < ns; i++) {
		diff[i] = rhs_dir[i] + rhs_exch[i];
	}

	max_fabs = get_max_fabs(diff, ns);
	norm = get_norm_real(diff, ns);

	norm /= ns;

	free(zsp);
	free(zs);
	free(rhs_dir);
	free(rhs_exch);
	free(diff);
	free(ke);

	return 0.5 * (norm + max_fabs);
}

double test_convergence(unsigned long ns, double kf, int seed)
{

	double *ke, st[6], en[6], *zs, *zsp, *zs1, *zsp1, rhs, rhs1, *diff, max_fabs, fac, g, tmp, norm;
	unsigned long i, nq[2], nth[2], nphi[2];
	int ret;

	fprintf(stderr, "test_convergence() %s:%d\n", __FILE__, __LINE__);

	nq[0] = 10;
	nth[0] = 10;
	nphi[0] = 15;

	nq[1] = 2 * nq[0];
	nth[1] = 2 * nth[0];
	nphi[1] = 2 * nphi[0];

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);

	zs = malloc(ns * sizeof(double));
	assert(zs);

	zs1 = malloc(ns * sizeof(double));
	assert(zs1);

	zsp = malloc(ns * sizeof(double));
	assert(zsp);

	zsp1 = malloc(ns * sizeof(double));
	assert(zsp1);

	diff = malloc(ns * sizeof(double));
	assert(diff);

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

	ret = zs_flow(zs, ke, ns, DIM, kf, nq[0], nth[0], nphi[0], v_exp, NULL, fac);
	ret = zsp_flow(zsp, ke, ns, DIM, kf, nq[0], nth[0], nphi[0], v_exp, NULL, fac);

	ret = zs_flow(zs1, ke, ns, DIM, kf, nq[1], nth[1], nphi[1], v_exp, NULL, fac);
	ret = zsp_flow(zsp1, ke, ns, DIM, kf, nq[1], nth[1], nphi[1], v_exp, NULL, fac);

	for (i = 0; i < ns; i++) {
		rhs = zs[i] - zsp[i];
		rhs1 = zs1[i] - zsp1[i];
		diff[i] = fabs(rhs - rhs1);
	}

	max_fabs = get_max_fabs(diff, ns);
	norm = get_norm_real(diff, ns);

	norm /= ns;

	free(diff);
	free(zsp1);
	free(zsp);
	free(zs1);
	free(zs);
	free(ke);

	return norm;
}

double test_mom_closure(double kmax, unsigned long ns, int seed)
{
	double *ke, st6[3], en6[3], dl, dlp, P, dl_dlp, P_dl, P_dlp, phi_dlp, q, th_q, phi_q, a, b, q0, q1,
	    kf, sgn, lims[2], limsgn, kl1[DIM], kl2[DIM], th_max, Q1, Q0;
	unsigned long i, j;
	dsfmt_t drng;

	ke = malloc(DIM * ns * sizeof(double));
	assert(ke);

	kf = 1.0;

	st6[0] = 0;
	en6[0] = kmax;
	st6[1] = 0;
	en6[1] = kmax;
	st6[2] = 0;
	en6[2] = kmax;

	fill_ext_momenta_3ball(ke, ns, st6, en6, seed);

	dsfmt_init_gen_rand(&drng, seed + 213);

	for (i = 0; i < ns; i++) {

		dl = ke[DIM * i + 0];
		dlp = ke[DIM * i + 1];
		P = ke[DIM * i + 2];
		dl_dlp = ke[DIM * i + 3];
		P_dl = ke[DIM * i + 4];
		P_dlp = ke[DIM * i + 5];

		phi_dlp = acos((cos(P_dlp) - cos(P_dl) * cos(dl_dlp)) / (sin(P_dl) * sin(dl_dlp)));

		for (j = 0; j < 1000; j++) {

			phi_q = 2 * PI * dsfmt_genrand_close_open(&drng);

			if (dl < kf) {
				th_q = PI * dsfmt_genrand_close_open(&drng);
				sgn = 1.0;
			} else {
				th_max = asin(kf / dl);
				th_q = (PI - th_max) + th_max * dsfmt_genrand_close_open(&drng);
				sgn = -1.0;

				a = dl * cos(th_q);
				b = sqrt(kf * kf - dl * dl * sin(th_q) * sin(th_q));

				q0 = sgn * (-a + b);
				q1 = a + b;

				get_zs_reg_limits(kmax, &ke[DIM * i], phi_dlp, th_q, phi_q, lims);

				Q0 = q0;
				Q1 = (q1 < lims[1]) ? q1 : lims[1];

				limsgn = Q1 - Q0;
				q = Q0 + (Q1 - Q0) * dsfmt_genrand_close_open(&drng);

				get_zs_loop_mom_ct(kl1, kl2, DIM, &ke[DIM * i], phi_dlp, q, th_q, phi_q);

				printf("%+.15E %+.15E %+.15E %+.15E\n", kl1[1], kl1[2], kl2[1], kl2[2]);
			}
		}
	}

	free(ke);

	return 0;
}
