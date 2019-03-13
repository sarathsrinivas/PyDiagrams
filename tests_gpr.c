#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_quadrature/lib_quadrature.h>
#include <lib_arr/lib_arr.h>
#include <lib_rng/lib_rng.h>
#include <lib_pots/lib_pots.h>
#include <lib_gpr/lib_gpr.h>
#include "lib_flow.h"

#define PI (3.1415926535897)
#define DIM (6)

double test_gpr_fit(unsigned long ns, unsigned long nt, double kf, int seed)
{
	double *ke, *ket, *ke_ct, *ket_ct, *zs, *zsp, *rhs, st6[6], en6[6], *zs_t, *zsp_t, *rhs_t, *rhs_p,
	    p[NP_SE_ARD], *wt, *krxx, *lkrxx, *krpx, *krpp, *var, llhd, ret, fac, *diff, stat[3];
	int np, i, j, k, nq, nth, nphi, info;

	nq = 10;
	nth = 10;
	nphi = 10;

	np = NP_SE_ARD;

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);
	ket = malloc(nt * DIM * sizeof(double));
	assert(ket);

	ke_ct = malloc(ns * DIM * sizeof(double));
	assert(ke_ct);
	ket_ct = malloc(nt * DIM * sizeof(double));
	assert(ket_ct);

	wt = malloc(ns * sizeof(double));
	assert(wt);

	zs = malloc(ns * sizeof(double));
	assert(zs);
	zs_t = malloc(nt * sizeof(double));
	assert(zs_t);

	zsp = malloc(ns * sizeof(double));
	assert(zsp);
	zsp_t = malloc(nt * sizeof(double));
	assert(zsp_t);

	rhs = malloc(ns * sizeof(double));
	assert(rhs);

	rhs_p = malloc(nt * sizeof(double));
	assert(rhs_p);
	rhs_t = malloc(nt * sizeof(double));
	assert(rhs_t);

	krxx = malloc(ns * ns * sizeof(double));
	assert(krxx);

	lkrxx = malloc(ns * ns * sizeof(double));
	assert(lkrxx);

	krpx = malloc(nt * ns * sizeof(double));
	assert(krpx);

	krpp = malloc(nt * nt * sizeof(double));
	assert(krpp);

	var = malloc(nt * nt * sizeof(double));
	assert(var);

	st6[0] = 0;
	en6[0] = 3 * kf;
	st6[1] = 0;
	en6[1] = 3 * kf;
	st6[2] = 0;
	en6[2] = 3 * kf;
	st6[3] = 0;
	en6[3] = PI;
	st6[4] = 0;
	en6[4] = PI;
	st6[5] = 0;
	en6[5] = 2 * PI;

	fill_ext_momenta_3ball(ke, ns, st6, en6, seed);
	fill_ext_momenta_3ball(ket, nt, st6, en6, seed + 1234);

	fac = 0.96;

	zs_flow(zs, ke, ns, DIM, kf, nq, nth, nphi, &v_exp, NULL, fac);
	zsp_flow(zsp, ke, ns, DIM, kf, nq, nth, nphi, v_exp, NULL, fac);
	zs_flow(zs_t, ket, nt, DIM, kf, nq, nth, nphi, &v_exp, NULL, fac);
	zsp_flow(zsp_t, ket, nt, DIM, kf, nq, nth, nphi, v_exp, NULL, fac);

	for (i = 0; i < np; i++) {
		p[i] = 1.0;
	}

	for (i = 0; i < ns; i++) {
		rhs[i] = zs[i] - zsp[i];
	}

	for (i = 0; i < nt; i++) {
		rhs_t[i] = zs_t[i] - zsp_t[i];
	}

	sph_ct_mom6_zs(ke_ct, ke, DIM, ns);
	sph_ct_mom6_zs(ket_ct, ket, DIM, nt);

	get_hyper_param_ard(p, np, ke, rhs, ns, DIM);

	print_vec_real(p, np);

	get_krn_se_ard(krxx, ke, ke, ns, ns, DIM, p, np);

	get_gpr_weights(wt, lkrxx, krxx, ns, DIM, rhs);

	get_krn_se_ard(krpx, ket, ke, nt, ns, DIM, p, np);

	gpr_predict(rhs_p, wt, krpx, nt, ns);

	diff = get_rel_error(rhs_t, rhs_p, nt);
	assert(diff);

	get_stat(diff, nt, stat, 3);

	for (i = 0; i < nt; i++) {
		printf("%+.15E %+.15E %+.15E\n", rhs_t[i], rhs_p[i], diff[i]);
	}

	printf("\n%+.15E : MEAN\n%+.15E : SD\n%+.15E : MAX_FABS\n", stat[0], stat[1], stat[2]);

	free(diff);
	free(zs);
	free(zs_t);
	free(zsp);
	free(zsp_t);
	free(rhs_p);
	free(rhs_t);
	free(rhs);
	free(var);
	free(krpp);
	free(krpx);
	free(lkrxx);
	free(krxx);
	free(wt);
	free(ket);
	free(ke);
	free(ket_ct);
	free(ke_ct);

	return 0;
}
