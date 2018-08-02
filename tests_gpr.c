#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../lib_quadrature/lib_quadrature.h"
#include "../lib_arr/lib_arr.h"
#include "../lib_rng/lib_rng.h"
#include "../lib_pots/lib_pot.h"
#include "../lib_gpr/lib_gpr.h"
#include "../lib_flow/lib_flow.h"

#define PI (3.1415926535897)
#define DIM (6)

double test_gpr_fit(unsigned long ns, unsigned long nt, double kf, int seed)
{
	double *ke, *ket, *zs, *zsp, *rhs, st6[6], en6[6], *zs_t, *zsp_t, *rhs_t, *rhs_p, p[NP_SE_ARD], *wt,
	    *krxx, *lkrxx, *krpx, *krpp, *var, llhd, ret, fac, *diff;
	int np, i, j, k, nq, nth, nphi, info;

	nq = 10;
	nth = 10;
	nphi = 10;

	np = NP_SE_ARD;

	ke = malloc(ns * DIM * sizeof(double));
	assert(ke);
	ket = malloc(nt * DIM * sizeof(double));
	assert(ket);

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
	en6[0] = 0.5 * kf;
	st6[1] = 0;
	en6[1] = 0.5 * kf;
	st6[2] = 0;
	en6[2] = 0.5 * kf;
	st6[3] = 0;
	en6[3] = PI;
	st6[4] = 0;
	en6[4] = PI;
	st6[5] = 0;
	en6[5] = 2 * PI;

	fill_ext_momenta6(ke, ns, st6, en6, seed);
	fill_ext_momenta6(ket, nt, st6, en6, seed + 1234);

	fac = 0.96;

	ret = zs_flow(zs, ke, ns, DIM, kf, nq, nth, nphi, v_exp, NULL, fac);
	ret = zsp_flow(zsp, ke, ns, DIM, kf, nq, nth, nphi, v_exp, NULL, fac);
	ret = zs_flow(zs_t, ket, nt, DIM, kf, nq, nth, nphi, v_exp, NULL, fac);
	ret = zsp_flow(zsp_t, ket, nt, DIM, kf, nq, nth, nphi, v_exp, NULL, fac);

	for (i = 0; i < np; i++) {
		p[i] = 1.0;
	}

	p[0] = +2.720248840222972E-01;
	p[1] = +2.779819266529766E-01;
	p[2] = +7.480748300074691E-01;
	p[3] = +3.608480347113331E+00;
	p[4] = +7.787434830890305E-01;
	p[5] = +7.570734481190660E-01;

	for (i = 0; i < ns; i++) {
		rhs[i] = zs[i] - zsp[i];
	}

	for (i = 0; i < nt; i++) {
		rhs_t[i] = zs_t[i] - zsp_t[i];
	}

	get_hyper_param_ard(p, np, ke, rhs, ns, DIM);

	print_vec_real(p, np);

	get_krn_se_ard(krxx, ke, ke, ns, ns, DIM, p, np);

	info = get_gpr_weights(wt, lkrxx, krxx, ns, DIM, rhs);
	assert(info == 0);

	get_krn_se_ard(krpx, ket, ke, nt, ns, DIM, p, np);

	gpr_predict(rhs_p, wt, krpx, nt, ns);

	diff = get_diff_vec_real(rhs_t, rhs_p, nt);
	assert(diff);

	for (i = 0; i < nt; i++) {
		printf("%+.15E %+.15E %+.15E\n", rhs_t[i], rhs_p[i], diff[i]);
	}

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
	free(diff);

	return 0;
}
