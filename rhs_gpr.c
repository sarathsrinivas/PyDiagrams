#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include "lib_flow.h"

void flow_rhs(double *gma, double *var_gma, double *gma0, double *var_gma0, unsigned long nke, void *param)
{

	double *lknxx_gma, *kxx_gma, *wt_gma, *var_gma12, *ke_ct, *q_ct, *pke_ct, *wt_fq, *var_fq, *A1, *B1,
	    *C, *A2, *B2, *lknxx_fq, *kxx_fq, *Iqe, *q_sph, *pq_sph, fac, kf, *IIe;
	unsigned long nq, npke, nth, i;
	unsigned int dimq, dimke, ke_flag;
	struct rhs_param *par = param;

	ke_ct = par->ke_ct;
	q_ct = par->q_ct;
	q_sph = par->q_sph;

	kxx_gma = par->kxx_gma;
	kxx_fq = par->kxx_fq;

	Iqe = par->Iqe;
	IIe = par->IIe;

	pke_ct = par->pke_ct;
	pq_sph = par->pq_sph;

	A1 = par->A1;
	B1 = par->B1;
	A2 = par->A2;
	B2 = par->B2;
	C = par->C;

	fac = par->fac;
	kf = par->kf;

	nth = par->nth;
	nq = par->nq;
	dimq = par->dimq;
	dimke = par->dimke;
	ke_flag = par->ke_flag;

	var_fq = par->var_fq;
	var_gma12 = par->var_gma12;

	lknxx_gma = malloc(nke * nke * sizeof(double));
	assert(lknxx_gma);
	lknxx_fq = malloc(nq * nq * sizeof(double));
	assert(lknxx_fq);

	wt_gma = malloc(nke * sizeof(double));
	assert(wt_gma);
	wt_fq = malloc(nke * nq * sizeof(double));
	assert(wt_fq);

	npke = dimke + 1;

	get_noise_covar_chd(lknxx_gma, kxx_gma, var_gma0, nke);

	get_gma_weight(wt_gma, lknxx_gma, gma0, nke);

	get_var_gma(var_gma12, lknxx_gma, ke_ct, dimke, nke, q_ct, dimq, nq, pke_ct, npke, ke_flag);

	get_fq_samples(wt_fq, var_fq, wt_gma, A1, B1, A2, B2, C, var_gma12, ke_flag, nq, nke);

	get_noise_covar_chd(lknxx_fq, kxx_fq, var_fq, nq);

	get_fq_weights(wt_fq, lknxx_fq, nq, nke);

	get_gma_gpr_mean(gma, Iqe, wt_fq, nke, nq);

	get_gma_gpr_var(var_gma, IIe, Iqe, lknxx_fq, nq, nke);

	free(wt_fq);
	free(wt_gma);
	free(lknxx_fq);
	free(lknxx_gma);
}
