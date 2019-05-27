#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include "lib_flow.h"

void get_rhs_block(double *gma, double *var_gma, const double *gma0, const double *var_gma0,
		   unsigned long nke, struct rhs_param *par)
{
	double *lknxx_gma, *kxx_gma, *wt_gma, *var_gma12, *ke_ct, *q_ct, *pke_ct, *wt_fq, *var_fq,
	    *A1, *B1, *C, *A2, *B2, *lknxx_fq, *kxx_fq, *Iqe, *q_sph, *pq_sph, fac, kf, *IIe, *fqe,
	    *ktt12, *ktx12, *kl12_ct;
	unsigned long nq, nth, i;
	unsigned int dimq, dimke, ke_flag;

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

	ktt12 = par->ktt12;
	ktx12 = par->ktx12;
	kl12_ct = par->kl12_ct;

	fqe = par->fqe;
	var_fq = par->var_fq;
	var_gma12 = par->var_gma12;

	lknxx_fq = malloc(nq * nq * sizeof(double));
	assert(lknxx_fq);
	lknxx_gma = malloc(nke * nke * sizeof(double));
	assert(lknxx_gma);

	wt_gma = malloc(nke * sizeof(double));
	assert(wt_gma);
	wt_fq = malloc(nke * nq * sizeof(double));
	assert(wt_fq);

	get_noise_covar_chd(lknxx_gma, kxx_gma, var_gma0, nke);

	get_gma_weight(wt_gma, lknxx_gma, gma0, nke);

	get_var_mat_chd(var_gma12, ktt12, ktx12, lknxx_gma, 2 * nq, nke);

	get_fq_samples(fqe, var_fq, wt_gma, A1, B1, A2, B2, C, var_gma12, ke_flag, nq, nke);

	get_noise_covar_chd(lknxx_fq, kxx_fq, var_fq, nq);

	get_fq_weights_2(wt_fq, lknxx_fq, fqe, nq, nke);

	get_gma_gpr_mean(gma, Iqe, wt_fq, nke, nq);

	get_gma_gpr_var(var_gma, IIe, Iqe, lknxx_fq, nq, nke);

	free(wt_fq);
	free(wt_gma);
	free(lknxx_fq);
	free(lknxx_gma);
}

void flow_rhs(double *gma, double *var_gma, double *gma0, double *var_gma0, unsigned long nke,
	      void *param)
{

	double *lknxx_gma, *kxx_gma, *wt_gma, *var_gma12, *ke_ct, *q_ct, *pke_ct, *wt_fq, *var_fq,
	    *A1, *B1, *C, *A2, *B2, *lknxx_fq, *kxx_fq, *Iqe, *q_sph, *pq_sph, fac, kf, *IIe, *fqe,
	    *ktt12, *ktx12, *kl12_ct;
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

	ktt12 = par->ktt12;
	ktx12 = par->ktx12;
	kl12_ct = par->kl12_ct;

	fqe = par->fqe;
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

	get_var_mat_chd(var_gma12, ktt12, ktx12, lknxx_gma, 2 * nq, nke);

	get_fq_samples(fqe, var_fq, wt_gma, A1, B1, A2, B2, C, var_gma12, ke_flag, nq, nke);

	get_noise_covar_chd(lknxx_fq, kxx_fq, var_fq, nq);

	get_fq_weights_2(wt_fq, lknxx_fq, fqe, nq, nke);

	get_gma_gpr_mean(gma, Iqe, wt_fq, nke, nq);

	get_gma_gpr_var(var_gma, IIe, Iqe, lknxx_fq, nq, nke);

	free(wt_fq);
	free(wt_gma);
	free(lknxx_fq);
	free(lknxx_gma);
}

void flow_rhs_ph(double *gma, double *var_gma, double *gma0, double *var_gma0, unsigned long nke,
		 void *param)
{

	double *lknxx_gma, *kxx_gma, *wt_gma, *var_gma12, *ke_ct, *q_ct, *pke_ct, *wt_fq, *var_fq,
	    *A1, *B1, *C, *A2, *B2, *lknxx_fq, *kxx_fq, *Iqe, *q_sph, *pq_sph, fac, kf, *IIe, *fqe,
	    *ktt12, *ktx12, *kl12_ct, *kep_ct, *kxxp_gma, *Iqep, *IIep, *lknxx_fqp, *zs, *zsp,
	    *var_zs, *var_zsp, *wt_fqp, *var_fqp, *A1p, *A2p, *B1p, *B2p, *Cp, *ktt12p, *ktx12p,
	    *kl12p_ct, *fqep, *var_gma12p;
	unsigned long nq, npke, nth, i;
	unsigned int dimq, dimke, ke_flag;
	struct rhs_ph_param *par = param;

	ke_ct = par->ke_ct;
	kep_ct = par->kep_ct;
	q_ct = par->q_ct;
	q_sph = par->q_sph;

	kxx_gma = par->kxx_gma;
	kxx_fq = par->kxx_fq;
	kxxp_gma = par->kxxp_gma;

	Iqe = par->Iqe;
	IIe = par->IIe;
	Iqep = par->Iqep;
	IIep = par->IIep;

	pke_ct = par->pke_ct;
	pq_sph = par->pq_sph;

	A1 = par->A1;
	B1 = par->B1;
	A2 = par->A2;
	B2 = par->B2;
	C = par->C;

	A1p = par->A1p;
	B1p = par->B1p;
	A2p = par->A2p;
	B2p = par->B2p;
	Cp = par->Cp;

	fac = par->fac;
	kf = par->kf;

	nth = par->nth;
	nq = par->nq;
	dimq = par->dimq;
	dimke = par->dimke;
	ke_flag = par->ke_flag;

	ktt12 = par->ktt12;
	ktx12 = par->ktx12;
	kl12_ct = par->kl12_ct;
	ktt12p = par->ktt12p;
	ktx12p = par->ktx12p;
	kl12p_ct = par->kl12p_ct;

	fqe = par->fqe;
	var_fq = par->var_fq;
	fqep = par->fqep;
	var_fqp = par->var_fqp;

	var_gma12 = par->var_gma12;
	var_gma12p = par->var_gma12p;

	lknxx_gma = malloc(nke * nke * sizeof(double));
	assert(lknxx_gma);
	lknxx_fq = malloc(nq * nq * sizeof(double));
	assert(lknxx_fq);
	lknxx_fqp = malloc(nq * nq * sizeof(double));
	assert(lknxx_fqp);

	wt_gma = malloc(nke * sizeof(double));
	assert(wt_gma);
	wt_fq = malloc(nke * nq * sizeof(double));
	assert(wt_fq);
	wt_fqp = malloc(nke * nq * sizeof(double));
	assert(wt_fqp);

	zs = malloc(nke * sizeof(double));
	assert(zs);
	zsp = malloc(nke * sizeof(double));
	assert(zsp);
	var_zs = malloc(nke * sizeof(double));
	assert(var_zs);
	var_zsp = malloc(nke * sizeof(double));
	assert(var_zsp);

	npke = dimke + 1;

	/* ZS */

	printf("ZS\n");

	get_noise_covar_chd(lknxx_gma, kxx_gma, var_gma0, nke);

	get_gma_weight(wt_gma, lknxx_gma, gma0, nke);

	get_var_mat_chd(var_gma12, ktt12, ktx12, lknxx_gma, 2 * nq, nke);

	get_fq_samples(fqe, var_fq, wt_gma, A1, B1, A2, B2, C, var_gma12, ke_flag, nq, nke);

	get_noise_covar_chd(lknxx_fq, kxx_fq, var_fq, nq);

	get_fq_weights_2(wt_fq, lknxx_fq, fqe, nq, nke);

	get_gma_gpr_mean(zs, Iqe, wt_fq, nke, nq);

	get_gma_gpr_var(var_zs, IIe, Iqe, lknxx_fq, nq, nke);

	/* ZSP */

	printf("ZSP\n");

	get_noise_covar_chd(lknxx_gma, kxxp_gma, var_gma0, nke);

	get_gma_weight(wt_gma, lknxx_gma, gma0, nke);

	get_var_mat_chd(var_gma12p, ktt12p, ktx12p, lknxx_gma, 2 * nq, nke);

	get_fq_samples(fqep, var_fqp, wt_gma, A1p, B1p, A2p, B2p, Cp, var_gma12p, ke_flag, nq, nke);

	get_noise_covar_chd(lknxx_fqp, kxx_fq, var_fqp, nq);

	get_fq_weights_2(wt_fqp, lknxx_fqp, fqep, nq, nke);

	get_gma_gpr_mean(zsp, Iqep, wt_fqp, nke, nq);

	get_gma_gpr_var(var_zsp, IIep, Iqep, lknxx_fqp, nq, nke);

	for (i = 0; i < nke; i++) {
		gma[i] = zs[i] - zsp[i];
		var_gma[i] = var_zs[i] + var_zsp[i];
	}

	free(wt_fq);
	free(wt_fqp);
	free(wt_gma);
	free(lknxx_fq);
	free(lknxx_fqp);
	free(lknxx_gma);
	free(zs);
	free(zsp);
	free(var_zs);
	free(var_zsp);
}
