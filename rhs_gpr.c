#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <lib_gpr/lib_gpr.h>
#include <lib_pots/lib_pots.h>
#include <atlas/blas.h>
#include "lib_flow.h"

#define DIMKE (7)

void get_rhs_block(double *gma, double *var_gma, const double *gma0, const double *var_gma0,
		   unsigned long nke, void *param)
{
	double *lknxx_gma, *kxx_gma, *wt_gma, *var_gma12, *ke_ct, *q_ct, *pke_ct, *wt_fq, *var_fq,
	    *A1, *B1, *C, *A2, *B2, *lknxx_fq, *kxx_fq, *Iqe, *q_sph, *pq_sph, fac, kf, *IIe, *fqe,
	    *ktt12, *ktx12, *kl12_ct, *reg12, *reg1x2;
	unsigned long nq, nth, i;
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

	reg12 = par->reg12;
	reg1x2 = par->reg1x2;

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

	get_fq_samples_reg(fqe, var_fq, wt_gma, A1, B1, A2, B2, C, var_gma12, reg12, reg1x2,
			   ke_flag, nq, nke);

	get_noise_covar_chd(lknxx_fq, kxx_fq, var_fq, nq);

	get_fq_weights_2(wt_fq, lknxx_fq, fqe, nq, nke);

	get_gma_gpr_mean(gma, Iqe, wt_fq, nke, nq);

	get_gma_gpr_var(var_gma, IIe, Iqe, lknxx_fq, nq, nke);

	free(wt_fq);
	free(wt_gma);
	free(lknxx_fq);
	free(lknxx_gma);
}

void get_rhs_ph(double *gma_2, double s, double *gma0_2, unsigned long n2ke, void *param)
{
	double *gma_zs, *gma_zsp, *var_gma_zs, *var_gma_zsp;
	unsigned long i, nke;
	struct rhs_param par_zs, par_zsp;
	struct rhs_param *par = param;

	nke = n2ke / 2;

	gma_zs = malloc(nke * sizeof(double));
	assert(gma_zs);
	gma_zsp = malloc(nke * sizeof(double));
	assert(gma_zsp);

	var_gma_zs = malloc(nke * sizeof(double));
	assert(var_gma_zs);
	var_gma_zsp = malloc(nke * sizeof(double));
	assert(var_gma_zsp);

	par_zs = par[0];
	par_zsp = par[1];

	get_rhs_block(gma_zs, var_gma_zs, gma0_2, &gma0_2[nke], nke, &par_zs);
	get_rhs_block(gma_zsp, var_gma_zsp, gma0_2, &gma0_2[nke], nke, &par_zsp);

	for (i = 0; i < nke; i++) {
		gma_2[i] = gma_zs[i] - gma_zsp[i];
		gma_2[nke + i] = var_gma_zs[i] + var_gma_zsp[i];
	}

	free(gma_zs);
	free(gma_zsp);
	free(var_gma_zs);
	free(var_gma_zsp);
}

void get_rhs_ph_lin(double *gma_2, double s, double *gma0_2, unsigned long n2ke, void *param)
{
	double *gma_zs, *gma_zsp, *var_gma_zs, *var_gma_zsp, e_ext;
	unsigned long i, nke;
	unsigned int dimke;
	struct rhs_param par_zs, par_zsp;
	struct rhs_param *par = param;

	nke = n2ke / 2;

	gma_zs = malloc(nke * sizeof(double));
	assert(gma_zs);
	gma_zsp = malloc(nke * sizeof(double));
	assert(gma_zsp);

	var_gma_zs = malloc(nke * sizeof(double));
	assert(var_gma_zs);
	var_gma_zsp = malloc(nke * sizeof(double));
	assert(var_gma_zsp);

	par_zs = par[0];
	par_zsp = par[1];

	dimke = DIMKE;

	get_rhs_block(gma_zs, var_gma_zs, gma0_2, &gma0_2[nke], nke, &par_zs);
	get_rhs_block(gma_zsp, var_gma_zsp, gma0_2, &gma0_2[nke], nke, &par_zsp);

	for (i = 0; i < nke; i++) {

		e_ext = get_energy_ext_7d_ct(&(par_zs.ke_ct[i * dimke]), dimke);

		gma_2[i] = -1.0 * e_ext * e_ext * gma0_2[i] + gma_zs[i] - gma_zsp[i];
		gma_2[nke + i]
		    = -2.0 * e_ext * e_ext * gma0_2[nke + i] + var_gma_zs[i] + var_gma_zsp[i];
	}

	free(gma_zs);
	free(gma_zsp);
	free(var_gma_zs);
	free(var_gma_zsp);
}

void get_rhs_diff_block(double *gma, double *var_gma, const double *gma0_zs,
			const double *var_gma0_zs, const double *gma0_zsp,
			const double *var_gma0_zsp, unsigned long nke, struct rhs_diff_param *par)
{
	double *lknxx_gma, *kxx_gma_zs, *kxx_gma_zsp, *wt_gma_zs, *wt_gma_zsp, *var_gma12_zs,
	    *var_gma12_zsp, *var_gma12, *ke_ct, *q_ct, *pke_ct_zs, *pke_ct_zsp, *wt_fq, *var_fq,
	    *A1, *B1, *C, *A2, *B2, *A1p, *B1p, *Cp, *A2p, *B2p, *lknxx_fq, *kxx_fq, *Iqe, *q_sph,
	    *pq_sph, fac, kf, *IIe, *fqe, *ktt12_zs, *ktx12_zs, *ktt12_zsp, *ktx12_zsp, *kl12_ct,
	    ALPHA;
	unsigned long nq, nth, i;
	unsigned int dimq, dimke, ke_flag;
	int N, INCX, INCY;

	ke_ct = par->ke_ct;
	q_ct = par->q_ct;
	q_sph = par->q_sph;

	kxx_gma_zs = par->kxx_gma_zs;
	kxx_gma_zsp = par->kxx_gma_zsp;
	kxx_fq = par->kxx_fq;

	Iqe = par->Iqe;
	IIe = par->IIe;

	pke_ct_zs = par->pke_ct_zs;
	pke_ct_zsp = par->pke_ct_zsp;
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

	ktt12_zs = par->ktt12_zs;
	ktx12_zs = par->ktx12_zs;

	kl12_ct = par->kl12_ct;

	ktt12_zsp = par->ktt12_zsp;
	ktx12_zsp = par->ktx12_zsp;

	fqe = par->fqe;
	var_fq = par->var_fq;
	var_gma12_zs = par->var_gma12_zs;
	var_gma12_zsp = par->var_gma12_zsp;
	var_gma12 = par->var_gma12;

	lknxx_fq = malloc(nq * nq * sizeof(double));
	assert(lknxx_fq);
	lknxx_gma = malloc(nke * nke * sizeof(double));
	assert(lknxx_gma);

	wt_gma_zs = malloc(nke * sizeof(double));
	assert(wt_gma_zs);
	wt_gma_zsp = malloc(nke * sizeof(double));
	assert(wt_gma_zsp);
	wt_fq = malloc(nke * nq * sizeof(double));
	assert(wt_fq);

	get_noise_covar_chd(lknxx_gma, kxx_gma_zs, var_gma0_zs, nke);

	get_gma_weight(wt_gma_zs, lknxx_gma, gma0_zs, nke);

	get_var_mat_chd(var_gma12_zs, ktt12_zs, ktx12_zs, lknxx_gma, 2 * nq, nke);

	get_noise_covar_chd(lknxx_gma, kxx_gma_zsp, var_gma0_zsp, nke);

	get_gma_weight(wt_gma_zsp, lknxx_gma, gma0_zsp, nke);

	get_var_mat_chd(var_gma12_zsp, ktt12_zsp, ktx12_zsp, lknxx_gma, 2 * nq, nke);

	N = 4 * nq * nq;
	INCX = 1;
	INCY = 1;
	ALPHA = 1.0;

	dcopy_(&N, var_gma12_zsp, &INCX, var_gma12, &INCY);

	daxpy_(&N, &ALPHA, var_gma12_zs, &INCX, var_gma12, &INCY);

	get_fq_as_samples(fqe, var_fq, wt_gma_zs, wt_gma_zsp, A1, B1, A2, B2, C, A1p, B1p, A2p, B2p,
			  Cp, var_gma12, ke_flag, nq, nke);

	get_noise_covar_chd(lknxx_fq, kxx_fq, var_fq, nq);

	get_fq_weights_2(wt_fq, lknxx_fq, fqe, nq, nke);

	get_gma_gpr_mean(gma, Iqe, wt_fq, nke, nq);

	get_gma_gpr_var(var_gma, IIe, Iqe, lknxx_fq, nq, nke);

	free(wt_fq);
	free(wt_gma_zs);
	free(wt_gma_zsp);
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
