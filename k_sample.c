#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include "lib_flow.h"

#define PI (3.1415926535897)

void fill_ke_sample_zs_ct_box(double *ke_ct, unsigned long ns, double *st, double *en,
			      unsigned int seed)
{
	unsigned long i, dim;
	double dl, dlp, P, th_P, phi_P, th_dlp, phi_dlp, dlz, dlpx, dlpy, dlpz, Px, Py, Pz;
	dsfmt_t drng;
	dim = 7;

	dsfmt_init_gen_rand(&drng, seed);

	for (i = 0; i < ns; i++) {

		dlz = st[0] + (en[0] - st[0]) * dsfmt_genrand_close_open(&drng);

		dlpx = st[1] + (en[1] - st[1]) * dsfmt_genrand_close_open(&drng);
		dlpy = st[1] + (en[1] - st[1]) * dsfmt_genrand_close_open(&drng);
		dlpz = st[1] + (en[1] - st[1]) * dsfmt_genrand_close_open(&drng);

		Px = st[2] + (en[2] - st[2]) * dsfmt_genrand_close_open(&drng);
		Py = st[2] + (en[2] - st[2]) * dsfmt_genrand_close_open(&drng);
		Pz = st[2] + (en[2] - st[2]) * dsfmt_genrand_close_open(&drng);

		ke_ct[dim * i + 0] = dlz;
		ke_ct[dim * i + 1] = dlpx;
		ke_ct[dim * i + 2] = dlpy;
		ke_ct[dim * i + 3] = dlpz;
		ke_ct[dim * i + 4] = Px;
		ke_ct[dim * i + 5] = Py;
		ke_ct[dim * i + 6] = Pz;
	}
}

void fill_ke_sample_zs_ct(double *ke_ct, unsigned long ns, double *st, double *en,
			  unsigned int seed)
{
	unsigned long i, dim;
	double dl, dlp, P, th_P, phi_P, th_dlp, phi_dlp, dlz, dlpx, dlpy, dlpz, Px, Py, Pz;
	dsfmt_t drng;
	dim = 7;

	dsfmt_init_gen_rand(&drng, seed);

	for (i = 0; i < ns; i++) {

		dl = dsfmt_genrand_close_open(&drng);
		dlp = dsfmt_genrand_close_open(&drng);
		P = dsfmt_genrand_close_open(&drng);

		dl = st[0] + (en[0] - st[0]) * pow(dl, 1.0 / 3.0);
		dlp = st[1] + (en[1] - st[1]) * pow(dlp, 1.0 / 3.0);
		P = st[2] + (en[2] - st[2]) * pow(P, 1.0 / 3.0);

		th_dlp = acos(1 - 2 * dsfmt_genrand_close_open(&drng));
		th_P = acos(1 - 2 * dsfmt_genrand_close_open(&drng));
		phi_dlp = 2 * PI * dsfmt_genrand_close_open(&drng);
		phi_P = 2 * PI * dsfmt_genrand_close_open(&drng);

		dlz = dl;

		dlpx = dlp * cos(phi_dlp) * sin(th_dlp);
		dlpy = dlp * sin(phi_dlp) * sin(th_dlp);
		dlpz = dlp * cos(th_dlp);

		Px = P * cos(phi_P) * sin(th_P);
		Py = P * sin(phi_P) * sin(th_P);
		Pz = P * cos(th_P);

		ke_ct[dim * i + 0] = dlz;
		ke_ct[dim * i + 1] = dlpx;
		ke_ct[dim * i + 2] = dlpy;
		ke_ct[dim * i + 3] = dlpz;
		ke_ct[dim * i + 4] = Px;
		ke_ct[dim * i + 5] = Py;
		ke_ct[dim * i + 6] = Pz;
	}
}

void fill_ke_sample_zs_ct_exp(double *ke_ct, unsigned long ns, double *st, double *en, double sig,
			      double a, unsigned int seed)
{
	unsigned long i, dim;
	double dl, dlp, P, th_P, phi_P, th_dlp, phi_dlp, dlz, dlpx, dlpy, dlpz, Px, Py, Pz;
	dsfmt_t drng;
	dim = 7;

	dsfmt_init_gen_rand(&drng, seed);

	for (i = 0; i < ns; i++) {

		dl = dsfmt_genrand_close_open(&drng);
		dlp = dsfmt_genrand_close_open(&drng);
		P = dsfmt_genrand_close_open(&drng);

		dl = a * fabs(1.0 + log(dl) * sig) + (1.0 - a) * dl;
		dlp = a * fabs(1.0 + log(dlp) * sig) + (1.0 - a) * dlp;
		P = a * fabs(1.0 + log(P) * sig) + (1.0 - a) * P;

		dl = st[0] + (en[0] - st[0]) * pow(dl, 1.0 / 3.0);
		dlp = st[1] + (en[1] - st[1]) * pow(dlp, 1.0 / 3.0);
		P = st[2] + (en[2] - st[2]) * pow(P, 1.0 / 3.0);

		th_dlp = acos(1 - 2 * dsfmt_genrand_close_open(&drng));
		th_P = acos(1 - 2 * dsfmt_genrand_close_open(&drng));
		phi_dlp = 2 * PI * dsfmt_genrand_close_open(&drng);
		phi_P = 2 * PI * dsfmt_genrand_close_open(&drng);

		dlz = dl;

		dlpx = dlp * cos(phi_dlp) * sin(th_dlp);
		dlpy = dlp * sin(phi_dlp) * sin(th_dlp);
		dlpz = dlp * cos(th_dlp);

		Px = P * cos(phi_P) * sin(th_P);
		Py = P * sin(phi_P) * sin(th_P);
		Pz = P * cos(th_P);

		ke_ct[dim * i + 0] = dlz;
		ke_ct[dim * i + 1] = dlpx;
		ke_ct[dim * i + 2] = dlpy;
		ke_ct[dim * i + 3] = dlpz;
		ke_ct[dim * i + 4] = Px;
		ke_ct[dim * i + 5] = Py;
		ke_ct[dim * i + 6] = Pz;
	}
}

void get_kep_sample_zsp_ct(double *kep_ct, const double *ke_ct, unsigned long nke, unsigned int dim)
{
	double dl, dlp, P, dl_dlp, P_dlp, P_dl, phi_dl, phi_P, dlx, dlz, dly, dlpz, dlpx, dlpy, Px,
	    Py, Pz;
	unsigned long i;

	for (i = 0; i < nke; i++) {

		dlz = ke_ct[dim * i + 0];
		dlpx = ke_ct[dim * i + 1];
		dlpy = ke_ct[dim * i + 2];
		dlpz = ke_ct[dim * i + 3];
		Px = ke_ct[dim * i + 4];
		Py = ke_ct[dim * i + 5];
		Pz = ke_ct[dim * i + 6];

		dl = dlz;
		dlp = sqrt(dlpx * dlpx + dlpy * dlpy + dlpz * dlpz);
		P = sqrt(Px * Px + Py * Py + Pz * Pz);

		dl_dlp = acos((dlpz * dlz) / (dlp * dl));
		P_dl = acos((Pz * dlz) / (P * dl));
		P_dlp = acos((Px * dlpx + Py * dlpy + Pz * dlpz) / (P * dlp));

		phi_P = 0.4;
		phi_dl = acos((cos(P_dl) - cos(P_dlp) * cos(dl_dlp)) / (sin(P_dlp) * sin(dl_dlp)))
			 + phi_P;

		dlpz = dlp;

		dlx = dl * cos(phi_dl) * sin(dl_dlp);
		dly = dl * sin(phi_dl) * sin(dl_dlp);
		dlz = dl * cos(dl_dlp);

		Px = P * cos(phi_P) * sin(P_dlp);
		Py = P * sin(phi_P) * sin(P_dlp);
		Pz = P * cos(P_dlp);

		kep_ct[dim * i + 0] = dlpz;
		kep_ct[dim * i + 1] = dlx;
		kep_ct[dim * i + 2] = dly;
		kep_ct[dim * i + 3] = dlz;
		kep_ct[dim * i + 4] = Px;
		kep_ct[dim * i + 5] = Py;
		kep_ct[dim * i + 6] = Pz;
	}
}

void fill_q_sample(double *ke, unsigned long ns, double st, double en, unsigned int seed)
{
	unsigned long i;
	double q, th_q, phi_q;
	dsfmt_t drng;

	dsfmt_init_gen_rand(&drng, seed);

	for (i = 0; i < ns; i++) {

		q = dsfmt_genrand_close_open(&drng);

		q = st + (en - st) * pow(q, 1.0 / 3.0);

		th_q = acos(1 - 2 * dsfmt_genrand_close_open(&drng));
		phi_q = 2 * PI * dsfmt_genrand_close_open(&drng);

		ke[3 * i + 0] = q;
		ke[3 * i + 1] = th_q;
		ke[3 * i + 2] = phi_q;
	}
}

void sph_to_ct(double *q_ct, const double *q, unsigned int dimq, unsigned long nq)
{
	double qx, qy, qz, r_q, th_q, phi_q;
	unsigned long i;

	for (i = 0; i < nq; i++) {

		r_q = q[dimq * i + 0];
		th_q = q[dimq * i + 1];
		phi_q = q[dimq * i + 2];

		qx = r_q * cos(phi_q) * sin(th_q);
		qy = r_q * sin(phi_q) * sin(th_q);
		qz = r_q * cos(th_q);

		q_ct[dimq * i + 0] = qx;
		q_ct[dimq * i + 1] = qy;
		q_ct[dimq * i + 2] = qz;
	}
}

void print_mom(const double *k, unsigned long nk, unsigned int dimk, FILE *out)
{
	unsigned long i;
	unsigned int j;

	for (i = 0; i < nk; i++) {
		for (j = 0; j < dimk; j++) {

			fprintf(out, "%+.15E ", k[i * dimk + j]);
		}
		fprintf(out, "\n");
	}
}

double test_zs_zsp_rot(unsigned long nke, int seed)
{
	double *ke_ct, *kep_ct, st[3], en[3], kmax, dl[2], dlp[2], P[2], dl_dlp[2], P_dl[2],
	    P_dlp[2], dlx, dly, dlz, dlpx, dlpy, dlpz, Px, Py, Pz, err;
	unsigned int dimke;
	unsigned long i;

	dimke = 7;
	kmax = 3;

	ke_ct = malloc(nke * dimke * sizeof(double));
	assert(ke_ct);
	kep_ct = malloc(nke * dimke * sizeof(double));
	assert(kep_ct);

	st[0] = 0;
	en[0] = kmax;
	st[1] = 0;
	en[1] = kmax;
	st[2] = 0;
	en[2] = kmax;

	fill_ke_sample_zs_ct(ke_ct, nke, st, en, seed);

	get_kep_sample_zsp_ct(kep_ct, ke_ct, nke, dimke);

	err = 0;
	for (i = 0; i < nke; i++) {

		dlz = ke_ct[dimke * i + 0];
		dlpx = ke_ct[dimke * i + 1];
		dlpy = ke_ct[dimke * i + 2];
		dlpz = ke_ct[dimke * i + 3];
		Px = ke_ct[dimke * i + 4];
		Py = ke_ct[dimke * i + 5];
		Pz = ke_ct[dimke * i + 6];

		dl[0] = dlz;
		dlp[0] = sqrt(dlpx * dlpx + dlpy * dlpy + dlpz * dlpz);
		P[0] = sqrt(Px * Px + Py * Py + Pz * Pz);

		dl_dlp[0] = dlz * dlpz;
		P_dl[0] = Pz * dlz;
		P_dlp[0] = Px * dlpx + Py * dlpy + Pz * dlpz;

		dlpz = kep_ct[dimke * i + 0];
		dlx = kep_ct[dimke * i + 1];
		dly = kep_ct[dimke * i + 2];
		dlz = kep_ct[dimke * i + 3];
		Px = kep_ct[dimke * i + 4];
		Py = kep_ct[dimke * i + 5];
		Pz = kep_ct[dimke * i + 6];

		dlp[1] = dlpz;
		dl[1] = sqrt(dlx * dlx + dly * dly + dlz * dlz);
		P[1] = sqrt(Px * Px + Py * Py + Pz * Pz);

		dl_dlp[1] = dlz * dlpz;
		P_dl[1] = Px * dlx + Py * dly + Pz * dlz;
		P_dlp[1] = Pz * dlpz;

		err += fabs(dl[1] - dl[0]);
		err += fabs(dlp[1] - dlp[0]);
		err += fabs(P[1] - P[0]);
		err += fabs(dl_dlp[1] - dl_dlp[0]);
		err += fabs(P_dl[1] - P_dl[0]);
		err += fabs(P_dlp[1] - P_dlp[0]);
	}

	free(ke_ct);
	free(kep_ct);

	return err;
}
