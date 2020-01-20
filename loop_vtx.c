#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "loop_vtx.h"

void get_ph_loop_mom(double *kl1_ct, double *kl2_ct, const double *ke_ct, unsigned int dimke,
		     const double *q_ct, unsigned long nq, unsigned int dimq, int shft)
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

			dlp_zs1[i]
			    = 0.5 * (P_ct[i] + dlp_ct[i] - q_ct[l * dimq + i] - shft * dl_ct[i]);

			P_zs1[i]
			    = 0.5 * (P_ct[i] + dlp_ct[i] + q_ct[l * dimq + i] + shft * dl_ct[i]);

			dlp_zs2[i]
			    = 0.5 * (P_ct[i] - dlp_ct[i] - q_ct[l * dimq + i] - shft * dl_ct[i]);

			P_zs2[i]
			    = 0.5 * (P_ct[i] - dlp_ct[i] + q_ct[l * dimq + i] + shft * dl_ct[i]);
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
