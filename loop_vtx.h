/*
 *  FUNCTIONS REGARDING LOOP VERTEX MOMENTA.
 */

/*
 * LOOP MOMENTA FOR PH
 *
 * shft = +1 F(Q + DELTA) SAMPLES
 * shft = -1 F(Q - DELTA) SAMPLES
 *
 * WIKI: fermion_loop_integral
 */

void get_ph_loop_mom(double *kl1_ct, double *kl2_ct, const double *ke_ct, unsigned int dimke,
		     const double *q_ct, unsigned long nq, unsigned int dimq, int shft);
