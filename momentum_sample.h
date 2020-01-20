/*
 * FUNCTIONS TO GET RANDOM SAMPLES OF MOMENTA
 */

/* EXTERNAL MOMENTA DISTRIBUTED UNIFORMLY IN 7-D BOX */

void fill_ph_sample_ct_box(double *ke_ct, unsigned long ns, double *st, double *en,
			   unsigned int seed);

/* EXCHANGE MOMENTA DELTA -> DELTA'  */

void get_ph_ex_sample(double *kep_ct, const double *ke_ct, unsigned long nke, unsigned int dim);

/* LOOP MOMENTA FOR INTEGRATION UNIFORM IN 3-BALL */

void fill_q_sample_ball(double *qsph, unsigned long ns, double st, double en, unsigned int seed);

void sph_to_ct(double *q_ct, const double *qsph, unsigned int dimq, unsigned long nq);

void print_mom(const double *k, unsigned long nk, unsigned int dimk, FILE *out);

/* TESTING EXCHANGE MOMENTA */

double test_zs_zsp_rot(unsigned long nke, int seed);
