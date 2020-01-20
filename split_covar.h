/*
 * FUNCTIONS TO COMPUTE TENSOR DECOMPOSED COVARIANCE MATRICES
 * FOR LOOP VERTEX INTERPOLATION.
 *
 * WIKI: covariance_split
 *
 */

struct split_covar {
	double *A1, *A2, *B1, *B2, *C;
};

/* FUNCTION TO COMPUTE ALL SPLIT COVAR IN SPLIT_COVAR STRUCTURE. */

void get_ph_split_covar(struct split_covar *scv, const double *ke_ct, unsigned long nke,
			unsigned int dimke, const double *q_ct, unsigned long nq, unsigned int dimq,
			const double *pke, unsigned long npke, int shft);

/* TESTS */

double test_zs_split_covar(unsigned long nke, unsigned long nq, int shft, int seed);
double test_zsp_split_covar(unsigned long nke, unsigned long nq, int shft, int seed);
