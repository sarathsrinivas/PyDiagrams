/*
 * Functions for vertex interpolation with
 * and without mean
 */

void get_covar_chd_noise(double *lknxx, const double *kxx, const double *var, unsigned long nx);
void get_gma_weight(double *wt_gma, const double *lknxx_gma, const double *gma, unsigned long nke);
void interpolate_gma(double *gma, const double *wt_gma, const double *Aeq, const double *Bes,
		     const double *Csq, unsigned long nq, unsigned long nke);
void add_mean(double *gma, const double *gma_mn, unsigned long ngma);
void subtract_mean(double *gma, const double *gma_mn, unsigned long ngma);

/* TESTS */

double test_interpolate_gma(unsigned long nke, unsigned long nq, int shft, int seed);
