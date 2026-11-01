#include <stdlib.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include "vjp/util/util.h"

#define OUTPUTIJ        1
#define INPUT_IJ        2

struct _AO2MOvjpEnvs {
    int nao;
    int nmo;
    int bra_start;
    int bra_count;
    int ket_start;
    int ket_count;
    double *mo_coeff;
};


int AO2MOmmm_nr_vjp_s2_iltj(double *eri_bar, double *mo_coeff_bar,
                            double *eri, double *ybar, double *buf,
                            struct _AO2MOvjpEnvs *envs, int seekdim)
{
    switch (seekdim) {
        case OUTPUTIJ: return envs->bra_count * envs->ket_count;
        case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
    }
    const double D0 = 0;
    const double D1 = 1;
    const char SIDE_L = 'L';
    const char SIDE_R = 'R';
    const char UPLO_U = 'U';
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    int nao = envs->nao;
    int nmo = envs->nmo;
    int i_start = envs->bra_start;
    int i_count = envs->bra_count;
    int j_start = envs->ket_start;
    int j_count = envs->ket_count;
    double *mo_coeff = envs->mo_coeff; // F order
    double *eri_bar_s1 = buf + nao * i_count;

    // |ij) C_qj = |iq); |ij) in C order
    dgemm_(&TRANS_T, &TRANS_T, &i_count, &nao, &j_count,
           &D1, ybar, &j_count, mo_coeff+j_start*nao, &nao,
           &D0, buf, &i_count);
    // |iq) C_pi = |pq); |pq) in C order
    dgemm_(&TRANS_T, &TRANS_T, &nao, &nao, &i_count,
           &D1, buf, &i_count, mo_coeff+i_start*nao, &nao,
           &D0, eri_bar_s1, &nao);

    pack_tril(nao, eri_bar, eri_bar_s1);

    // |iq) |pq) = C'_pi; |pq), C'_pi in C order
    dsymm_(&SIDE_R, &UPLO_U, &i_count, &nao,
           &D1, eri, &nao, buf, &i_count,
           &D1, mo_coeff_bar+i_start, &nmo);

    // |pq) C_pi = |qi); |pq) in C order
    dsymm_(&SIDE_L, &UPLO_U, &nao, &i_count,
           &D1, eri, &nao, mo_coeff+i_start*nao, &nao,
           &D0, buf, &nao);

    // |ij) |qi) = C'_qj; |ij), C'_qj in C order
    dgemm_(&TRANS_N, &TRANS_T, &j_count, &nao, &i_count,
           &D1, ybar, &j_count, buf, &nao,
           &D1, mo_coeff_bar+j_start, &nmo);
    return 0;
}


int AO2MOmmm_nr_vjp_s2_igtj(double *eri_bar, double *mo_coeff_bar,
                            double *eri, double *ybar, double *buf,
                            struct _AO2MOvjpEnvs *envs, int seekdim)
{
    switch (seekdim) {
        case OUTPUTIJ: return envs->bra_count * envs->ket_count;
        case INPUT_IJ: return envs->nao * (envs->nao+1) / 2;
    }
    const double D0 = 0;
    const double D1 = 1;
    const char SIDE_L = 'L';
    const char SIDE_R = 'R';
    const char UPLO_U = 'U';
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    int nao = envs->nao;
    int nmo = envs->nmo;
    int i_start = envs->bra_start;
    int i_count = envs->bra_count;
    int j_start = envs->ket_start;
    int j_count = envs->ket_count;
    double *mo_coeff = envs->mo_coeff; // C order
    double *eri_bar_s1 = buf + nao * j_count;

    // C_pi |ij) = |pj); C_pi, |ij) in C order
    dgemm_(&TRANS_T, &TRANS_T, &nao, &j_count, &i_count,
           &D1, mo_coeff+i_start, &nmo, ybar, &j_count,
           &D0, buf, &nao);
    // C_qj |pj) = |pq); C_qj, |pq) in C order
    dgemm_(&TRANS_T, &TRANS_T, &nao, &nao, &j_count,
           &D1, mo_coeff+j_start, &nmo, buf, &nao,
           &D0, eri_bar_s1, &nao);

    pack_tril(nao, eri_bar, eri_bar_s1);

    // |pq) |pj) = C'_qj; |pq) in C order
    dsymm_(&SIDE_L, &UPLO_U, &nao, &j_count,
           &D1, eri, &nao, buf, &nao,
           &D1, mo_coeff_bar+j_start*nao, &nao);

    // C_qj |pq) = |jp); |pq), C_qj in C order
    dsymm_(&SIDE_R, &UPLO_U, &j_count, &nao,
           &D1, eri, &nao, mo_coeff+j_start, &nmo,
           &D0, buf, &j_count);

    // |jp) |ij) = C'_pi; |ij) in C order
    dgemm_(&TRANS_T, &TRANS_N, &nao, &i_count, &j_count,
           &D1, buf, &j_count, ybar, &j_count,
           &D1, mo_coeff_bar+i_start*nao, &nao);
    return 0;
}


void AO2MOtranse2_nr_vjp_s2kl(int (*fmmm)(), int row_id,
                              double *eri_bar, double *mo_coeff_bar,
                              double *eri, double *ybar, double *buf,
                              struct _AO2MOvjpEnvs *envs)
{
    int nao = envs->nao;
    size_t ij_pair = (*fmmm)(NULL, NULL, NULL, NULL, buf, envs, OUTPUTIJ);
    size_t nao2 = (*fmmm)(NULL, NULL, NULL, NULL, buf, envs, INPUT_IJ);
    NPdunpack_tril(nao, eri+nao2*row_id, buf, 0);
    (*fmmm)(eri_bar+nao2*row_id, mo_coeff_bar, buf, ybar+ij_pair*row_id, buf+nao*nao, envs, 0);
}


void AO2MOtranse2_nr_vjp_s2(int (*fmmm)(), int row_id,
                            double *eri_bar, double *mo_coeff_bar,
                            double *eri, double *ybar, double *buf,
                            struct _AO2MOvjpEnvs *envs)
{
    AO2MOtranse2_nr_vjp_s2kl(fmmm, row_id, eri_bar, mo_coeff_bar, eri, ybar, buf, envs);
}


void AO2MOtranse2_nr_vjp_s4(int (*fmmm)(), int row_id,
                            double *eri_bar, double *mo_coeff_bar,
                            double *eri, double *ybar, double *buf,
                            struct _AO2MOvjpEnvs *envs)
{
    AO2MOtranse2_nr_vjp_s2kl(fmmm, row_id, eri_bar, mo_coeff_bar, eri, ybar, buf, envs);
}


void AO2MOnr_e2_vjp_drv(void (*ftrans)(), int (*fmmm)(),
                        double *eri_bar, double *mo_coeff_bar,
                        double *eri, double *mo_coeff, double *ybar,
                        int nij, int nao, int nmo, int *orbs_slice)
{
    struct _AO2MOvjpEnvs envs;
    envs.bra_start = orbs_slice[0];
    envs.bra_count = orbs_slice[1] - orbs_slice[0];
    envs.ket_start = orbs_slice[2];
    envs.ket_count = orbs_slice[3] - orbs_slice[2];
    envs.nao = nao;
    envs.nmo = nmo;
    envs.mo_coeff = mo_coeff;

    double **mo_coeff_bar_bufs = calloc(omp_get_max_threads_safe(), sizeof(double *));
    #pragma omp parallel
    {
        int i;
        int i_count = envs.bra_count;
        int j_count = envs.ket_count;
        int thread_id = omp_get_thread_num();
        double *mo_coeff_bar_priv;
        if (thread_id == 0) {
            mo_coeff_bar_priv = mo_coeff_bar;
        } else {
            mo_coeff_bar_priv = calloc(nao*nmo, sizeof(double));
        }
        mo_coeff_bar_bufs[thread_id] = mo_coeff_bar_priv;
        double *buf = malloc(sizeof(double) * (nao*nao*2 + nao*MIN(i_count, j_count)));
        #pragma omp for schedule(static)
        for (i = 0; i < nij; i++) {
            (*ftrans)(fmmm, i, eri_bar, mo_coeff_bar_priv, eri, ybar, buf, &envs);
        }
        free(buf);

        omp_dsum_reduce_inplace(mo_coeff_bar_bufs, nao*nmo);
        if (thread_id != 0) {
            free(mo_coeff_bar_priv);
        }
    }
    free(mo_coeff_bar_bufs);
}


void AO2MOnr_e2_mo_coeff_vjp_drv(int (*fmmm)(),
                        double *mo_coeff_bar,
                        double *eri, double *mo_coeff, double *ybar,
                        int nij, int nao, int nmo, int *orbs_slice)
{
    struct _AO2MOvjpEnvs envs;
    envs.bra_start = orbs_slice[0];
    envs.bra_count = orbs_slice[1] - orbs_slice[0];
    envs.ket_start = orbs_slice[2];
    envs.ket_count = orbs_slice[3] - orbs_slice[2];
    envs.nao = nao;
    envs.nmo = nmo;
    envs.mo_coeff = mo_coeff;

    size_t ij_pair = (*fmmm)(NULL, NULL, NULL, NULL, NULL, &envs, OUTPUTIJ);
    size_t nao2 = (*fmmm)(NULL, NULL, NULL, NULL, NULL, &envs, INPUT_IJ);

    double **mo_coeff_bar_bufs = calloc(omp_get_max_threads_safe(), sizeof(double *));
    #pragma omp parallel
    {
        int i;
        int i_count = envs.bra_count;
        int j_count = envs.ket_count;
        int thread_id = omp_get_thread_num();
        double *mo_coeff_bar_priv;
        if (thread_id == 0) {
            mo_coeff_bar_priv = mo_coeff_bar;
        } else {
            mo_coeff_bar_priv = calloc(nao*nmo, sizeof(double));
        }
        mo_coeff_bar_bufs[thread_id] = mo_coeff_bar_priv;
        double *eri_bar_row = malloc(sizeof(double) * nao2);
        double *buf = malloc(sizeof(double) * (nao*nao*2 + nao*MIN(i_count, j_count)));
        #pragma omp for schedule(static)
        for (i = 0; i < nij; i++) {
            NPdunpack_tril(nao, eri + nao2*i, buf, 0);
            (*fmmm)(eri_bar_row, mo_coeff_bar_priv, buf,
                    ybar + ij_pair*i, buf + nao*nao, &envs, 0);
        }
        free(buf);
        free(eri_bar_row);

        omp_dsum_reduce_inplace(mo_coeff_bar_bufs, nao*nmo);
        if (thread_id != 0) {
            free(mo_coeff_bar_priv);
        }
    }
    free(mo_coeff_bar_bufs);
}


void AO2MOnr_e2_cderi_bar_project_omp(
        double *out, const double *y2,
        const double *mok_rows, const double *mol_cols,
        int naux, int kc, int lc, int npos, int blksize)
{
    if (naux <= 0 || kc <= 0 || lc <= 0 || npos <= 0) {
        return;
    }

    const int m = naux * lc;
    const double D1 = 1.0;
    const double D0 = 0.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    int block = blksize;
    if (block <= 0) {
        block = 1;
    }
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (max_threads > 1) {
        block = (block + max_threads - 1) / max_threads;
    }
#endif
    if (block < 1) {
        block = 1;
    }
    if (block > npos) {
        block = npos;
    }

    const int nblocks = (npos + block - 1) / block;

#pragma omp parallel
    {
        double *tmp = malloc(sizeof(double) * (size_t)m * block);

#pragma omp for schedule(static)
        for (int ib = 0; ib < nblocks; ib++) {
            const int p0 = ib * block;
            const int nb = (p0 + block <= npos) ? block : (npos - p0);

            /*
             * tmp[(P,l),p] = sum_k ybar[P,k,l] * mok_rows[p,k]
             *
             * y2 is the C-contiguous view ybar.transpose(0,2,1)
             * reshaped to (naux*lc, kc).  The row-major result tmp
             * (m, nb) is represented to Fortran BLAS as (nb, m).
             */
            dgemm_(&TRANS_T, &TRANS_N,
                   &nb, &m, &kc,
                   &D1, mok_rows + (size_t)p0 * kc, &kc,
                   y2, &kc,
                   &D0, tmp, &nb);

            for (int p = 0; p < nb; p++) {
                const double *col = mol_cols + (size_t)(p0 + p) * lc;
                for (int x = 0; x < naux; x++) {
                    double acc = 0.0;
                    const double *tmp_xp = tmp + (size_t)x * lc * nb + p;
                    for (int l = 0; l < lc; l++) {
                        acc += tmp_xp[(size_t)l * nb] * col[l];
                    }
                    out[(size_t)x * npos + p0 + p] = acc;
                }
            }
        }

        free(tmp);
    }
}


/*
 * Transform one auxiliary slab of an MO-integral cotangent back to every
 * packed AO pair.
 *
 * y2 is the C-contiguous view
 *
 *     ybar.transpose(0, 2, 1).reshape(naux * lc, kc)
 *
 * and mo_k/mo_l are C-contiguous (nao,kc)/(nao,lc) coefficient blocks.
 * The caller may transpose ybar and exchange mo_k/mo_l before entering this
 * routine so that lc = min(original_kc, original_lc).  For each auxiliary
 * row P, the two BLAS contractions are
 *
 *     tmp[P,l,u] = sum_k ybar[P,k,l] * mo_k[u,k]
 *     mat[P,u,v] = sum_l tmp[P,l,u] * mo_l[v,l].
 *
 * The packed s2 cotangent is mat[u,v] + mat[v,u] off diagonal and mat[u,u]
 * on the diagonal.  This is the same symmetry convention as pack_tril, but
 * values are assigned here (rather than accumulated) so ``out`` need not be
 * initialized by the caller.
 */
int AO2MOnr_e2_cderi_bar_pack_aux_block(
        double *out, const double *y2,
        const double *mo_k, const double *mo_l,
        int naux, int nao, int kc, int lc)
{
    if (naux <= 0 || nao <= 0 || kc <= 0 || lc <= 0) {
        return 0;
    }

    const int m = naux * lc;
    const size_t npair = (size_t)nao * (nao + 1) / 2;
    const double D1 = 1.0;
    const double D0 = 0.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    /* Row-major tmp(m,nao) = y2(m,kc) * mo_k(nao,kc)^T. */
    double *tmp = malloc(sizeof(double) * (size_t)m * nao);
    if (tmp == NULL) {
        return 1;
    }
    dgemm_(&TRANS_T, &TRANS_N,
           &nao, &m, &kc,
           &D1, mo_k, &kc,
           y2, &kc,
           &D0, tmp, &nao);

    int allocation_failed = 0;
    int nthreads = omp_get_max_threads_safe();
    if (nthreads > naux) {
        nthreads = naux;
    }
    if (nthreads < 1) {
        nthreads = 1;
    }
#pragma omp parallel num_threads(nthreads)
    {
        double *mat = malloc(sizeof(double) * (size_t)nao * nao);
        if (mat == NULL) {
#pragma omp atomic write
            allocation_failed = 1;
        }

#pragma omp for schedule(static)
        for (int p = 0; p < naux; p++) {
            if (mat == NULL) {
                continue;
            }

            /*
             * Row-major mat(nao,nao) = tmp[P](lc,nao)^T
             *                              * mo_l(nao,lc)^T.
             */
            const double *tmp_p = tmp + (size_t)p * lc * nao;
            dgemm_(&TRANS_T, &TRANS_T,
                   &nao, &nao, &lc,
                   &D1, mo_l, &lc,
                   tmp_p, &nao,
                   &D0, mat, &nao);

            double *out_p = out + (size_t)p * npair;
            size_t uv = 0;
            for (int u = 0; u < nao; u++) {
                for (int v = 0; v < u; v++, uv++) {
                    out_p[uv] = mat[(size_t)u * nao + v]
                              + mat[(size_t)v * nao + u];
                }
                out_p[uv] = mat[(size_t)u * nao + u];
                uv++;
            }
        }

        free(mat);
    }
    free(tmp);
    return allocation_failed;
}
