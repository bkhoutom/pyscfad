#include <stdlib.h>
#include "config.h"
#include "vhf/fblas.h"
#include "vjp/util/util.h"

void dfccsd_ovvv_vjp(
    double *Lov_bar, double *Lvv_bar,
    const double *Lov, const double *Lvv,
    const double *t1,
    const double *theta,
    const double *tau,
    const double *t1new_bar,
    const double *fvv_bar,
    const double *wVOov_bar,
    const double *tmp_acc_bar,
    const double *wooVV_flat_bar,
    int naux, int nocc, int nvir)
{
    const size_t nvir_pair = (size_t)nvir * (nvir + 1) / 2;
    const size_t Lov_size = (size_t)naux * nocc * nvir;
    const size_t Lvv_size = (size_t)naux * nvir_pair;
    const size_t nrow = (size_t)nocc * nvir;
    const size_t packed_size = nrow * nvir_pair;
    const int nocc2 = nocc * nocc;
    const int nvir_pair_i = (int)nvir_pair;
    const int nrow_i = (int)nrow;
    const double D1 = 1.0;
    const double D0 = 0.0;
    const char TRANS_N = 'N';
    const char TRANS_T = 'T';

    for (size_t n = 0; n < Lov_size; n++) {
        Lov_bar[n] = 0.0;
    }
    for (size_t n = 0; n < Lvv_size; n++) {
        Lvv_bar[n] = 0.0;
    }

    double *packed_all = calloc(packed_size, sizeof(double));

#pragma omp parallel
    {
        double *dense = malloc(sizeof(double) * (size_t)nvir * nvir);
        double *tau_buf = malloc(sizeof(double) * (size_t)nocc2 * nvir);
        double *tmp_buf = malloc(sizeof(double) * (size_t)nocc2 * nvir);

#pragma omp for schedule(dynamic)
        for (int p = 0; p < nvir; p++) {
            for (int io = 0; io < nocc; io++) {
                for (int jo = 0; jo < nocc; jo++) {
                    const int ij = io * nocc + jo;
                    for (int d = 0; d < nvir; d++) {
                        const size_t tidx = (((size_t)io*nocc + jo)*nvir + p)*nvir + d;
                        tau_buf[(size_t)ij*nvir + d] = tau[tidx];
                    }
                }
            }

            for (int i = 0; i < nocc; i++) {
                double *packed = packed_all + ((size_t)i * nvir + p) * nvir_pair;

                for (size_t n = 0; n < (size_t)nvir * nvir; n++) {
                    dense[n] = 0.0;
                }

                /*
                 * t1new += einsum('icjb,cjba->ia', theta[:,p,:,:], vovv[p])
                 * vovv[p,i,b,a] bar += sum_I theta[I,p,i,b] * t1new_bar[I,a]
                 */
                for (int b = 0; b < nvir; b++) {
                    for (int a = 0; a < nvir; a++) {
                        double s = 0.0;
                        for (int io = 0; io < nocc; io++) {
                            const size_t th = (((size_t)io*nvir + p)*nocc + i)*nvir + b;
                            s += theta[th] * t1new_bar[(size_t)io*nvir + a];
                        }
                        dense[(size_t)b*nvir + a] += s;
                    }
                }

                /*
                 * wVOov[p,i,j,a] = einsum('piac,jc->pija', vovv[p,i,a,c], t1[j,c])
                 */
                for (int a = 0; a < nvir; a++) {
                    for (int c = 0; c < nvir; c++) {
                        double s = 0.0;
                        for (int j = 0; j < nocc; j++) {
                            const size_t widx = (((size_t)p*nocc + i)*nocc + j)*nvir + a;
                            s += wVOov_bar[widx] * t1[(size_t)j*nvir + c];
                        }
                        dense[(size_t)a*nvir + c] += s;
                    }
                }

                /*
                 * tmp_acc[I,J,b,i] += einsum('IJpd,pdbi->IJbi', tau, vvvo)
                 * with vvvo[p,d,b,i] = vovv[p,i,d,b].
                 */
                for (int io = 0; io < nocc; io++) {
                    for (int jo = 0; jo < nocc; jo++) {
                        const int ij = io * nocc + jo;
                        for (int b = 0; b < nvir; b++) {
                            const size_t bidx = (((size_t)io*nocc + jo)*nvir + b)*nocc + i;
                            tmp_buf[(size_t)ij*nvir + b] = tmp_acc_bar[bidx];
                        }
                    }
                }
                dgemm_(&TRANS_N, &TRANS_T,
                       &nvir, &nvir, &nocc2,
                       &D1, tmp_buf, &nvir,
                       tau_buf, &nvir,
                       &D1, dense, &nvir);

                /*
                 * fvv[:,p] += -einsum('kc,pkca->ap', t1, vovv[p])
                 */
                for (int c = 0; c < nvir; c++) {
                    const double t1ic = t1[(size_t)i*nvir + c];
                    for (int a = 0; a < nvir; a++) {
                        dense[(size_t)c*nvir + a] -= fvv_bar[(size_t)a*nvir + p] * t1ic;
                    }
                }

                /*
                 * fvv += 2 * einsum('kp,pkab->ab', t1[:,p], vovv[p])
                 */
                const double t1ip2 = 2.0 * t1[(size_t)i*nvir + p];
                for (int a = 0; a < nvir; a++) {
                    for (int b = 0; b < nvir; b++) {
                        dense[(size_t)a*nvir + b] += t1ip2 * fvv_bar[(size_t)a*nvir + b];
                    }
                }

                for (size_t n = 0; n < nvir_pair; n++) {
                    packed[n] = 0.0;
                }
                pack_tril(nvir, packed, dense);

                /*
                 * wooVV_flat -= dot(t1[:,p].T, vovv_packed[p].reshape(...))
                 */
                for (int k = 0; k < nocc; k++) {
                    const double fac = -t1[(size_t)k*nvir + p];
                    const double *woo = wooVV_flat_bar + (size_t)k * nocc * nvir_pair
                                                      + (size_t)i * nvir_pair;
                    for (size_t q = 0; q < nvir_pair; q++) {
                        packed[q] += fac * woo[q];
                    }
                }
            }
        }

        free(dense);
        free(tau_buf);
        free(tmp_buf);
    }

    /*
     * packed_all is row-major B[nocc*nvir, npair], with row i*nvir+p.
     * In column-major BLAS views:
     *   B       is Bc[npair, nrow]
     *   Lvv     is A[npair, naux]
     *   Lov_bar is C[nrow, naux]
     * so Lov_bar = B.T @ Lvv.
     */
    dgemm_(&TRANS_T, &TRANS_N,
           &nrow_i, &naux, &nvir_pair_i,
           &D1, packed_all, &nvir_pair_i,
           Lvv, &nvir_pair_i,
           &D0, Lov_bar, &nrow_i);

    /*
     * Lvv_bar = B @ Lov, using the same column-major views:
     *   B       is Bc[npair, nrow]
     *   Lov     is L[nrow, naux]
     *   Lvv_bar is C[npair, naux].
     */
    dgemm_(&TRANS_N, &TRANS_N,
           &nvir_pair_i, &naux, &nrow_i,
           &D1, packed_all, &nvir_pair_i,
           Lov, &nrow_i,
           &D0, Lvv_bar, &nvir_pair_i);

    free(packed_all);
}
