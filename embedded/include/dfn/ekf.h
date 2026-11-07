// Structured EKF of dfnekf.ekf.EKF (coords = ("Rf",), transform "free": theta = log Rf multiplier; LAM, Ds frozen).
//
// Step Jacobian from the implicit function theorem on the converged closure, A = Lambda + U V^T, never formed:
//   Lambda = diag(1 | e_m | rho_k) on the particles, E on c_e, 1 on theta;
//   U = [B (flux columns) | dPhi/dc_avg,p at fixed flux (D_p(y) law)],  V^T = [C = dj/dx ; unit rows on c_avg,p].
// Covariance in the scaled coordinates of the Python filter (u = (x - x0) / scale). Two forms (DFN_COV_UD):
//   1 (default) UD: P = U D U^T. Update: Bierman's scalar update, O(NX^2). Prediction: Thornton's modified weighted
//     Gram-Schmidt on W = [Phi U | e_Rf] with weights [D, q_Rf], O(NX^3), in place (U, transposed, holds Phi U).
//     Positive (semi)definite by construction: the form for float. D is floored at the Python filter's jitter (1e-12).
//     Memory: (NX + 1) x NX + two NX x NR buffers (NR = NEL + NP).
//   0 Joseph: P dense; rank-1 Joseph update written symmetric (P - K Ph^T - Ph K^T + S K K^T); prediction in place,
//     P <- Lambda P Lambda^T, P += W U^T + U W^T, W = Lambda P V + U (V^T P V) / 2, O(NX^2 NR). As dfnekf in double;
//     in float it loses definiteness (CYC20: at the first top of charge). Memory: NX^2 + four NX x NR buffers.
#pragma once
#include "dfn/dfn.h"
#ifndef DFN_COV_UD
#define DFN_COV_UD 1
#endif

namespace dfn {

constexpr int NX = NZ + NTH;
constexpr int NR = NEL + NP;
constexpr int ITH = NZ;                    // index of the Rf coordinate

struct EkfOut {
    real_t V, innov, Sy, soc, sd_soc, Rf, sd_Rf, res, min_diag;
    int nit;
};

class EKF {
public:
    real_t x[NX];
#if DFN_COV_UD
    real_t U[NX + 1][NX];                  // unit upper triangular (row NX: the noise row of the time update)
    real_t D[NX + 1];
#else
    real_t P[NX][NX];
    real_t PV[NX][NR], Wb[NX][NR];
#endif
    creal_t g[NG];
    real_t Us[NX][NR], Vts[NR][NX];
    real_t f[NX], hs[NX], Ph[NX], Kg[NX], du[NX];
    real_t Swc[NO][NEL], Swe[NO][N], Ckp[NO][NP], Cth[NO];
    real_t dvec[NZP];
    creal_t Xo[NO][NG];
    StepConsts C;
    ClosureIn in;
    Closure cl;
    real_t w_ws[NX], w_ld[NX], w_fv[NX], w_v[NX], w_b[NX], w_t[NX], w_Dw[NX + 1], w_z[NZ];   // work arrays
    real_t R;
    real_t jitter;
    real_t ce_floor;

    // x0: initial state (NZ) and coordinate; P0 from the model header: diag(P0_sd^2) + p0_q^2 dz_dq dz_dq^T (z block)
    void init(const real_t* z0, real_t c0, int idt0, real_t I0) {
        for (int i = 0; i < NZ; ++i) x[i] = z0[i];
        x[ITH] = c0;
        R = R_meas; jitter = (real_t)1e-12; ce_floor = (real_t)20;
#if DFN_COV_UD
        real_t (*P)[NX] = U;                                      // P0 built in U's storage, factorized in place
#endif
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NX; ++j) {
                real_t v = 0;
                if (i < NZ && j < NZ) v = p0_q * p0_q * dz_dq[i] * dz_dq[j];
                if (i == j) v += P0_sd[i] * P0_sd[i];
                P[i][j] = v / (scale[i] * scale[j]);
            }
#if DFN_COV_UD
        ud_factor();
#endif
        step_consts(x, idt0, C);
        closure_in(x, I0, Rf0 * r_exp(x[ITH]), C, in);
        cl.init_refs(in);
        cl.guess(in, in.I, g);
        int k;
        cl.newton(g, in, (creal_t)DFN_NEWTON_TOL, 60, &k);
    }

#if !DFN_COV_UD
    // ---- Lambda applied to the rows of an NX x NR block (in place)
    void lam_rows(real_t (*M)[NR]) {
        for (int i = 0; i < NZP; ++i)
            for (int c = 0; c < NR; ++c) M[i][c] *= dvec[i];
        real_t t[N];
        for (int c = 0; c < NR; ++c) {
            for (int r = 0; r < N; ++r) {
                real_t s = 0;
                for (int l = 0; l < N; ++l) s += E[C.idt][r][l] * M[IE + l][c];
                t[r] = s;
            }
            for (int r = 0; r < N; ++r) M[IE + r][c] = t[r];
        }
    }

#endif
    real_t dpart(int i) const { return i < NZP ? dvec[i] : (real_t)1; }

#if !DFN_COV_UD
    // P <- Lambda P Lambda^T (in place, symmetric)
    void lam_P() {
        const int d = C.idt;
        real_t t[N];
        for (int i = 0; i < NX; ++i) {
            if (i >= IE && i < IE + N) continue;
            const real_t di = dpart(i);
            for (int j = 0; j < NX; ++j)
                if (!(j >= IE && j < IE + N)) P[i][j] *= di * dpart(j);
            for (int r = 0; r < N; ++r) {
                real_t s = 0;
                for (int l = 0; l < N; ++l) s += E[d][r][l] * P[i][IE + l];
                t[r] = s;
            }
            for (int r = 0; r < N; ++r) { P[i][IE + r] = di * t[r]; P[IE + r][i] = di * t[r]; }
        }
        real_t T1[N][N];
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                real_t s = 0;
                for (int l = 0; l < N; ++l) s += E[d][r][l] * P[IE + l][IE + c];
                T1[r][c] = s;
            }
        for (int r = 0; r < N; ++r)
            for (int c = r; c < N; ++c) {
                real_t s = 0;
                for (int l = 0; l < N; ++l) s += T1[r][l] * E[d][c][l];
                P[IE + r][IE + c] = s; P[IE + c][IE + r] = s;
            }
    }

#endif

    // Phi u = Lambda u + U (V^T u) in scaled coordinates (out must not alias u)
    void phi_apply(const real_t* u, real_t* out) const {
        real_t vd[NR];
        for (int c = 0; c < NR; ++c) {
            real_t s = 0;
            const real_t* v = Vts[c];
            for (int j = 0; j < NX; ++j) if (v[j] != 0) s += v[j] * u[j];
            vd[c] = s;
        }
        for (int i = 0; i < NX; ++i) out[i] = (i >= IE && i < IE + N) ? (real_t)0 : dpart(i) * u[i];
        for (int r = 0; r < N; ++r) {
            real_t s = 0;
            for (int l = 0; l < N; ++l) s += E[C.idt][r][l] * u[IE + l];
            out[IE + r] = s;
        }
        for (int i = 0; i < NX; ++i) {
            real_t s = 0;
            for (int c = 0; c < NR; ++c) if (Us[i][c] != 0) s += Us[i][c] * vd[c];
            out[i] += s;
        }
    }

#if DFN_COV_UD
    // P (in U's storage, upper triangle) -> U D U^T in place
    void ud_factor() {
        for (int j = NX - 1; j >= 0; --j) {
            for (int i = j; i >= 0; --i) {
                real_t sg = U[i][j];
                for (int k = j + 1; k < NX; ++k) sg -= U[i][k] * D[k] * U[j][k];
                if (i == j) { D[j] = sg > jitter ? sg : jitter; U[j][j] = 1; }
                else U[i][j] = sg / D[j];
            }
        }
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < i; ++j) U[i][j] = 0;
    }
    // w^T P w
    real_t quad(const real_t* w) const {
        real_t s = 0;
        for (int j = 0; j < NX; ++j) {
            real_t fj = w[j];
            for (int i = 0; i < j; ++i) fj += U[i][j] * w[i];
            s += D[j] * fj * fj;
        }
        return s;
    }
    real_t P_ii(int i) const {
        real_t s = D[i];
        for (int j = i + 1; j < NX; ++j) s += D[j] * U[i][j] * U[i][j];
        return s;
    }
    // P = U D U^T (dense, host tests)
    void P_dense(double (*Pd)[NX]) const {
        for (int i = 0; i < NX; ++i)
            for (int j = i; j < NX; ++j) {
                double s = 0;
                for (int k = j; k < NX; ++k) s += (double)U[i][k] * (double)D[k] * (double)U[j][k];
                Pd[i][j] = s; Pd[j][i] = s;
            }
    }
    // Bierman: scalar measurement h (scaled), variance Rm; gain K; returns the innovation variance
    real_t ud_update(const real_t* h, real_t Rm, real_t* K) {
        real_t *fv = w_fv, *v = w_v, *b = w_b;
        for (int j = 0; j < NX; ++j) {
            real_t s = h[j];
            for (int i = 0; i < j; ++i) s += U[i][j] * h[i];
            fv[j] = s;
            v[j] = D[j] * s;
        }
        real_t alpha = Rm + v[0] * fv[0];
        D[0] = D[0] * Rm / alpha;
        b[0] = v[0];
        for (int j = 1; j < NX; ++j) {
            const real_t a0 = alpha;
            alpha += v[j] * fv[j];
            D[j] = D[j] * a0 / alpha;
            if (D[j] < jitter) D[j] = jitter;
            const real_t lam = -fv[j] / a0;
            for (int i = 0; i < j; ++i) {
                const real_t u = U[i][j];
                U[i][j] = u + lam * b[i];
                b[i] += u * v[j];
            }
            b[j] = v[j];
        }
        for (int i = 0; i < NX; ++i) K[i] = b[i] / alpha;
        return alpha;
    }
    // Thornton: P <- Phi P Phi^T + q_Rf e e^T via MWGS on W = [Phi U | e_Rf] (kept as W^T: rows = columns of W)
    void ud_predict(real_t qth) {
        real_t *t = w_t, *Dw = w_Dw;
        for (int i = 0; i < NX; ++i)                                 // U <- U^T: row k = column k of U
            for (int j = i + 1; j < NX; ++j) { real_t a = U[i][j]; U[i][j] = U[j][i]; U[j][i] = a; }
        for (int k = 0; k < NX; ++k) {                               // row k <- Phi (column k of U)
            phi_apply(U[k], t);
            for (int i = 0; i < NX; ++i) U[k][i] = t[i];
            Dw[k] = D[k];
        }
        for (int i = 0; i < NX; ++i) U[NX][i] = 0;
        U[NX][ITH] = 1;
        Dw[NX] = qth;
        for (int j = NX - 1; j >= 0; --j) {                          // MWGS from the last row of W
            real_t sg = 0;
            for (int k = 0; k <= NX; ++k) sg += U[k][j] * U[k][j] * Dw[k];
            D[j] = sg > jitter ? sg : jitter;                        // floor on D only: the projections use the true sg
            for (int i = 0; i < j; ++i) {
                real_t c = 0;
                if (sg > jitter) {
                    for (int k = 0; k <= NX; ++k) c += U[k][i] * Dw[k] * U[k][j];
                    c /= sg;
                }
                for (int k = 0; k <= NX; ++k) U[k][i] -= c * U[k][j];
                t[i] = c;
            }
            for (int i = 0; i < j; ++i) U[i][j] = t[i];             // column j of the new U (column j of W^T is dead)
            U[j][j] = 1;
            for (int k = j + 1; k < NX; ++k) U[k][j] = 0;
        }
    }
#endif

    // ---- linearization at the converged closure (structjac.Struct: lam_B, sens, sens_kp, dlaw_cols, C_of)
    void linearize(const real_t* z, real_t Rf_mult) {
        const int d = C.idt;
        cl.jacobian(g, in);
        bool ok = cl.factor();
        (void)ok;
        for (int o = 0; o < NO; ++o) {
            creal_t* b = Xo[o];
            for (int q = 0; q < NG; ++q) b[q] = 0;
            b[o < NEL ? o : NG - 1] = 1;
            cl.solve_lin_t(b);
        }
        for (int o = 0; o < NO; ++o) {
            const creal_t* X = Xo[o];
            for (int i = 0; i < NN; ++i) Swc[o][i] = (real_t)(-X[i] * cl.bn[i].dc);
            for (int i = 0; i < NP; ++i) {
                Swc[o][NN + i] = (real_t)(-X[NN + i] * cl.bp[i].dc);
                Ckp[o][i] = (real_t)(X[NN + i] * cl.bp[i].dc * g[NN + i] * (creal_t)C.dkp[i]);
            }
            creal_t yv[N];
            for (int k = 0; k < N; ++k) yv[k] = 0;
            for (int i = 0; i < NN; ++i) yv[i] += X[i] * cl.bn[i].dce;
            for (int i = 0; i < NP; ++i) yv[NN + NS + i] += X[NN + i] * cl.bp[i].dce;
            for (int i = 0; i < N - 1; ++i) {                          // the last electrolyte row is the charge balance
                creal_t xs = X[OPE + i] * (creal_t)s_e;
                if (i > 0) yv[i - 1] += xs * cl.De[i][0];
                yv[i] += xs * cl.De[i][1];
                if (i < N - 1) yv[i + 1] += xs * cl.De[i][2];
            }
            for (int k = 0; k < N; ++k) {
                creal_t s = 0;
                for (int l = 0; l < N; ++l) s += yv[l] * (creal_t)E[d][l][k];
                Swe[o][k] = (real_t)(-s);
            }
            creal_t st = 0;
            for (int i = 0; i < NN; ++i) st += X[i] * cl.bn[i].Gd * g[i];
            Cth[o] = (real_t)(-st * (creal_t)Rf0 * (creal_t)Rf_mult);
        }
        if (ent_dT != 0) {
            for (int i = 0; i < NN; ++i) Swc[NEL][i] += ent_dT * ent_dn / (cmax_n * NN);
            for (int i = 0; i < NP; ++i) Swc[NEL][NN + i] += ent_dT * ent_dp / (cmax_p * NP);
        }
        // Lambda diagonal
        for (int i = 0; i < NEL; ++i) {
            dvec[IC + i] = 1;
            for (int m = 0; m < NM; ++m) dvec[IA + m * NEL + i] = i < NN ? an_e[d][m] : C.ep[m][i - NN];
            for (int k = 0; k < K; ++k) dvec[IS + k * NEL + i] = i < NN ? an_rho[d][k] : C.rp[k][i - NN];
        }
        // U (scaled rows)
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NR; ++c) Us[r][c] = 0;
        for (int j = 0; j < NEL; ++j) {
            const bool an = j < NN;
            const int jp = j - NN;
            Us[IC + j][j] = -3 * C.dt / ((an ? Rn : Rp) * FARADAY);
            for (int m = 0; m < NM; ++m) Us[IA + m * NEL + j][j] = -(an ? an_gain[d][m] : C.gp[m][jp]) / FARADAY;
            for (int k = 0; k < K; ++k) {
                real_t rho = an ? an_rho[d][k] : C.rp[k][jp], W = an ? an_W[d][k] : C.Wp[k][jp];
                Us[IS + k * NEL + j][j] = -(1 - rho) * W / FARADAY;
            }
            for (int r = 0; r < N; ++r) Us[IE + r][j] = G[d][r][j];
        }
        for (int i = 0; i < NP; ++i) {
            const int node = NN + i;
            const real_t jF = (real_t)g[NN + i] / FARADAY;
            for (int m = 0; m < NM; ++m)
                Us[IA + m * NEL + node][NEL + i] = C.dep[m][i] * z[IA + m * NEL + node] - C.dgp[m][i] * jF;
            for (int k = 0; k < K; ++k)
                Us[IS + k * NEL + node][NEL + i] = C.drp[k][i] * z[IS + k * NEL + node]
                    - (-C.drp[k][i] * C.Wp[k][i] + (1 - C.rp[k][i]) * C.dWp[k][i]) * jF;
        }
        for (int r = 0; r < NX; ++r) {
            const real_t is = 1 / scale[r];
            for (int c = 0; c < NR; ++c) Us[r][c] *= is;
        }
        // V^T (scaled columns): rows j = dj/dx, rows NEL + i select c_avg,p
        for (int o = 0; o < NO; ++o) {
            real_t* row = o < NEL ? Vts[o] : hs;
            for (int i = 0; i < NEL; ++i) {
                real_t v = Swc[o][i];
                row[IC + i] = v + (i >= NN ? Ckp[o][i - NN] : (real_t)0);
                for (int m = 0; m < NM; ++m) row[IA + m * NEL + i] = v;
                for (int k = 0; k < K; ++k) row[IS + k * NEL + i] = v;
            }
            for (int k = 0; k < N; ++k) row[IE + k] = Swe[o][k];
            row[ITH] = Cth[o];
            for (int c = 0; c < NX; ++c) row[c] *= scale[c];
        }
        for (int i = 0; i < NP; ++i) {
            for (int c = 0; c < NX; ++c) Vts[NEL + i][c] = 0;
            Vts[NEL + i][IC + NN + i] = scale[IC + NN + i];
        }
    }

    // one filter step: prediction of V at the current state, update with y (if valid), propagation over DT[idt]
    // closure at the current state (warm start g), predicted V, f = Phi(x), and the linearization
    void predict_lin(real_t I, int idt, EkfOut& out) {
        real_t* z = w_z;
        for (int i = 0; i < NZ; ++i) z[i] = x[i];
        for (int i = 0; i < N; ++i) if (z[IE + i] < ce_floor) z[IE + i] = ce_floor;
        const real_t Rf_mult = r_exp(x[ITH]);
        step_consts(z, idt, C);
        closure_in(z, I, Rf0 * Rf_mult, C, in);
        out.res = (real_t)cl.solve(g, in);
        out.nit = cl.nit_last;
        out.V = (real_t)cl.V_of(g, in.I, in);
        advance(z, g, C, f);
        f[ITH] = x[ITH];
        linearize(z, Rf_mult);
    }

    // open loop with the filter's buffers: closure, V, x <- Phi(x) (theta held); for the board's replay
    EkfOut open_step(real_t I, int idt) {
        EkfOut out;
        real_t* z = w_z;
        for (int i = 0; i < NZ; ++i) z[i] = x[i];
        step_consts(z, idt, C);
        closure_in(z, I, Rf0 * r_exp(x[ITH]), C, in);
        out.res = (real_t)cl.solve(g, in);
        out.nit = cl.nit_last;
        out.V = (real_t)cl.V_of(g, in.I, in);
        advance(z, g, C, x);
        out.soc = soc_of_ybar(ybar(x));
        out.innov = out.Sy = out.sd_soc = out.sd_Rf = out.min_diag = 0;
        out.Rf = r_exp(x[ITH]);
        return out;
    }

    EkfOut step(real_t I, real_t y, int idt, bool valid = true) {
        EkfOut out;
        predict_lin(I, idt, out);
        out.innov = y - out.V;
        real_t* ws = w_ws;                                           // SOC read-out weights (scaled)
        for (int i = 0; i < NX; ++i) ws[i] = 0;
        for (int i = 0; i < NP; ++i) ws[IC + NN + i] = w_p[i] * scale[IC + NN + i];
#if DFN_COV_UD
        // ---- update (Bierman)
        if (valid) {
            real_t Sy = ud_update(hs, R, Kg);
            out.Sy = r_sqrt(Sy);
            for (int i = 0; i < NX; ++i) du[i] = Kg[i] * out.innov;
        } else {
            out.Sy = r_sqrt(quad(hs) + R);
            for (int i = 0; i < NX; ++i) du[i] = 0;
        }
        real_t vv = quad(ws), pth = P_ii(ITH), md = D[0];
        for (int i = 1; i < NX; ++i) if (D[i] < md) md = D[i];
#else
        // ---- update (rank 1, Joseph form written symmetric)
        real_t Sy = R;
        for (int i = 0; i < NX; ++i) {
            real_t s = 0;
            for (int j = 0; j < NX; ++j) s += P[i][j] * hs[j];
            Ph[i] = s;
        }
        for (int i = 0; i < NX; ++i) Sy += hs[i] * Ph[i];
        out.Sy = r_sqrt(Sy);
        if (valid) {
            for (int i = 0; i < NX; ++i) { Kg[i] = Ph[i] / Sy; du[i] = Kg[i] * out.innov; }
            for (int i = 0; i < NX; ++i)
                for (int j = i; j < NX; ++j) {
                    real_t v = P[i][j] - Kg[i] * Ph[j] - Ph[i] * Kg[j] + Sy * Kg[i] * Kg[j];
                    P[i][j] = v; P[j][i] = v;
                }
            for (int i = 0; i < NX; ++i) P[i][i] += jitter;
        } else {
            for (int i = 0; i < NX; ++i) du[i] = 0;
        }
        real_t vv = 0, pth = P[ITH][ITH], md = P[0][0];
        for (int i = 0; i < NX; ++i) {
            if (ws[i] == 0) continue;
            for (int j = 0; j < NX; ++j) vv += ws[i] * P[i][j] * ws[j];
        }
        for (int i = 1; i < NX; ++i) if (P[i][i] < md) md = P[i][i];
#endif
        // ---- outputs at the updated state
        real_t yb = 0;
        for (int i = 0; i < NP; ++i) yb += w_p[i] * (x[IC + NN + i] + scale[IC + NN + i] * du[IC + NN + i]);
        out.soc = soc_of_ybar(yb);
        out.sd_soc = r_sqrt(vv) * Q_p / Q_ref;
        out.Rf = r_exp(x[ITH] + scale[ITH] * du[ITH]);
        out.sd_Rf = out.Rf * r_sqrt(pth) * scale[ITH];
        out.min_diag = md;
        // ---- prediction
        const real_t qth = (Q_sd[ITH] / scale[ITH]) * (Q_sd[ITH] / scale[ITH]);
#if DFN_COV_UD
        ud_predict(qth);
#else
        (void)qth;
        for (int i = 0; i < NX; ++i)
            for (int c = 0; c < NR; ++c) {
                real_t s = 0;
                const real_t* v = Vts[c];
                for (int j = 0; j < NX; ++j) if (v[j] != 0) s += P[i][j] * v[j];
                PV[i][c] = s;
            }
        real_t Z[NR][NR];
        for (int a = 0; a < NR; ++a)
            for (int b = a; b < NR; ++b) {
                real_t s = 0;
                for (int j = 0; j < NX; ++j) s += Vts[a][j] * PV[j][b];
                Z[a][b] = s; Z[b][a] = s;
            }
        for (int i = 0; i < NX; ++i)
            for (int c = 0; c < NR; ++c) Wb[i][c] = PV[i][c];
        lam_rows(Wb);
        for (int i = 0; i < NX; ++i)
            for (int b = 0; b < NR; ++b) {
                const real_t u = Us[i][b];
                if (u == 0) continue;
                for (int c = 0; c < NR; ++c) Wb[i][c] += (real_t)0.5 * u * Z[b][c];
            }
        lam_P();
        for (int c = 0; c < NR; ++c)                                 // P += W U^T + U W^T over the non-zeros of U
            for (int s = 0; s < NX; ++s) {
                const real_t u = Us[s][c];
                if (u == 0) continue;
                for (int i = 0; i < NX; ++i) {
                    const real_t v = Wb[i][c] * u;
                    P[i][s] += v;
                    P[s][i] += v;
                }
            }
        for (int i = 0; i < NX; ++i) {
            const real_t q = Q_sd[i] / scale[i];
            P[i][i] += q * q;
        }
#endif
        // ---- state: x+ = f + scale Phi du
        real_t* ld = w_ld;
        phi_apply(du, ld);
        for (int i = 0; i < NX; ++i) x[i] = f[i] + scale[i] * ld[i];
        return out;
    }
};

}  // namespace dfn
