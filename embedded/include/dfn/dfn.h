// Grey-box DFN of dfnekf (step.DFN, closure.Alg, particles) in portable C++: exact modal particles, exact ZOH
// electrolyte, implicit closure solved by Newton, and the pieces of the step Jacobian A = Lambda + B C (structjac).
//
// The model is a compile-time choice: DFN_MODEL_HEADER names a header written by embedded/export_model.py, e.g.
//   -DDFN_MODEL_HEADER='"lp604765_T20_10-6-10_m3k1/model.h"'
// State z = [c_avg (NEL), a (NM x NEL), s (K x NEL), c_e (N)], NEL = NN + NP (anode then cathode).
// Closure unknowns g = [j_n (NN), j_p (NP), phi_e (N), phi_s,n (NN), phi_s,p (NP)].
// Convention: y_k = V(z_{k-1}, I_k), z_k = Phi(z_{k-1}, I_k); I > 0 on discharge.
#pragma once
#include "dfn/config.h"
#include "dfn/linalg.h"
#ifndef DFN_MODEL_HEADER
#error "define DFN_MODEL_HEADER (a model.h written by embedded/export_model.py)"
#endif
#include DFN_MODEL_HEADER

namespace dfn {
using namespace dfn_model;

constexpr int NCP = 2 * NM + 2 * K;                       // cathode table functions: e, gain, rho, W
constexpr int IC = 0, IA = NEL, IS = NEL + NM * NEL, IE = NEL + NM * NEL + K * NEL;
constexpr int NZP = IE;                                   // particle states (Lambda diagonal)
constexpr int OPE = NEL, OSN = NEL + N, OSP = NEL + N + NN;   // offsets in g
constexpr int NO = NEL + 1;                               // closure outputs: j (NEL) and V
constexpr real_t TINY = (real_t)1e-6;

// ---------------------------------------------------------------- OCP tables
template <typename T>
inline T ocp_lookup(const real_t* U, int n, T x0, T dx, T s, T* dU) {
    T xmax = x0 + dx * T(n - 1);
    bool clip = false;
    if (s <= x0) { s = x0; clip = true; }
    if (s >= xmax) { s = xmax; clip = true; }
    int i = (int)((s - x0) / dx);
    if (i < 0) i = 0;
    if (i > n - 2) i = n - 2;
    T t = (s - (x0 + dx * T(i))) / dx;
    T u0 = (T)U[i], u1 = (T)U[i + 1];
    if (dU) *dU = clip ? T(0) : (u1 - u0) / dx;
    return u0 + t * (u1 - u0);
}
template <typename T> inline T Un(T x, T* dU = 0) { return ocp_lookup<T>(ocpn_U, NOCPN, (T)ocpn_x0, (T)ocpn_dx, x, dU); }
template <typename T> inline T Up(T y, T* dU = 0) { return ocp_lookup<T>(ocpp_U, NOCPP, (T)ocpp_x0, (T)ocpp_dx, y, dU); }

// ---------------------------------------------------------------- step constants
// Per step: the anode's scalars for the step length, the cathode's per node from the D_p(y) law at the step's initial
// mean stoichiometry (cubic Hermite in s), with their derivatives along c_avg,p (for the EKF).
struct StepConsts {
    int idt;
    real_t dt;
    real_t ep[NM][NP], gp[NM][NP], rp[K][NP], Wp[K][NP];
    real_t dep[NM][NP], dgp[NM][NP], drp[K][NP], dWp[K][NP];      // d / d c_avg,p
    real_t kn, kp[NP], dkp[NP];                                    // implicit closure: d c_surf / d j over the step
};

inline void step_consts(const real_t* z, int idt, StepConsts& C) {
    C.idt = idt;
    C.dt = DT[idt];
    real_t sk = 0;
    for (int m = 0; m < NM; ++m) sk += an_gain[idt][m];
    for (int k = 0; k < K; ++k) sk += (1 - an_rho[idt][k]) * an_W[idt][k];
    C.kn = (3 * C.dt / Rn + sk) / FARADAY;
    const real_t h = tab_smax / (real_t)(NTAB - 1);
    for (int i = 0; i < NP; ++i) {
        real_t y = z[IC + NN + i] / cmax_p;
        real_t u = (y - law_y0) / (law_y1 - law_y0), dsdy = 0, s;
        if (u > 0) {
            s = r_pow(u + (real_t)1e-12, law_shape);
            dsdy = law_shape * s / (u + (real_t)1e-12) / (law_y1 - law_y0);
        } else {
            s = r_pow((real_t)1e-12, law_shape);
        }
        if (s >= tab_smax) { s = tab_smax; dsdy = 0; }
        int j = (int)(s / h);
        if (j > NTAB - 2) j = NTAB - 2;
        if (j < 0) j = 0;
        real_t t = (s - h * (real_t)j) / h, t2 = t * t, t3 = t2 * t;
        real_t h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
        real_t g00 = (6 * t2 - 6 * t) / h, g10 = 3 * t2 - 4 * t + 1, g01 = (-6 * t2 + 6 * t) / h, g11 = 3 * t2 - 2 * t;
        const real_t dcd = dsdy / cmax_p;                          // ds / d c_avg,p
        real_t f[NCP], df[NCP];
        for (int q = 0; q < NCP; ++q) {
            real_t f0 = cp_val[idt][j][q], f1 = cp_val[idt][j + 1][q], d0 = cp_der[idt][j][q], d1 = cp_der[idt][j + 1][q];
            f[q] = h00 * f0 + h10 * h * d0 + h01 * f1 + h11 * h * d1;
            df[q] = (g00 * f0 + g10 * d0 + g01 * f1 + g11 * d1) * dcd;
        }
        real_t kk = 3 * C.dt / Rp, dk = 0;
        for (int m = 0; m < NM; ++m) {
            C.ep[m][i] = f[m]; C.dep[m][i] = df[m];
            C.gp[m][i] = f[NM + m]; C.dgp[m][i] = df[NM + m];
            kk += f[NM + m]; dk += df[NM + m];
        }
        for (int k = 0; k < K; ++k) {
            real_t rho = f[2 * NM + k], W = f[2 * NM + K + k], drho = df[2 * NM + k], dW = df[2 * NM + K + k];
            C.rp[k][i] = rho; C.drp[k][i] = drho; C.Wp[k][i] = W; C.dWp[k][i] = dW;
            kk += (1 - rho) * W; dk += -drho * W + (1 - rho) * dW;
        }
        C.kp[i] = kk / FARADAY; C.dkp[i] = dk / FARADAY;
    }
}

inline void surf(const real_t* z, real_t* csn, real_t* csp) {
    for (int i = 0; i < NEL; ++i) {
        real_t c = z[IC + i];
        for (int m = 0; m < NM; ++m) c += z[IA + m * NEL + i];
        for (int k = 0; k < K; ++k) c += z[IS + k * NEL + i];
        if (i < NN) csn[i] = c; else csp[i - NN] = c;
    }
}

// z+ = Phi(z, g) with the flux held over the step (particles.step, electrolyte E, G). z2 may not alias z.
inline void advance(const real_t* z, const creal_t* g, const StepConsts& C, real_t* z2) {
    const int d = C.idt;
    for (int i = 0; i < NEL; ++i) {
        const bool an = i < NN;
        const int ip = i - NN;
        const real_t jF = (real_t)g[i] / FARADAY;
        z2[IC + i] = z[IC + i] - 3 / (an ? Rn : Rp) * jF * C.dt;
        for (int m = 0; m < NM; ++m) {
            real_t e = an ? an_e[d][m] : C.ep[m][ip], gn = an ? an_gain[d][m] : C.gp[m][ip];
            z2[IA + m * NEL + i] = e * z[IA + m * NEL + i] - gn * jF;
        }
        for (int k = 0; k < K; ++k) {
            real_t rho = an ? an_rho[d][k] : C.rp[k][ip], W = an ? an_W[d][k] : C.Wp[k][ip];
            z2[IS + k * NEL + i] = rho * z[IS + k * NEL + i] - (1 - rho) * W * jF;
        }
    }
    for (int r = 0; r < N; ++r) {
        real_t s = 0;
        for (int c = 0; c < N; ++c) s += E[d][r][c] * z[IE + c];
        for (int c = 0; c < NEL; ++c) s += G[d][r][c] * (real_t)g[c];
        z2[IE + r] = s;
    }
}

// Total charge of the window read from the cathode: SOC = 1 - (ybar - y100) Q_p / Q_ref
inline real_t ybar(const real_t* z) {
    real_t s = 0;
    for (int i = 0; i < NP; ++i) s += w_p[i] * z[IC + NN + i];
    return s;
}
inline real_t soc_of_ybar(real_t yb) { return 1 - (yb - y100) * Q_p / Q_ref; }

// ---------------------------------------------------------------- closure
struct ClosureIn {
    creal_t csn[NN], csp[NP], ce[N];   // state's surface and electrolyte concentrations (start of the step)
    creal_t I, Rf;                     // current [A], film resistance [Ohm m2]
    const StepConsts* C;
};

// Electrode-node quantities of Butler-Volmer at the end-of-step concentrations
struct BV {
    creal_t j0, eta, Gd;               // Gd = d(2 j0 sinh(f eta))/d eta
    creal_t dc;                        // d r_bv / d c_surf (effective)
    creal_t dce;                       // d r_bv / d c_e (effective, at the node)
    creal_t r;                         // r_bv
};

class Closure {
public:
    creal_t J[NG * NG];
    int piv[NG];
    creal_t dr[NG], dc[NG];            // equilibration of J (factor() applies it, solve()/solve_t() undo it)
    creal_t X[NG * NO];                // J^T X = E_sel (sensitivities)
    creal_t cse_n[NN], cse_p[NP], cee[N];
    BV bn[NN], bp[NP];
    creal_t De[N][3];                  // d (Dm ie)_i / d c_e,{i-1,i,i+1}
    int nit_last;
    // phi_e and phi_s,p are solved as offsets from these references (re-centred every solve): in float, absolute
    // potentials of ~4 V cannot resolve the micro-volt differences across an electrode (sigma / dx ~ 3e7 S/m2)
    creal_t ref_pe, ref_sp;
    creal_t w_r[NG], w_dg[NG], w_gt[NG], w_rt[NG], w_gw[NG], w_gb[NG];   // work arrays (kept off the stack)

    // Soft stoichiometry edges (closure.Alg.bv_parts, soft_edges): lo = e sp(c / e), c_s = c_max - e sp((c_max - lo) / e).
    // c_s and the gap c_max - c_s are formed without cancellation: past c_max the gap is ~1e-3 mol/m3, below the
    // float resolution of c_max (~9e4 -> 0.008), and j0 ~ sqrt(c_s gap) would underflow to 0.
    static inline void soft(creal_t c, creal_t cm, creal_t e, creal_t* cs, creal_t* gap, creal_t* dcs) {
        const creal_t x = c / e;
        const creal_t l1 = r_log1p(r_exp(-r_abs(x)));
        const creal_t lo = (x > 0 ? c : creal_t(0)) + e * l1;               // e softplus(c / e)
        const creal_t dm = x > 0 ? (cm - c) - e * l1 : cm - lo;             // c_max - lo
        const creal_t arg = dm / e;
        const creal_t l2 = r_log1p(r_exp(-r_abs(arg)));
        if (arg > 0) { *gap = dm + e * l2; *cs = lo - e * l2; }
        else { *gap = e * l2; *cs = cm - *gap; }
        *dcs = sigmoid(arg) * sigmoid(x);
    }

    // concentrations the closure sees (implicit coupling) and the BV terms; electrolyte derivative De
    void eval(const creal_t* g, const ClosureIn& in, bool derivs) {
        const StepConsts& C = *in.C;
        const int d = C.idt;
        for (int i = 0; i < NN; ++i) cse_n[i] = in.csn[i] - (creal_t)C.kn * g[i];
        for (int i = 0; i < NP; ++i) cse_p[i] = in.csp[i] - (creal_t)C.kp[i] * g[NN + i];
        for (int r = 0; r < N; ++r) {
            creal_t s = 0;
            for (int c = 0; c < N; ++c) s += (creal_t)E[d][r][c] * in.ce[c];
            for (int c = 0; c < NEL; ++c) s += (creal_t)G[d][r][c] * g[c];
            cee[r] = s;
        }
        const creal_t f = (creal_t)f_RT, al = (creal_t)alpha;
        for (int e = 0; e < 2; ++e) {
            const int n = e == 0 ? NN : NP;
            const creal_t cm = e == 0 ? (creal_t)cmax_n : (creal_t)cmax_p;
            const creal_t ew = e == 0 ? (creal_t)soft_e_n : (creal_t)soft_e_p;
            const creal_t k0 = e == 0 ? (creal_t)k0_n : (creal_t)k0_p;
            for (int i = 0; i < n; ++i) {
                BV& b = e == 0 ? bn[i] : bp[i];
                const creal_t c = e == 0 ? cse_n[i] : cse_p[i];
                const int ie = e == 0 ? i : NN + NS + i;
                const creal_t jj = e == 0 ? g[i] : g[NN + i];
                const creal_t phis = e == 0 ? g[OSN + i] : g[OSP + i];
                creal_t cs, gap, dcs;
                soft(c, cm, ew, &cs, &gap, &dcs);
                creal_t ceb = cee[ie] > (creal_t)TINY ? cee[ie] : (creal_t)TINY;
                creal_t prod = ceb * cs * gap;
                b.j0 = k0 * r_pow(prod, al);
                creal_t dU;
                creal_t U = e == 0 ? Un<creal_t>(cs / cm, &dU) : Up<creal_t>(cs / cm, &dU);
                b.eta = e == 0 ? (phis - g[OPE + ie]) - (ref_pe + U) - in.Rf * jj
                               : (phis - g[OPE + ie]) + ((ref_sp - ref_pe) - U);
                creal_t sh = r_sinh(f * b.eta), ch = r_cosh(f * b.eta);
                b.r = jj - 2 * b.j0 * sh;
                b.Gd = 2 * f * b.j0 * ch;
                if (derivs) {
                    creal_t dj0 = b.j0 * al * (1 / cs - 1 / gap) * dcs;
                    creal_t deta = -dU / cm * dcs;
                    b.dc = -2 * (dj0 * sh + b.j0 * ch * f * deta);
                    b.dce = cee[ie] > (creal_t)TINY ? -2 * sh * b.j0 * al / cee[ie] : creal_t(0);
                }
            }
        }
        if (derivs) {
            const creal_t chi_ = (creal_t)chi;
            creal_t a[N], bb[N];                  // d ie_k / d ce_k, d ie_k / d ce_{k+1}
            for (int k = 0; k < N - 1; ++k) {
                creal_t wl = (creal_t)wl_c[k], m = wl * cee[k] + (1 - wl) * cee[k + 1], inv = 1 / m;
                creal_t dce = cee[k + 1] - cee[k], kc = (creal_t)Kd[k] * chi_;
                a[k] = kc * (-inv - dce * wl * inv * inv);
                bb[k] = kc * (inv - dce * (1 - wl) * inv * inv);
            }
            for (int i = 0; i < N; ++i) {
                De[i][0] = i > 0 ? -a[i - 1] : creal_t(0);
                De[i][1] = (i < N - 1 ? a[i] : creal_t(0)) - (i > 0 ? bb[i - 1] : creal_t(0));
                De[i][2] = i < N - 1 ? bb[i] : creal_t(0);
            }
        }
    }

    void residual(const creal_t* g, const ClosureIn& in, creal_t* r) {
        eval(g, in, false);
        const creal_t* jn = g; const creal_t* jp = g + NN; const creal_t* pe = g + OPE;
        const creal_t* sn = g + OSN; const creal_t* sp = g + OSP;
        for (int i = 0; i < NN; ++i) r[i] = bn[i].r;
        for (int i = 0; i < NP; ++i) r[NN + i] = bp[i].r;
        const creal_t chi_ = (creal_t)chi;
        creal_t fl_prev = 0, ie_prev = 0, qe = 0;
        for (int i = 0; i < N; ++i) {
            creal_t fl = 0, iec = 0;
            if (i < N - 1) {
                creal_t wl = (creal_t)wl_c[i];
                fl = (creal_t)Kd[i] * (pe[i + 1] - pe[i]);
                iec = (creal_t)Kd[i] * chi_ / (wl * cee[i] + (1 - wl) * cee[i + 1]) * (cee[i + 1] - cee[i]);
            }
            creal_t aj = 0;
            if (i < NN) aj = (creal_t)a_n * jn[i] * (creal_t)dxn[i];
            else if (i >= NN + NS) aj = (creal_t)a_p * jp[i - NN - NS] * (creal_t)dxp[i - NN - NS];
            r[OPE + i] = (-(fl - fl_prev) + (iec - ie_prev) - aj) * (creal_t)s_e;
            fl_prev = fl; ie_prev = iec;
            qe += aj;
        }
        // electrolyte: the last row is the sum of all, -(Q_n + Q_p): the cell's charge balance (no phi_e, no c_e)
        r[OPE + N - 1] = -qe * sq_n();
        for (int i = 0; i < NN; ++i) {
            creal_t s = i == 0 ? (creal_t)Bn_dir * sn[0] : creal_t(0);
            if (i > 0) s += (creal_t)Bn_c[i - 1] * (sn[i] - sn[i - 1]);
            if (i < NN - 1) s += (creal_t)Bn_c[i] * (sn[i] - sn[i + 1]);
            r[OSN + i] = (s + (creal_t)a_n * jn[i] * (creal_t)dxn[i]) * (creal_t)s_sn;
        }
        // cathode solid phase: the last row is the sum of all (the cathode's charge balance, free of sigma / dx). Same
        // solution; the uniform phi_s,p direction is then well conditioned (equilibrated cond ~4e6 -> ~7e3).
        // Both balance rows are scaled to a mean reaction current density [A/m2], so that the Newton tolerance bounds
        // the charge error itself (the potential rows, scaled to ~V, allow ~1e4 A/m2 per unit of residual).
        creal_t qsum = in.I / (creal_t)area;
        for (int i = 0; i < NP; ++i) {
            creal_t s = 0;
            if (i > 0) s += (creal_t)Bp_c[i - 1] * (sp[i] - sp[i - 1]);
            if (i < NP - 1) s += (creal_t)Bp_c[i] * (sp[i] - sp[i + 1]);
            creal_t aj = (creal_t)a_p * jp[i] * (creal_t)dxp[i];
            qsum += aj;
            r[OSP + i] = (s + aj) * (creal_t)s_sp;
        }
        r[NG - 1] = qsum * sq_p();
    }
    static creal_t sq_n() { return creal_t(1) / ((creal_t)a_n * (creal_t)Ln); }
    static creal_t sq_p() { return creal_t(1) / ((creal_t)a_p * (creal_t)Lp); }

    // dense Jacobian dF/dg (scaled as the residual), implicit coupling included
    void jacobian(const creal_t* g, const ClosureIn& in) {
        eval(g, in, true);
        const StepConsts& C = *in.C;
        const int d = C.idt;
        for (int q = 0; q < NG * NG; ++q) J[q] = 0;
        creal_t* Jr;
        for (int i = 0; i < NN; ++i) {
            Jr = J + i * NG;
            Jr[i] = 1 + bn[i].Gd * in.Rf - bn[i].dc * (creal_t)C.kn;
            Jr[OPE + i] = bn[i].Gd;
            Jr[OSN + i] = -bn[i].Gd;
        }
        for (int i = 0; i < NP; ++i) {
            Jr = J + (NN + i) * NG;
            Jr[NN + i] = 1 - bp[i].dc * (creal_t)C.kp[i];
            Jr[OPE + NN + NS + i] = bp[i].Gd;
            Jr[OSP + i] = -bp[i].Gd;
        }
        const creal_t se = (creal_t)s_e;
        for (int i = 0; i < N; ++i) {
            Jr = J + (OPE + i) * NG;
            if (i == N - 1) {                                         // charge-balance row
                for (int q = 0; q < NN; ++q) Jr[q] = -(creal_t)a_n * (creal_t)dxn[q] * sq_n();
                for (int q = 0; q < NP; ++q) Jr[NN + q] = -(creal_t)a_p * (creal_t)dxp[q] * sq_n();
                continue;
            }
            creal_t kl = i > 0 ? (creal_t)Kd[i - 1] : creal_t(0), kr = i < N - 1 ? (creal_t)Kd[i] : creal_t(0);
            Jr[OPE + i] = (kl + kr) * se;
            if (i > 0) Jr[OPE + i - 1] = -kl * se;
            if (i < N - 1) Jr[OPE + i + 1] = -kr * se;
            if (i < NN) Jr[i] = -(creal_t)a_n * (creal_t)dxn[i] * se;
            else if (i >= NN + NS) Jr[i - NS] = -(creal_t)a_p * (creal_t)dxp[i - NN - NS] * se;
        }
        for (int i = 0; i < NN; ++i) {
            Jr = J + (OSN + i) * NG;
            creal_t cl_ = i > 0 ? (creal_t)Bn_c[i - 1] : creal_t(0), cr_ = i < NN - 1 ? (creal_t)Bn_c[i] : creal_t(0);
            Jr[OSN + i] = (cl_ + cr_ + (i == 0 ? (creal_t)Bn_dir : creal_t(0))) * (creal_t)s_sn;
            if (i > 0) Jr[OSN + i - 1] = -cl_ * (creal_t)s_sn;
            if (i < NN - 1) Jr[OSN + i + 1] = -cr_ * (creal_t)s_sn;
            Jr[i] = (creal_t)a_n * (creal_t)dxn[i] * (creal_t)s_sn;
        }
        for (int i = 0; i < NP; ++i) {
            Jr = J + (OSP + i) * NG;
            Jr[NG * 0 + NN + i] = (creal_t)a_p * (creal_t)dxp[i] * (creal_t)s_sp;
            J[(NG - 1) * NG + NN + i] = (creal_t)a_p * (creal_t)dxp[i] * sq_p();             // charge-balance row
            if (i == NP - 1) continue;
            creal_t cl_ = i > 0 ? (creal_t)Bp_c[i - 1] : creal_t(0), cr_ = i < NP - 1 ? (creal_t)Bp_c[i] : creal_t(0);
            Jr[OSP + i] = (cl_ + cr_) * (creal_t)s_sp;
            if (i > 0) Jr[OSP + i - 1] = -cl_ * (creal_t)s_sp;
            if (i < NP - 1) Jr[OSP + i + 1] = -cr_ * (creal_t)s_sp;
        }
        // c_e,eff = E c_e + G j: J[:, :NEL] += (dF/dc_e) G
        for (int j = 0; j < NEL; ++j) {
            for (int i = 0; i < NN; ++i) J[i * NG + j] += bn[i].dce * (creal_t)G[d][i][j];
            for (int i = 0; i < NP; ++i) J[(NN + i) * NG + j] += bp[i].dce * (creal_t)G[d][NN + NS + i][j];
            for (int i = 0; i < N - 1; ++i) {                          // (the balance row has no c_e)
                creal_t s = De[i][1] * (creal_t)G[d][i][j];
                if (i > 0) s += De[i][0] * (creal_t)G[d][i - 1][j];
                if (i < N - 1) s += De[i][2] * (creal_t)G[d][i + 1][j];
                J[(OPE + i) * NG + j] += s * se;
            }
        }
    }

    // LU of the equilibrated Jacobian; J x = b and J^T x = b with it
    bool factor() {
        equilibrate(J, NG, dr, dc);
        return lu_factor(J, NG, piv);
    }
    void solve_lin(creal_t* b) const {
        for (int i = 0; i < NG; ++i) b[i] *= dr[i];
        lu_solve(J, piv, NG, b);
        for (int i = 0; i < NG; ++i) b[i] *= dc[i];
    }
    void solve_lin_t(creal_t* b) const {
        for (int i = 0; i < NG; ++i) b[i] *= dc[i];
        lu_solve_t(J, piv, NG, b);
        for (int i = 0; i < NG; ++i) b[i] *= dr[i];
    }

    // ---- initial guess and solvers (closure.Alg.guess / newton / newton_plain / solve_robust)
    // initial guess relative to the current references (init_refs first on a cold start)
    void guess(const ClosureIn& in, creal_t I, creal_t* g) const {
        const creal_t icc = I / (creal_t)area;
        creal_t un = 0, up = 0;
        for (int i = 0; i < NN; ++i) un += Un<creal_t>(in.csn[i] / (creal_t)cmax_n);
        for (int i = 0; i < NP; ++i) up += Up<creal_t>(in.csp[i] / (creal_t)cmax_p);
        un /= NN; up /= NP;
        for (int i = 0; i < NN; ++i) g[i] = icc / ((creal_t)a_n * (creal_t)Ln);
        for (int i = 0; i < NP; ++i) g[NN + i] = -icc / ((creal_t)a_p * (creal_t)Lp);
        for (int i = 0; i < N; ++i) g[OPE + i] = -un - ref_pe;
        for (int i = 0; i < NN; ++i) g[OSN + i] = 0;
        for (int i = 0; i < NP; ++i) g[OSP + i] = (up - un) - ref_sp;
    }
    void init_refs(const ClosureIn& in) {
        creal_t un = 0, up = 0;
        for (int i = 0; i < NN; ++i) un += Un<creal_t>(in.csn[i] / (creal_t)cmax_n);
        for (int i = 0; i < NP; ++i) up += Up<creal_t>(in.csp[i] / (creal_t)cmax_p);
        un /= NN; up /= NP;
        ref_pe = -un; ref_sp = up - un;
    }
    // move the references to phi_e(0) and phi_s,p at the collector of the current solution
    void recenter(creal_t* g) {
        creal_t d = g[OPE];
        ref_pe += d;
        for (int i = 0; i < N; ++i) g[OPE + i] -= d;
        d = g[NG - 1];
        ref_sp += d;
        for (int i = 0; i < NP; ++i) g[OSP + i] -= d;
    }
    // absolute potentials <-> offsets (tests, golden comparison)
    void load_abs(const double* ga, creal_t* g) {
        ref_pe = (creal_t)ga[OPE]; ref_sp = (creal_t)ga[NG - 1];
        for (int q = 0; q < NG; ++q) {
            double v = ga[q];
            if (q >= OPE && q < OPE + N) v -= ga[OPE];
            if (q >= OSP) v -= ga[NG - 1];
            g[q] = (creal_t)v;
        }
    }
    double abs_of(const creal_t* g, int q) const {
        double v = (double)g[q];
        if (q >= OPE && q < OPE + N) v += (double)ref_pe;
        if (q >= OSP) v += (double)ref_sp;
        return v;
    }

    // Newton with backtracking; returns ||F||_inf, iterations in *nit
    creal_t newton(creal_t* g, const ClosureIn& in, creal_t tol, int maxit, int* nit) {
        creal_t *r = w_r, *dg = w_dg, *gt = w_gt, *rt = w_rt;
        residual(g, in, r);
        creal_t rn = norm_inf(r, NG);
        int k = 0;
        while (!(rn <= tol) && k < maxit) {
            jacobian(g, in);
            for (int q = 0; q < NG; ++q) dg[q] = -r[q];
            if (!factor()) { rn = creal_t(NAN); break; }
            solve_lin(dg);
            creal_t n0 = norm2(r, NG), a = 1;
            bool ok = false;
            while (!ok && a > (creal_t)1e-6) {
                for (int q = 0; q < NG; ++q) gt[q] = g[q] + a * dg[q];
                residual(gt, in, rt);
                creal_t nt = norm2(rt, NG);
                if (r_finite(nt) && nt < n0 * (1 - (creal_t)1e-4 * a)) ok = true;
                else a *= creal_t(0.5);
            }
            if (ok) {
                for (int q = 0; q < NG; ++q) { g[q] = gt[q]; r[q] = rt[q]; }
            } else {
                for (int q = 0; q < NG; ++q) g[q] += dg[q];
                residual(g, in, r);
            }
            creal_t rp = rn;
            rn = norm_inf(r, NG);
            ++k;
#ifdef DFN_DEBUG_NEWTON
            {
                int im = 0;
                for (int q = 1; q < NG; ++q) if (r_abs(r[q]) > r_abs(r[im])) im = q;
                printf("    newton it %d: |F| %.3e -> %.3e (row %d)  step a %.3g %s\n", k, (double)rp, (double)rn, im,
                       (double)a, ok ? "" : "(no descent: full step)");
            }
#endif
            if (rn <= (creal_t)DFN_NEWTON_ACCEPT && !(rn < creal_t(0.9) * rp)) break;    // stagnation at the noise floor
        }
        *nit = k;
        return rn;
    }

    creal_t newton_plain(creal_t* g, const ClosureIn& in, creal_t tol, int maxit, int* nit) {
        creal_t *r = w_r, *dg = w_dg;
        residual(g, in, r);
        creal_t rn = norm_inf(r, NG);
        int k = 0;
        while (rn > tol && r_finite(rn) && k < maxit) {
            jacobian(g, in);
            for (int q = 0; q < NG; ++q) dg[q] = -r[q];
            if (!factor()) { rn = creal_t(NAN); break; }
            solve_lin(dg);
            for (int q = 0; q < NG; ++q) g[q] += dg[q];
            residual(g, in, r);
            rn = norm_inf(r, NG);
            ++k;
        }
        *nit = k;
        return rn;
    }

    static bool better(creal_t a, creal_t b) { return r_finite(a) && !(a >= b); }

    // warm start; undamped Newton from the warm start, cold start, continuation in I as fallbacks (soft_edges branch)
    creal_t solve(creal_t* g, const ClosureIn& in, creal_t tol = (creal_t)DFN_NEWTON_TOL) {
        creal_t *gw = w_gw, *gb = w_gb;
        recenter(g);
        for (int q = 0; q < NG; ++q) gw[q] = g[q];
        int k;
        creal_t r = newton(g, in, tol, 60, &k);
        nit_last = k;
        const creal_t acc = tol > (creal_t)DFN_NEWTON_ACCEPT ? tol : (creal_t)DFN_NEWTON_ACCEPT;
        if (r <= acc) return r;
        for (int q = 0; q < NG; ++q) gb[q] = gw[q];
        creal_t r0 = newton_plain(gb, in, tol, 30, &k);
        if (better(r0, r)) { r = r0; for (int q = 0; q < NG; ++q) g[q] = gb[q]; }
        nit_last += k;
        if (r <= acc) return r;
        guess(in, in.I, gb);
        creal_t r1 = newton(gb, in, tol, 60, &k);
        nit_last += k;
        if (better(r1, r)) { r = r1; for (int q = 0; q < NG; ++q) g[q] = gb[q]; }
        if (r <= acc) return r;
        ClosureIn in2 = in;
        guess(in, 0, gb);
        const creal_t sc[4] = {0.05f, 0.2f, 0.5f, 1.0f};
        creal_t r2 = 0;
        for (int s = 0; s < 4; ++s) {
            in2.I = sc[s] * in.I;
            r2 = newton(gb, in2, tol, 60, &k);
            nit_last += k;
        }
        if (better(r2, r)) { r = r2; for (int q = 0; q < NG; ++q) g[q] = gb[q]; }
        return r;
    }

    creal_t V_of(const creal_t* g, creal_t I, const ClosureIn& in) const {
        creal_t V = (ref_sp + g[NG - 1]) - creal_t(0.5) * (creal_t)dxp[NP - 1] * (I / (creal_t)area) / (creal_t)sig_p - I * (creal_t)R_contact;
        if (ent_dT != 0) {
            creal_t mn = 0, mp = 0;
            for (int i = 0; i < NN; ++i) mn += in.csn[i];
            for (int i = 0; i < NP; ++i) mp += in.csp[i];
            mn /= (creal_t)(NN * cmax_n); mp /= (creal_t)(NP * cmax_p);
            V += (creal_t)ent_dT * ((creal_t)ent_d0 + (creal_t)ent_dn * (mn - creal_t(0.5)) + (creal_t)ent_dp * (mp - creal_t(0.5)));
        }
        return V;
    }
};

inline void closure_in(const real_t* z, real_t I, real_t Rf, const StepConsts& C, ClosureIn& in) {
    real_t csn[NN], csp[NP];
    surf(z, csn, csp);
    for (int i = 0; i < NN; ++i) in.csn[i] = (creal_t)csn[i];
    for (int i = 0; i < NP; ++i) in.csp[i] = (creal_t)csp[i];
    for (int i = 0; i < N; ++i) in.ce[i] = (creal_t)z[IE + i];
    in.I = (creal_t)I; in.Rf = (creal_t)Rf; in.C = &C;
}

// ---------------------------------------------------------------- open loop
class OpenLoop {
public:
    real_t z[NZ];
    creal_t g[NG];
    StepConsts C;
    ClosureIn in;
    Closure cl;
    bool started;
    real_t Rf_mult;
    creal_t res_last;

    void reset(const real_t* z0) {
        for (int i = 0; i < NZ; ++i) z[i] = z0[i];
        started = false;
        Rf_mult = 1;
    }
    // one step of length DT[idt] under the current I; returns the voltage at the start of the step (y_k)
    real_t step(real_t I, int idt) {
        step_consts(z, idt, C);
        closure_in(z, I, Rf0 * Rf_mult, C, in);
        if (!started) {
            cl.init_refs(in);
            cl.guess(in, in.I, g);
            int k;
            res_last = cl.newton(g, in, (creal_t)DFN_NEWTON_TOL, 60, &k);
            started = true;
        }
        res_last = cl.solve(g, in);
        real_t V = (real_t)cl.V_of(g, in.I, in);
        real_t z2[NZ];
        advance(z, g, C, z2);
        for (int i = 0; i < NZ; ++i) z[i] = z2[i];
        return V;
    }
};

}  // namespace dfn
