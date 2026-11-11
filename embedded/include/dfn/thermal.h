// Temperature of the model at run time. The header holds the model at one temperature (export --T); with
// DFN_MODEL_THERMAL it also holds the temperature-dependent constants on a grid TG (export --T-grid), as
// cellset.params_for builds them: Arrhenius k0 (both electrodes), R_contact, D_s,n (the anode particle constants),
// D_s,p (a shift of the cathode tables' axis s: log10 D is linear in s), kappa (a factor on the electrolyte
// conductances), D_e (E, G), F / 2RT and chi; the entropic term follows dT = T - T_ref.
// dfn::TH starts at the header's temperature; TH.set(T_degC) interpolates linearly on the grid (clamped at its ends).
// Cost: ~(N^2 + N NEL) NDT multiply-adds, so call it when the temperature changes (it skips changes below 0.05 degC).
#pragma once
#include "dfn/config.h"

#ifndef DFN_MODEL_THERMAL
namespace dfn_model {
static const real_t tab_s0 = 0;
}
#endif

namespace dfn {
using namespace dfn_model;

struct Thermal {
    real_t Tc;                                  // [degC]
    real_t k0n, k0p, Rc, fRT, chi, kscale, sshift, entdT;
    real_t E[NDT][N][N], G[NDT][N][NEL];
    real_t an_e[NDT][NM], an_gain[NDT][NM], an_rho[NDT][K], an_W[NDT][K];
    bool tables() const {
#ifdef DFN_MODEL_THERMAL
        return true;
#else
        return false;
#endif
    }
    void reset() {
        Tc = T - (real_t)273.15;
        k0n = k0_n; k0p = k0_p; Rc = R_contact; fRT = f_RT; chi = dfn_model::chi; kscale = 1; sshift = 0; entdT = ent_dT;
        for (int d = 0; d < NDT; ++d) {
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) E[d][r][c] = dfn_model::E[d][r][c];
                for (int c = 0; c < NEL; ++c) G[d][r][c] = dfn_model::G[d][r][c];
            }
            for (int m = 0; m < NM; ++m) { an_e[d][m] = dfn_model::an_e[d][m]; an_gain[d][m] = dfn_model::an_gain[d][m]; }
            for (int k = 0; k < K; ++k) { an_rho[d][k] = dfn_model::an_rho[d][k]; an_W[d][k] = dfn_model::an_W[d][k]; }
        }
    }
    // returns false when the header has no tables (the model stays at its temperature)
    bool set(real_t T_degC) {
#ifdef DFN_MODEL_THERMAL
        if (!(T_degC == T_degC)) return false;
        if (r_abs(T_degC - Tc) < (real_t)0.05) return true;
        real_t t = T_degC;
        if (t < TG[0]) t = TG[0];
        if (t > TG[NTG - 1]) t = TG[NTG - 1];
        int i = (int)((t - TG[0]) / (TG[1] - TG[0]));
        if (i > NTG - 2) i = NTG - 2;
        if (i < 0) i = 0;
        const real_t w = (t - TG[i]) / (TG[i + 1] - TG[i]), v = 1 - w;
        auto L = [&](const real_t* a) { return v * a[i] + w * a[i + 1]; };
        k0n = L(th_k0_n); k0p = L(th_k0_p); Rc = L(th_R_contact); fRT = L(th_f_RT); chi = L(th_chi);
        kscale = L(th_kscale); sshift = L(th_sshift);
        entdT = (t + (real_t)273.15) - ent_Tref;
        for (int d = 0; d < NDT; ++d) {
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) E[d][r][c] = v * th_E[i][d][r][c] + w * th_E[i + 1][d][r][c];
                for (int c = 0; c < NEL; ++c) G[d][r][c] = v * th_G[i][d][r][c] + w * th_G[i + 1][d][r][c];
            }
            for (int m = 0; m < NM; ++m) {
                an_e[d][m] = v * th_an_e[i][d][m] + w * th_an_e[i + 1][d][m];
                an_gain[d][m] = v * th_an_gain[i][d][m] + w * th_an_gain[i + 1][d][m];
            }
            for (int k = 0; k < K; ++k) {
                an_rho[d][k] = v * th_an_rho[i][d][k] + w * th_an_rho[i + 1][d][k];
                an_W[d][k] = v * th_an_W[i][d][k] + w * th_an_W[i + 1][d][k];
            }
        }
        Tc = T_degC;
        return true;
#else
        (void)T_degC;
        return false;
#endif
    }
    Thermal() { reset(); }
};

inline Thermal TH;

}  // namespace dfn
