// Precision and small math helpers of the embedded DFN + EKF. No dynamic allocation, no STL.
//
//   DFN_DOUBLE          1: real_t = double (host reference); 0 (default): float (the board)
//   DFN_CLOSURE_DOUBLE  precision of the closure's Newton solve (default: as DFN_DOUBLE). The closure is the
//                       ill-conditioned part (soft stoichiometry edges); double there costs soft-float time on the M4F.
//   DFN_NEWTON_TOL      convergence of the closure, inf-norm of the scaled residual (default 1e-10 / 1e-4)
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef DFN_DOUBLE
#define DFN_DOUBLE 0
#endif
#ifndef DFN_CLOSURE_DOUBLE
#define DFN_CLOSURE_DOUBLE DFN_DOUBLE
#endif

#if DFN_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif
#if DFN_CLOSURE_DOUBLE
typedef double creal_t;
#else
typedef float creal_t;
#endif

// float: the residual floor is ~5e-5 (BV rows: phi_s,p - phi_e - U_p of ~4 V resolved to ~2e-7 V), so the tolerance
// sits above it; Newton also stops when it stagnates below DFN_NEWTON_ACCEPT, and the fallbacks of solve() run only
// above DFN_NEWTON_ACCEPT (double: = DFN_NEWTON_TOL, as dfnekf).
#ifndef DFN_NEWTON_TOL
#if DFN_CLOSURE_DOUBLE
#define DFN_NEWTON_TOL 1e-10
#else
#define DFN_NEWTON_TOL 1e-4
#endif
#endif
#ifndef DFN_NEWTON_ACCEPT
#if DFN_CLOSURE_DOUBLE
#define DFN_NEWTON_ACCEPT DFN_NEWTON_TOL
#else
#define DFN_NEWTON_ACCEPT 2e-4
#endif
#endif

namespace dfn {
inline float r_exp(float x) { return expf(x); }
inline double r_exp(double x) { return exp(x); }
inline float r_log(float x) { return logf(x); }
inline double r_log(double x) { return log(x); }
inline float r_log1p(float x) { return log1pf(x); }
inline double r_log1p(double x) { return log1p(x); }
inline float r_sqrt(float x) { return sqrtf(x); }
inline double r_sqrt(double x) { return sqrt(x); }
inline float r_sinh(float x) { return sinhf(x); }
inline double r_sinh(double x) { return sinh(x); }
inline float r_cosh(float x) { return coshf(x); }
inline double r_cosh(double x) { return cosh(x); }
inline float r_pow(float x, float y) { return powf(x, y); }
inline double r_pow(double x, double y) { return pow(x, y); }
inline float r_abs(float x) { return fabsf(x); }
inline double r_abs(double x) { return fabs(x); }
template <typename T> inline T r_max(T a, T b) { return a > b ? a : b; }
inline int r_exponent(float x) { int e = 0; frexpf(x, &e); return e; }
inline int r_exponent(double x) { int e = 0; frexp(x, &e); return e; }
inline float r_pow2(float, int e) { return ldexpf(1.0f, e); }
inline double r_pow2(double, int e) { return ldexp(1.0, e); }
template <typename T> inline T r_min(T a, T b) { return a < b ? a : b; }
template <typename T> inline bool r_finite(T x) { return x == x && x - x == 0; }
template <typename T> inline T softplus(T x) { return r_max(x, T(0)) + r_log1p(r_exp(-r_abs(x))); }
template <typename T> inline T sigmoid(T x) {
    if (x >= 0) { T e = r_exp(-x); return T(1) / (T(1) + e); }
    T e = r_exp(x); return e / (T(1) + e);
}
}  // namespace dfn
