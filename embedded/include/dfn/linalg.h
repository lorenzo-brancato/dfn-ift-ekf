// Dense LU with partial pivoting on fixed-size row-major buffers (no allocation).
#pragma once
#include "dfn/config.h"

namespace dfn {

// In place: P A = L U (L unit lower, U upper), piv[k] = row swapped with k at step k. False if a pivot is zero/non-finite.
template <typename T>
bool lu_factor(T* A, int n, int* piv) {
    for (int k = 0; k < n; ++k) {
        int p = k;
        T amax = r_abs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            T v = r_abs(A[i * n + k]);
            if (v > amax) { amax = v; p = i; }
        }
        piv[k] = p;
        if (!(amax > T(0)) || !r_finite(amax)) return false;
        if (p != k) {
            T* rk = A + k * n; T* rp = A + p * n;
            for (int j = 0; j < n; ++j) { T t = rk[j]; rk[j] = rp[j]; rp[j] = t; }
        }
        const T inv = T(1) / A[k * n + k];
        const T* rk = A + k * n;
        for (int i = k + 1; i < n; ++i) {
            T* ri = A + i * n;
            T l = ri[k] * inv;
            ri[k] = l;
            if (l != T(0))
                for (int j = k + 1; j < n; ++j) ri[j] -= l * rk[j];
        }
    }
    return true;
}

// A x = b (b overwritten by x)
template <typename T>
void lu_solve(const T* LU, const int* piv, int n, T* b) {
    for (int k = 0; k < n; ++k)
        if (piv[k] != k) { T t = b[k]; b[k] = b[piv[k]]; b[piv[k]] = t; }
    for (int i = 1; i < n; ++i) {
        T s = b[i]; const T* ri = LU + i * n;
        for (int j = 0; j < i; ++j) s -= ri[j] * b[j];
        b[i] = s;
    }
    for (int i = n - 1; i >= 0; --i) {
        T s = b[i]; const T* ri = LU + i * n;
        for (int j = i + 1; j < n; ++j) s -= ri[j] * b[j];
        b[i] = s / ri[i];
    }
}

// A^T x = b (b overwritten by x): U^T w = b, L^T v = w, x = P^T v
template <typename T>
void lu_solve_t(const T* LU, const int* piv, int n, T* b) {
    for (int i = 0; i < n; ++i) {
        T s = b[i];
        for (int j = 0; j < i; ++j) s -= LU[j * n + i] * b[j];
        b[i] = s / LU[i * n + i];
    }
    for (int i = n - 1; i >= 0; --i) {
        T s = b[i];
        for (int j = i + 1; j < n; ++j) s -= LU[j * n + i] * b[j];
        b[i] = s;
    }
    for (int k = n - 1; k >= 0; --k)
        if (piv[k] != k) { T t = b[k]; b[k] = b[piv[k]]; b[piv[k]] = t; }
}

// Row and column equilibration by powers of two (exact): A <- Dr A Dc with the largest entry of every row and column
// of order one. The closure's Jacobian has a condition number ~1e11 as built, ~4e6 equilibrated (float: eps 6e-8).
template <typename T>
void equilibrate(T* A, int n, T* dr, T* dc) {
    for (int i = 0; i < n; ++i) {
        T m = 0;
        for (int j = 0; j < n; ++j) m = r_max(m, r_abs(A[i * n + j]));
        const int e = m > 0 ? r_exponent(m) : 0;
        dr[i] = r_pow2(m, -e);
        for (int j = 0; j < n; ++j) A[i * n + j] *= dr[i];
    }
    for (int j = 0; j < n; ++j) {
        T m = 0;
        for (int i = 0; i < n; ++i) m = r_max(m, r_abs(A[i * n + j]));
        const int e = m > 0 ? r_exponent(m) : 0;
        dc[j] = r_pow2(m, -e);
        for (int i = 0; i < n; ++i) A[i * n + j] *= dc[j];
    }
}

template <typename T>
T norm_inf(const T* v, int n) {
    T m = 0;
    for (int i = 0; i < n; ++i) {
        T a = r_abs(v[i]);
        if (!(a <= m)) m = a;          // NaN propagates
    }
    return m;
}

template <typename T>
T norm2(const T* v, int n) {
    T s = 0;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return r_sqrt(s);
}

}  // namespace dfn
