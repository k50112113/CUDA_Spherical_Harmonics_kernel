// Reference
// https://github.com/NVIDIA/torch-harmonics
// Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.

#include <cmath>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
const uint max_allowed_n = 13;

__host__
void compute_associated_legendre_polynomial_coefficients(
  uint n,
  vector<int>& a_n_m,
  vector<int>& b_n_m)
{
  // precompute the coefficients for associated legendre polynomials on host
  a_n_m.resize((n + 1) * (n + 1), 0.0);
  b_n_m.resize((n + 1) * (n + 1), 0.0);
  a_n_m[0] = 1.0;
  for (uint l = 1; l <= n; ++l) {
    for (uint m = 0; m < l-1; ++m) {
      a_n_m[l * (n + 1) + m] = sqrt((2.0*l - 1) / (l - m) * (2.0*l + 1) / (l + m));
      b_n_m[l * (n + 1) + m] = sqrt(((float)l + m - 1) / (l - m) * (2.0*l + 1) / (2.0*l - 3) * (l - m - 1) / (l + m));
    }
    a_n_m[l * (n + 1) + l - 1] = sqrt(2.0*l + 1);
    a_n_m[l * (n + 1) + l]     = sqrt((2.0*l + 1) / 2.0 / l);
    // the following is the true value for a_n_m(l, l) but the cumulative products are done during the generation process of the polynomials
    // a_n_m[l * (n + 1) + l]  = sqrt((2.0*l + 1) / 2.0 / l) * a_n_m[(l - 1) * (n + 1) + l - 1];
  }
}

__device__
void compute_associated_legendre_polynomial_kernel(
  uint n,
  const float* __restrict__ a_n_m,
  const float* __restrict__ b_n_m,
  float* p_m,
  float x)
{
  // return Pnm at selected n from m = 0 ~ n on device
  float p_m_last_three_rows[3*max_allowed_n];
  p_m_last_three_rows[0] = 1.0;
  for (uint l = 1; l <= n; ++l) {
    uint l0 = l % 3;
    uint l1 = (l-1) % 3;
    uint l2 = (l-2) % 3;
    for (uint m = 0; m < l-1; ++m) {
      p_m_last_three_rows[l0 * max_allowed_n + m] = x * a_n_m[l * (n + 1) + m] * p_m_last_three_rows[l1 * max_allowed_n + m] -
                                                                 b_n_m[l * (n + 1) + m] * p_m_last_three_rows[l2 * max_allowed_n + m];
    }
    p_m_last_three_rows[l0 * max_allowed_n + l-1] =                           x * a_n_m[l * (n + 1) + l-1] * p_m_last_three_rows[l1 * max_allowed_n + l-1];                     
    p_m_last_three_rows[l0 * max_allowed_n + l  ] = sqrt((1.0 + x) * (1.0 - x)) * a_n_m[l * (n + 1) + l  ] * p_m_last_three_rows[l1 * max_allowed_n + l-1];
  }
  for (uint m = 0; m <= n; ++m) {
    p_m[m] = p_m_last_three_rows[(n % 3) * max_allowed_n + m];
  }
  if (n + 1 >= 2) {
    for (uint m = 0; m <= n; ++m) {
      p_m[m] /= a_n_m[n * (n + 1) + n-1];
    }
  }
  for (uint m = 1; m <= n; m+=2){
    p_m[m] *= -1.0;
  }
}

__device__
void compute_spherical_harmonics_kernel(
  uint n,
  const float* __restrict__ a_n_m,
  const float* __restrict__ b_n_m,
  float theta,
  float phi,
  float* Y_m_real,
  float* Y_m_imag)
{
  // return Ynm at selected n from m = 0 ~ n on device
  // theta [0, pi]
  // phi   [0, 2pi)
  float p_m[max_allowed_n];
  compute_associated_legendre_polynomial_kernel(n, a_n_m, b_n_m, p_m, cos(theta));
  float pre_factor = sqrt((2.0*n + 1) / (4.0 * M_PI));
  for (uint i_m = 0; i_m <= n; ++i_m) {
    atomicAdd(&Y_m_real[i_m], pre_factor * p_m[i_m] * cos(i_m * phi));
    atomicAdd(&Y_m_imag[i_m], pre_factor * p_m[i_m] * sin(i_m * phi));
  }
}
