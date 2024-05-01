#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main()
{
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], a[N];
  for (int i = 0; i < N; i++)
  {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 fxvec = _mm512_load_ps(fx);
  __m512 fyvec = _mm512_load_ps(fy);
  for (int i = 0; i < N; i++)
  {
    // for (int j = 0; j < N; j++)
    // {
    //   if (i != j)
    //   {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }
    __m512 rxvec = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec);
    __m512 ryvec = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec);
    __m512 rvec = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_sub_ps(rxvec, rxvec), _mm512_sub_ps(ryvec, ryvec)));
    __mmask16 mask = ~(1 << i);
    rvec = _mm512_mask_blend_ps(mask, _mm512_set1_ps(0), rvec);
    fx[i] -= _mm512_reduce_add_ps(_mm512_mul_ps(rxvec, _mm512_mul_ps(mvec, _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec)))));
    fy[i] -= _mm512_reduce_add_ps(_mm512_mul_ps(ryvec, _mm512_mul_ps(mvec, _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec)))));
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
