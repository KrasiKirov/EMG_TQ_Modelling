/*
 * Supine_SpeedGoat_v3.cpp
 *
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Code generation for model "Supine_SpeedGoat_v3".
 *
 * Model version              : 21.53
 * Simulink Coder version : 9.7 (R2022a) 13-Nov-2021
 * C++ source code generated on : Thu Mar 26 12:57:30 2026
 *
 * Target selection: slrealtime.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Linux 64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "Supine_SpeedGoat_v3.h"
#include "rtwtypes.h"
#include "Supine_SpeedGoat_v3_cal.h"
#include <cmath>

extern "C" {

#include "rt_nonfinite.h"

}
#include "Supine_SpeedGoat_v3_private.h"
#include "rte_Supine_SpeedGoat_v3_parameters.h"
#include <cstring>
#include "zero_crossing_types.h"

const real_T Supine_SpeedGoat_v3_RGND = 0.0;/* real_T ground */

/* Block signals (default storage) */
B_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_B;

/* Continuous states */
X_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_X;

/* Block states (default storage) */
DW_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_DW;

/* Previous zero-crossings (trigger) states */
PrevZCX_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_PrevZCX;

/* External inputs (root inport signals with default storage) */
ExtU_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_U;

/* Real-time model */
RT_MODEL_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_M_ =
  RT_MODEL_Supine_SpeedGoat_v3_T();
RT_MODEL_Supine_SpeedGoat_v3_T *const Supine_SpeedGoat_v3_M =
  &Supine_SpeedGoat_v3_M_;

/* n-D Spline interpolation function */
real_T look_SplNBinXZcd(uint32_T numDims, const real_T* u, const
  rt_LUTSplineWork * const SWork)
{
  /*
   *   n-D column-major table lookup operating on real_T with:
   *       - Spline interpolation
   *       - Linear extrapolation
   *       - Binary breakpoint search
   *       - Index search starts at the same place each time
   */
  rt_LUTnWork * const TWork_look = SWork->m_TWork;
  real_T* const fraction = static_cast<real_T*>(TWork_look->m_bpLambda);
  uint32_T* const bpIdx = TWork_look->m_bpIndex;
  const uint32_T* const maxIndex = TWork_look->m_maxIndex;
  uint32_T k;
  for (k = 0U; k < numDims; k++) {
    const real_T* const bpData = ((const real_T * const *)
      TWork_look->m_bpDataSet)[k];
    bpIdx[k] = plook_binx(u[k], bpData, maxIndex[k], &fraction[k]);
  }

  return(intrp_NSplcd(numDims, SWork, 2U));
}

/*
 * Second derivative initialization function for spline
 * for last dimension.
 */
void rt_Spline2Derivd(const real_T *x, const real_T *y, uint32_T n, real_T *u,
                      real_T *y2)
{
  real_T p, qn, sig, un;
  uint32_T n1, i, k;
  n1 = n - 1U;
  y2[0U] = 0.0;
  u[0U] = 0.0;
  for (i = 1U; i < n1; i++) {
    real_T dxm1 = x[i] - x[i - 1U];
    real_T dxp1 = x[i + 1U] - x[i];
    real_T dxpm = dxp1 + dxm1;
    sig = dxm1 / dxpm;
    p = (sig * y2[i - 1U]) + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = ((y[i + 1U] - y[i]) / dxp1) - ((y[i] - y[i - 1U]) / dxm1);
    u[i] = (((6.0 * u[i]) / dxpm) - (sig * u[i - 1U])) / p;
  }

  qn = 0.0;
  un = 0.0;
  y2[n1] = (un - (qn * u[n1 - 1U])) / ((qn * y2[n1 - 1U]) + 1.0);
  for (k = n1; k > 0U; k--) {
    y2[k-1U] = (y2[k-1U] * y2[k]) + u[k-1U];
  }

  return;
}

/* n-D natural spline calculation function */
real_T intrp_NSplcd(uint32_T numDims, const rt_LUTSplineWork * const splWork,
                    uint32_T extrapMethod)
{
  uint32_T il;
  uint32_T iu, k, i;
  real_T h, s, p, smsq, pmsq;

  /* intermediate results work areas "this" and "next" */
  const rt_LUTnWork *TWork_interp = static_cast<const rt_LUTnWork *>
    (splWork->m_TWork);
  const real_T *fraction = static_cast<real_T *>(TWork_interp->m_bpLambda);
  const real_T *yp = static_cast<real_T *>(TWork_interp->m_tableData);
  real_T *yyA = static_cast<real_T *>(splWork->m_yyA);
  real_T *yyB = static_cast<real_T *>(splWork->m_yyB);
  real_T *yy2 = static_cast<real_T *>(splWork->m_yy2);
  real_T *up = static_cast<real_T *>(splWork->m_up);
  real_T *y2 = static_cast<real_T *>(splWork->m_y2);
  uint8_T* reCalc = splWork->m_reCalc;
  real_T *dp = static_cast<real_T *>(splWork->m_preBp0AndTable);
  const real_T **bpDataSet = (const real_T **) TWork_interp->m_bpDataSet;
  const real_T *xp = bpDataSet[0U];
  real_T *yy = yyA;
  uint32_T bufBank = 0U;
  uint32_T len = TWork_interp->m_maxIndex[0U] + 1U;

  /* compare bp0 and table to see whether they get changed */
  {
    /* compare the bp0 data */
    if (std::memcmp(dp, xp,
                    len * sizeof(real_T)) != 0) {
      *reCalc = 1;
      (void) std::memcpy(dp, xp,
                         len * sizeof(real_T));
    }

    /* compare the table data */
    dp = &(dp[len]);
    if (std::memcmp(dp, yp,
                    len * splWork->m_numYWorkElts[0U] * sizeof(real_T)) != 0) {
      *reCalc = 1;
      (void) std::memcpy(dp, yp,
                         len * splWork->m_numYWorkElts[0U] * sizeof(real_T));
    }
  }

  if (*reCalc == 1) {
    /* If table and bps are tunable calculate 1st dim 2nd deriv */
    /* Generate first dimension's second derivatives */
    for (i = 0U; i < splWork->m_numYWorkElts[0U]; i++) {
      rt_Spline2Derivd(xp, yp, len, up, y2);
      yp = &yp[len];
      y2 = &y2[len];
    }

    /* Set pointers back to beginning */
    yp = (const real_T *) TWork_interp->m_tableData;
    y2 = (real_T *) splWork->m_y2;
  }

  *reCalc = 0;

  /* Generate at-point splines in each dimension */
  for (k = 0U; k < numDims; k++ ) {
    /* this dimension's input setup */
    xp = bpDataSet[k];
    len = TWork_interp->m_maxIndex[k] + 1U;
    il = TWork_interp->m_bpIndex[k];
    iu = il + 1U;
    h = xp[iu] - xp[il];
    p = fraction[k];
    s = 1.0 - p;
    pmsq = p * ((p*p) - 1.0);
    smsq = s * ((s*s) - 1.0);

    /*
     * Calculate spline curves for input in this
     * dimension at each value of the higher
     * other dimensions\' points in the table.
     */
    if ((p > 1.0) && (extrapMethod == 2U) ) {
      real_T slope;
      for (i = 0U; i < splWork->m_numYWorkElts[k]; i++) {
        slope = (yp[iu] - yp[il]) + ((y2[il]*h*h)*(1.0/6.0));
        yy[i] = yp[iu] + (slope * (p-1.0));
        yp = &yp[len];
        y2 = &y2[len];
      }
    } else if ((p < 0.0) && (extrapMethod == 2U) ) {
      real_T slope;
      for (i = 0U; i < splWork->m_numYWorkElts[k]; i++) {
        slope = (yp[iu] - yp[il]) - ((y2[iu]*h*h)*(1.0/6.0));
        yy[i] = yp[il] + (slope * p);
        yp = &yp[len];
        y2 = &y2[len];
      }
    } else {
      for (i = 0U; i < splWork->m_numYWorkElts[k]; i++) {
        yy[i] = yp[il] + p * (yp[iu] - yp[il]) +
          ((smsq * y2[il] + pmsq * y2[iu])*h*h)*(1.0/6.0);
        yp = &yp[len];
        y2 = &y2[len];
      }
    }

    /* set pointers to new result and calculate second derivatives */
    yp = yy;
    y2 = yy2;
    if (splWork->m_numYWorkElts[k+1U] > 0U ) {
      uint32_T nextLen = TWork_interp->m_maxIndex[k+1U] + 1U;
      const real_T *nextXp = bpDataSet[k+1U];
      for (i = 0U; i < splWork->m_numYWorkElts[k+1U]; i++) {
        rt_Spline2Derivd(nextXp, yp, nextLen, up, y2);
        yp = &yp[nextLen];
        y2 = &y2[nextLen];
      }
    }

    /*
     * Set work vectors yp, y2 and yy for next iteration;
     * the yy just calculated becomes the yp in the
     * next iteration, y2 was just calculated for these
     * new points and the yy buffer is swapped to the space
     * for storing the next iteration\'s results.
     */
    yp = yy;
    y2 = yy2;

    /*
     * Swap buffers for next dimension and
     * toggle bufBank for next iteration.
     */
    if (bufBank == 0U) {
      yy = yyA;
      bufBank = 1U;
    } else {
      yy = yyB;
      bufBank = 0U;
    }
  }

  return( yp[0U] );
}

uint32_T plook_binx(real_T u, const real_T bp[], uint32_T maxIndex, real_T
                    *fraction)
{
  uint32_T bpIndex;

  /* Prelookup - Index and Fraction
     Index Search method: 'binary'
     Extrapolation method: 'Linear'
     Use previous index: 'off'
     Use last breakpoint for index at or above upper limit: 'off'
     Remove protection against out-of-range input in generated code: 'off'
   */
  if (u <= bp[0U]) {
    bpIndex = 0U;
    *fraction = (u - bp[0U]) / (bp[1U] - bp[0U]);
  } else if (u < bp[maxIndex]) {
    bpIndex = binsearch_u32d(u, bp, maxIndex >> 1U, maxIndex);
    *fraction = (u - bp[bpIndex]) / (bp[bpIndex + 1U] - bp[bpIndex]);
  } else {
    bpIndex = maxIndex - 1U;
    *fraction = (u - bp[maxIndex - 1U]) / (bp[maxIndex] - bp[maxIndex - 1U]);
  }

  return bpIndex;
}

uint32_T binsearch_u32d(real_T u, const real_T bp[], uint32_T startIndex,
  uint32_T maxIndex)
{
  uint32_T bpIdx;
  uint32_T bpIndex;
  uint32_T iRght;

  /* Binary Search */
  bpIdx = startIndex;
  bpIndex = 0U;
  iRght = maxIndex;
  while (iRght - bpIndex > 1U) {
    if (u < bp[bpIdx]) {
      iRght = bpIdx;
    } else {
      bpIndex = bpIdx;
    }

    bpIdx = (iRght + bpIndex) >> 1U;
  }

  return bpIndex;
}

/*
 * This function updates continuous states using the ODE5 fixed-step
 * solver algorithm
 */
static void rt_ertODEUpdateContinuousStates(RTWSolverInfo *si )
{
  /* Solver Matrices */
  static const real_T rt_ODE5_A[6] = {
    1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0
  };

  static const real_T rt_ODE5_B[6][6] = {
    { 1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0 },

    { 3.0/40.0, 9.0/40.0, 0.0, 0.0, 0.0, 0.0 },

    { 44.0/45.0, -56.0/15.0, 32.0/9.0, 0.0, 0.0, 0.0 },

    { 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0, 0.0, 0.0 },

    { 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0,
      0.0 },

    { 35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0 }
  };

  time_T t = rtsiGetT(si);
  time_T tnew = rtsiGetSolverStopTime(si);
  time_T h = rtsiGetStepSize(si);
  real_T *x = rtsiGetContStates(si);
  ODE5_IntgData *id = static_cast<ODE5_IntgData *>(rtsiGetSolverData(si));
  real_T *y = id->y;
  real_T *f0 = id->f[0];
  real_T *f1 = id->f[1];
  real_T *f2 = id->f[2];
  real_T *f3 = id->f[3];
  real_T *f4 = id->f[4];
  real_T *f5 = id->f[5];
  real_T hB[6];
  int_T i;
  int_T nXc = 16;
  rtsiSetSimTimeStep(si,MINOR_TIME_STEP);

  /* Save the state values at time t in y, we'll use x as ynew. */
  (void) std::memcpy(y, x,
                     static_cast<uint_T>(nXc)*sizeof(real_T));

  /* Assumes that rtsiSetT and ModelOutputs are up-to-date */
  /* f0 = f(t,y) */
  rtsiSetdX(si, f0);
  Supine_SpeedGoat_v3_derivatives();

  /* f(:,2) = feval(odefile, t + hA(1), y + f*hB(:,1), args(:)(*)); */
  hB[0] = h * rt_ODE5_B[0][0];
  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0]);
  }

  rtsiSetT(si, t + h*rt_ODE5_A[0]);
  rtsiSetdX(si, f1);
  Supine_SpeedGoat_v3_step0();
  Supine_SpeedGoat_v3_derivatives();

  /* f(:,3) = feval(odefile, t + hA(2), y + f*hB(:,2), args(:)(*)); */
  for (i = 0; i <= 1; i++) {
    hB[i] = h * rt_ODE5_B[1][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1]);
  }

  rtsiSetT(si, t + h*rt_ODE5_A[1]);
  rtsiSetdX(si, f2);
  Supine_SpeedGoat_v3_step0();
  Supine_SpeedGoat_v3_derivatives();

  /* f(:,4) = feval(odefile, t + hA(3), y + f*hB(:,3), args(:)(*)); */
  for (i = 0; i <= 2; i++) {
    hB[i] = h * rt_ODE5_B[2][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2]);
  }

  rtsiSetT(si, t + h*rt_ODE5_A[2]);
  rtsiSetdX(si, f3);
  Supine_SpeedGoat_v3_step0();
  Supine_SpeedGoat_v3_derivatives();

  /* f(:,5) = feval(odefile, t + hA(4), y + f*hB(:,4), args(:)(*)); */
  for (i = 0; i <= 3; i++) {
    hB[i] = h * rt_ODE5_B[3][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2] +
                   f3[i]*hB[3]);
  }

  rtsiSetT(si, t + h*rt_ODE5_A[3]);
  rtsiSetdX(si, f4);
  Supine_SpeedGoat_v3_step0();
  Supine_SpeedGoat_v3_derivatives();

  /* f(:,6) = feval(odefile, t + hA(5), y + f*hB(:,5), args(:)(*)); */
  for (i = 0; i <= 4; i++) {
    hB[i] = h * rt_ODE5_B[4][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2] +
                   f3[i]*hB[3] + f4[i]*hB[4]);
  }

  rtsiSetT(si, tnew);
  rtsiSetdX(si, f5);
  Supine_SpeedGoat_v3_step0();
  Supine_SpeedGoat_v3_derivatives();

  /* tnew = t + hA(6);
     ynew = y + f*hB(:,6); */
  for (i = 0; i <= 5; i++) {
    hB[i] = h * rt_ODE5_B[5][i];
  }

  for (i = 0; i < nXc; i++) {
    x[i] = y[i] + (f0[i]*hB[0] + f1[i]*hB[1] + f2[i]*hB[2] +
                   f3[i]*hB[3] + f4[i]*hB[4] + f5[i]*hB[5]);
  }

  rtsiSetSimTimeStep(si,MAJOR_TIME_STEP);
}

real_T rt_urand_Upu32_Yd_f_pw_snf(uint32_T *u)
{
  uint32_T hi;
  uint32_T lo;

  /* Uniform random number generator (random number between 0 and 1)

     #define IA      16807                      magic multiplier = 7^5
     #define IM      2147483647                 modulus = 2^31-1
     #define IQ      127773                     IM div IA
     #define IR      2836                       IM modulo IA
     #define S       4.656612875245797e-10      reciprocal of 2^31-1
     test = IA * (seed % IQ) - IR * (seed/IQ)
     seed = test < 0 ? (test + IM) : test
     return (seed*S)
   */
  lo = *u % 127773U * 16807U;
  hi = *u / 127773U * 2836U;
  if (lo < hi) {
    *u = 2147483647U - (hi - lo);
  } else {
    *u = lo - hi;
  }

  return static_cast<real_T>(*u) * 4.6566128752457969E-10;
}

/* Model step function for TID0 */
void Supine_SpeedGoat_v3_step0(void)   /* Sample time: [0.0s, 0.0s] */
{
  if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
    /* set solver stop time */
    if (!(Supine_SpeedGoat_v3_M->Timing.clockTick0+1)) {
      rtsiSetSolverStopTime(&Supine_SpeedGoat_v3_M->solverInfo,
                            ((Supine_SpeedGoat_v3_M->Timing.clockTickH0 + 1) *
        Supine_SpeedGoat_v3_M->Timing.stepSize0 * 4294967296.0));
    } else {
      rtsiSetSolverStopTime(&Supine_SpeedGoat_v3_M->solverInfo,
                            ((Supine_SpeedGoat_v3_M->Timing.clockTick0 + 1) *
        Supine_SpeedGoat_v3_M->Timing.stepSize0 +
        Supine_SpeedGoat_v3_M->Timing.clockTickH0 *
        Supine_SpeedGoat_v3_M->Timing.stepSize0 * 4294967296.0));
    }

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [0.02s, 0.0s]  */
    Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits[10] =
      (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_2 == 0);
    (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_2)++;
    if ((Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_2) > 19) {
      Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_2 = 0;
    }

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [0.2s, 0.0s]  */
    Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits[11] =
      (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_3 == 0);
    (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_3)++;
    if ((Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_3) > 199) {
      Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_3 = 0;
    }

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [0.5s, 0.0s]  */
    Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits[12] =
      (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_4 == 0);
    (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_4)++;
    if ((Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_4) > 499) {
      Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_4 = 0;
    }

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [0.7s, 0.0s]  */
    Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits[13] =
      (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_5 == 0);
    (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_5)++;
    if ((Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_5) > 699) {
      Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_5 = 0;
    }

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [1.0s, 0.0s]  */
    Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits[14] =
      (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_6 == 0);
    (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_6)++;
    if ((Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_6) > 999) {
      Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_6 = 0;
    }

    /* Update the flag to indicate when data transfers from
     *  Sample time: [0.001s, 0.0s] to Sample time: [2.0s, 0.0s]  */
    Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits[15] =
      (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_7 == 0);
    (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_7)++;
    if ((Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_7) > 1999) {
      Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_7 = 0;
    }
  }                                    /* end MajorTimeStep */

  /* Update absolute time of base rate at minor time step */
  if (rtmIsMinorTimeStep(Supine_SpeedGoat_v3_M)) {
    Supine_SpeedGoat_v3_M->Timing.t[0] = rtsiGetT
      (&Supine_SpeedGoat_v3_M->solverInfo);
  }

  {
    real_T lastTime;
    real_T tmp;
    real_T tmp_0;
    real_T u1;
    real_T u2;
    real_T *lastU;

    /* Sin: '<S3>/Sine Wave' */
    tmp_0 = *get_maxPF();
    tmp = *get_maxDF();

    /* Reset subsysRan breadcrumbs */
    srClearBC(Supine_SpeedGoat_v3_DW.SampleandHold_SubsysRanBC);
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* S-Function (sg_IO191_setup_s): '<Root>/Setup ' */

      /* Level2 S-Function Block: '<Root>/Setup ' (sg_IO191_setup_s) */
      {
        SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[0];
        sfcnOutputs(rts,0);
      }
    }

    /* StateSpace: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
    Supine_SpeedGoat_v3_B.uthorderBeselwithcutoff07HzbyGo = 0.0;

    /* StateSpace: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
    for (int32_T i = 0; i < 8; i++) {
      for (uint32_T ri =
           Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_c
           [static_cast<uint32_T>(i)]; ri <
           Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_c[i + 1U]; ri
           ++) {
        /* StateSpace: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
        Supine_SpeedGoat_v3_B.uthorderBeselwithcutoff07HzbyGo +=
          Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_o *
          Supine_SpeedGoat_v3_X.uthorderBeselwithcutoff07HzbyGo
          [static_cast<uint32_T>(i)];
      }
    }

    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* S-Function (sg_IO191_ad_s): '<Root>/Analog input ' */

      /* Level2 S-Function Block: '<Root>/Analog input ' (sg_IO191_ad_s) */
      {
        SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[1];
        sfcnOutputs(rts,0);
      }

      /* Gain: '<Root>/Gain1' */
      Supine_SpeedGoat_v3_B.Gain1 = Supine_SpeedGoat_v3_cal->Gain1_Gain *
        Supine_SpeedGoat_v3_B.Possition;

      /* S-Function (sdspnsamp2): '<S2>/N-Sample Enable1' */
      /* DSP System Toolbox N-Sample Enable  (sdspnsamp2) - '<S2>/N-Sample Enable1' */
      {
        {
          if (Supine_SpeedGoat_v3_DW.NSampleEnable1_Counter ==
              Supine_SpeedGoat_v3_cal->NSampleEnable1_N) {
            Supine_SpeedGoat_v3_B.NSampleEnable1 = (boolean_T)(2 -
              Supine_SpeedGoat_v3_cal->NSampleEnable1_ActiveLevel);
          } else {
            Supine_SpeedGoat_v3_B.NSampleEnable1 = (boolean_T)
              (Supine_SpeedGoat_v3_cal->NSampleEnable1_ActiveLevel - 1);
            (Supine_SpeedGoat_v3_DW.NSampleEnable1_Counter)++;
          }
        }
      }

      /* Outputs for Triggered SubSystem: '<S2>/Sample and Hold' incorporates:
       *  TriggerPort: '<S6>/Trigger'
       */
      if (rtsiIsModeUpdateTimeStep(&Supine_SpeedGoat_v3_M->solverInfo)) {
        boolean_T zcEvent;
        zcEvent = ((!Supine_SpeedGoat_v3_B.NSampleEnable1) &&
                   (Supine_SpeedGoat_v3_PrevZCX.SampleandHold_Trig_ZCE !=
                    ZERO_ZCSIG));
        if (zcEvent) {
          /* SignalConversion generated from: '<S6>/In' */
          Supine_SpeedGoat_v3_B.In = Supine_SpeedGoat_v3_B.Gain1;
          Supine_SpeedGoat_v3_DW.SampleandHold_SubsysRanBC = 4;
        }

        Supine_SpeedGoat_v3_PrevZCX.SampleandHold_Trig_ZCE =
          Supine_SpeedGoat_v3_B.NSampleEnable1;
      }

      /* End of Outputs for SubSystem: '<S2>/Sample and Hold' */

      /* Sum: '<S2>/Sum2' */
      Supine_SpeedGoat_v3_B.Sum2 = Supine_SpeedGoat_v3_B.Gain1 -
        Supine_SpeedGoat_v3_B.In;
    }

    /* StateSpace: '<S2>/Cut-off 1 Hz1' */
    Supine_SpeedGoat_v3_B.Cutoff1Hz1 = 0.0;

    /* StateSpace: '<S2>/Cut-off 1 Hz1' */
    for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff1Hz1_C_jc[0U]; ri <
         Supine_SpeedGoat_v3_cal->Cutoff1Hz1_C_jc[1U]; ri++) {
      /* StateSpace: '<S2>/Cut-off 1 Hz1' */
      Supine_SpeedGoat_v3_B.Cutoff1Hz1 +=
        Supine_SpeedGoat_v3_cal->Cutoff1Hz1_C_pr *
        Supine_SpeedGoat_v3_X.Cutoff1Hz1_CSTATE[0U];
    }

    for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff1Hz1_C_jc[1U]; ri <
         Supine_SpeedGoat_v3_cal->Cutoff1Hz1_C_jc[2U]; ri++) {
      /* StateSpace: '<S2>/Cut-off 1 Hz1' */
      Supine_SpeedGoat_v3_B.Cutoff1Hz1 +=
        Supine_SpeedGoat_v3_cal->Cutoff1Hz1_C_pr *
        Supine_SpeedGoat_v3_X.Cutoff1Hz1_CSTATE[1U];
    }

    /* Sum: '<Root>/Sum2' incorporates:
     *  Inport: '<Root>/In1'
     */
    Supine_SpeedGoat_v3_B.Sum2_m = Supine_SpeedGoat_v3_U.In1;

    /* Gain: '<Root>/em_null' */
    Supine_SpeedGoat_v3_B.em_null = Supine_SpeedGoat_v3_cal->em_null_Gain *
      Supine_SpeedGoat_v3_B.Sum2_m;

    /* Sum: '<S2>/Sum5' */
    Supine_SpeedGoat_v3_B.Sum5 = Supine_SpeedGoat_v3_B.Cutoff1Hz1 +
      Supine_SpeedGoat_v3_B.em_null;

    /* Sum: '<S2>/Sum' */
    Supine_SpeedGoat_v3_B.Sum = Supine_SpeedGoat_v3_B.Sum5 -
      Supine_SpeedGoat_v3_B.Sum2;

    /* Gain: '<S2>/P' */
    Supine_SpeedGoat_v3_B.P = Supine_SpeedGoat_v3_cal->P_Gain *
      Supine_SpeedGoat_v3_B.Sum;

    /* Gain: '<S2>/D' */
    Supine_SpeedGoat_v3_B.D = Supine_SpeedGoat_v3_cal->D_Gain *
      Supine_SpeedGoat_v3_B.Sum;

    /* Derivative: '<S2>/Derivative' */
    if ((Supine_SpeedGoat_v3_DW.TimeStampA >= Supine_SpeedGoat_v3_M->Timing.t[0])
        && (Supine_SpeedGoat_v3_DW.TimeStampB >= Supine_SpeedGoat_v3_M->
            Timing.t[0])) {
      /* Derivative: '<S2>/Derivative' */
      Supine_SpeedGoat_v3_B.Derivative = 0.0;
    } else {
      lastTime = Supine_SpeedGoat_v3_DW.TimeStampA;
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeA;
      if (Supine_SpeedGoat_v3_DW.TimeStampA < Supine_SpeedGoat_v3_DW.TimeStampB)
      {
        if (Supine_SpeedGoat_v3_DW.TimeStampB < Supine_SpeedGoat_v3_M->Timing.t
            [0]) {
          lastTime = Supine_SpeedGoat_v3_DW.TimeStampB;
          lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB;
        }
      } else if (Supine_SpeedGoat_v3_DW.TimeStampA >=
                 Supine_SpeedGoat_v3_M->Timing.t[0]) {
        lastTime = Supine_SpeedGoat_v3_DW.TimeStampB;
        lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB;
      }

      lastTime = Supine_SpeedGoat_v3_M->Timing.t[0] - lastTime;

      /* Derivative: '<S2>/Derivative' */
      Supine_SpeedGoat_v3_B.Derivative = (Supine_SpeedGoat_v3_B.D - *lastU) /
        lastTime;
    }

    /* End of Derivative: '<S2>/Derivative' */

    /* Saturate: '<S2>/Saturation' */
    lastTime = Supine_SpeedGoat_v3_B.Derivative;
    u1 = Supine_SpeedGoat_v3_cal->Saturation_LowerSat;
    u2 = Supine_SpeedGoat_v3_cal->Saturation_UpperSat;
    if (lastTime > u2) {
      /* Saturate: '<S2>/Saturation' */
      Supine_SpeedGoat_v3_B.Saturation = u2;
    } else if (lastTime < u1) {
      /* Saturate: '<S2>/Saturation' */
      Supine_SpeedGoat_v3_B.Saturation = u1;
    } else {
      /* Saturate: '<S2>/Saturation' */
      Supine_SpeedGoat_v3_B.Saturation = lastTime;
    }

    /* End of Saturate: '<S2>/Saturation' */

    /* Integrator: '<S2>/Integrator' */
    /* Limited  Integrator  */
    if (Supine_SpeedGoat_v3_X.Integrator_CSTATE >=
        Supine_SpeedGoat_v3_cal->Integrator_UpperSat) {
      Supine_SpeedGoat_v3_X.Integrator_CSTATE =
        Supine_SpeedGoat_v3_cal->Integrator_UpperSat;
    } else if (Supine_SpeedGoat_v3_X.Integrator_CSTATE <=
               Supine_SpeedGoat_v3_cal->Integrator_LowerSat) {
      Supine_SpeedGoat_v3_X.Integrator_CSTATE =
        Supine_SpeedGoat_v3_cal->Integrator_LowerSat;
    }

    /* Integrator: '<S2>/Integrator' */
    Supine_SpeedGoat_v3_B.Integrator = Supine_SpeedGoat_v3_X.Integrator_CSTATE;

    /* Saturate: '<S2>/Saturation1' */
    lastTime = Supine_SpeedGoat_v3_B.Integrator;
    u1 = Supine_SpeedGoat_v3_cal->Saturation1_LowerSat;
    u2 = Supine_SpeedGoat_v3_cal->Saturation1_UpperSat;
    if (lastTime > u2) {
      /* Saturate: '<S2>/Saturation1' */
      Supine_SpeedGoat_v3_B.Saturation1 = u2;
    } else if (lastTime < u1) {
      /* Saturate: '<S2>/Saturation1' */
      Supine_SpeedGoat_v3_B.Saturation1 = u1;
    } else {
      /* Saturate: '<S2>/Saturation1' */
      Supine_SpeedGoat_v3_B.Saturation1 = lastTime;
    }

    /* End of Saturate: '<S2>/Saturation1' */

    /* Derivative: '<S2>/Derivative1' */
    if ((Supine_SpeedGoat_v3_DW.TimeStampA_d >= Supine_SpeedGoat_v3_M->Timing.t
         [0]) && (Supine_SpeedGoat_v3_DW.TimeStampB_n >=
                  Supine_SpeedGoat_v3_M->Timing.t[0])) {
      /* Derivative: '<S2>/Derivative1' */
      Supine_SpeedGoat_v3_B.Derivative1 = 0.0;
    } else {
      lastTime = Supine_SpeedGoat_v3_DW.TimeStampA_d;
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeA_m;
      if (Supine_SpeedGoat_v3_DW.TimeStampA_d <
          Supine_SpeedGoat_v3_DW.TimeStampB_n) {
        if (Supine_SpeedGoat_v3_DW.TimeStampB_n <
            Supine_SpeedGoat_v3_M->Timing.t[0]) {
          lastTime = Supine_SpeedGoat_v3_DW.TimeStampB_n;
          lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB_m;
        }
      } else if (Supine_SpeedGoat_v3_DW.TimeStampA_d >=
                 Supine_SpeedGoat_v3_M->Timing.t[0]) {
        lastTime = Supine_SpeedGoat_v3_DW.TimeStampB_n;
        lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB_m;
      }

      lastTime = Supine_SpeedGoat_v3_M->Timing.t[0] - lastTime;

      /* Derivative: '<S2>/Derivative1' */
      Supine_SpeedGoat_v3_B.Derivative1 = (Supine_SpeedGoat_v3_B.Sum5 - *lastU) /
        lastTime;
    }

    /* End of Derivative: '<S2>/Derivative1' */

    /* Gain: '<S2>/ddt_gain' */
    Supine_SpeedGoat_v3_B.ddt_gain = Supine_SpeedGoat_v3_cal->ddt_gain_Gain *
      Supine_SpeedGoat_v3_B.Derivative1;

    /* Saturate: '<S2>/Saturation2' */
    lastTime = Supine_SpeedGoat_v3_B.ddt_gain;
    u1 = Supine_SpeedGoat_v3_cal->Saturation2_LowerSat;
    u2 = Supine_SpeedGoat_v3_cal->Saturation2_UpperSat;
    if (lastTime > u2) {
      /* Saturate: '<S2>/Saturation2' */
      Supine_SpeedGoat_v3_B.Saturation2 = u2;
    } else if (lastTime < u1) {
      /* Saturate: '<S2>/Saturation2' */
      Supine_SpeedGoat_v3_B.Saturation2 = u1;
    } else {
      /* Saturate: '<S2>/Saturation2' */
      Supine_SpeedGoat_v3_B.Saturation2 = lastTime;
    }

    /* End of Saturate: '<S2>/Saturation2' */

    /* Sum: '<S2>/Sum1' */
    Supine_SpeedGoat_v3_B.Sum1 = ((Supine_SpeedGoat_v3_B.P +
      Supine_SpeedGoat_v3_B.Saturation) + Supine_SpeedGoat_v3_B.Saturation1) +
      Supine_SpeedGoat_v3_B.Saturation2;
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* UniformRandomNumber: '<S2>/Uniform Random Number 0.1 ms' */
      Supine_SpeedGoat_v3_B.UniformRandomNumber01ms =
        Supine_SpeedGoat_v3_DW.UniformRandomNumber01ms_NextOut;
    }

    /* StateSpace: '<S2>/Cut-off 35 Hz' */
    Supine_SpeedGoat_v3_B.Cutoff35Hz = 0.0;

    /* StateSpace: '<S2>/Cut-off 35 Hz' */
    for (int32_T i = 0; i < 5; i++) {
      for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff35Hz_C_jc
           [static_cast<uint32_T>(i)]; ri <
           Supine_SpeedGoat_v3_cal->Cutoff35Hz_C_jc[i + 1U]; ri++) {
        /* StateSpace: '<S2>/Cut-off 35 Hz' */
        Supine_SpeedGoat_v3_B.Cutoff35Hz +=
          Supine_SpeedGoat_v3_cal->Cutoff35Hz_C_pr[ri] *
          Supine_SpeedGoat_v3_X.Cutoff35Hz_CSTATE[static_cast<uint32_T>(i)];
      }
    }

    for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff35Hz_D_jc[0U]; ri <
         Supine_SpeedGoat_v3_cal->Cutoff35Hz_D_jc[1U]; ri++) {
      /* StateSpace: '<S2>/Cut-off 35 Hz' */
      Supine_SpeedGoat_v3_B.Cutoff35Hz +=
        Supine_SpeedGoat_v3_cal->Cutoff35Hz_D_pr *
        Supine_SpeedGoat_v3_B.UniformRandomNumber01ms;
    }

    /* Gain: '<S2>/didder_gain' */
    Supine_SpeedGoat_v3_B.didder_gain =
      Supine_SpeedGoat_v3_cal->didder_gain_Gain *
      Supine_SpeedGoat_v3_B.Cutoff35Hz;

    /* Sum: '<S2>/Sum3' */
    Supine_SpeedGoat_v3_B.Sum3 = Supine_SpeedGoat_v3_B.Sum1 +
      Supine_SpeedGoat_v3_B.didder_gain;

    /* Gain: '<Root>/Gain' */
    Supine_SpeedGoat_v3_B.Gain = Supine_SpeedGoat_v3_cal->Gain_Gain *
      Supine_SpeedGoat_v3_B.Sum3;

    /* Gain: '<Root>/Gain3' incorporates:
     *  Inport: '<Root>/In1'
     */
    Supine_SpeedGoat_v3_B.Gain3 = Supine_SpeedGoat_v3_cal->Gain3_Gain *
      Supine_SpeedGoat_v3_U.In1;
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* S-Function (sg_IO191_da_s): '<Root>/Analog output ' */

      /* Level2 S-Function Block: '<Root>/Analog output ' (sg_IO191_da_s) */
      {
        SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[2];
        sfcnOutputs(rts,0);
      }

      /* Constant: '<Root>/position_switch4' */
      Supine_SpeedGoat_v3_B.position_switch4 =
        Supine_SpeedGoat_v3_cal->position_switch4_Value;
    }

    /* MultiPortSwitch: '<Root>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' incorporates:
     *  Constant: '<Root>/scope_switch_signal'
     */
    switch (static_cast<int32_T>
            (Supine_SpeedGoat_v3_cal->scope_switch_signal_Value)) {
     case 1:
      /* MultiPortSwitch: '<Root>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirstinpu =
        Supine_SpeedGoat_v3_B.Sum;
      break;

     case 2:
      /* MultiPortSwitch: '<Root>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirstinpu =
        Supine_SpeedGoat_v3_B.Sum5;
      break;

     default:
      /* MultiPortSwitch: '<Root>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirstinpu =
        Supine_SpeedGoat_v3_B.position_switch4;
      break;
    }

    /* End of MultiPortSwitch: '<Root>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */

    /* Sum: '<Root>/Sum1' incorporates:
     *  Constant: '<Root>/Scope_offset'
     */
    Supine_SpeedGoat_v3_B.Sum1_l = Supine_SpeedGoat_v3_cal->Scope_offset_Value +
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirstinpu;

    /* Gain: '<Root>/Scope_gain' */
    Supine_SpeedGoat_v3_B.Scope_gain = Supine_SpeedGoat_v3_cal->Scope_gain_Gain *
      Supine_SpeedGoat_v3_B.Sum1_l;
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* Lookup_n-D: '<S3>/MVC_LUT' incorporates:
       *  Constant: '<S3>/Constant'
       *  Constant: '<S3>/MVC_TQ'
       *  Constant: '<S3>/isometric_ankle_angles'
       */
      /*
       * About '<S3>/MVC_LUT':
       *       Table size:  8
       *    Interpolation:  Spline
       *    Extrapolation:  Linear
       *   Breakpt Search:  Binary
       *    Breakpt Cache:  OFF
       */
      Supine_SpeedGoat_v3_B.MVC_LUT = look_SplNBinXZcd(1U,
        &Supine_SpeedGoat_v3_cal->Constant_Value, (rt_LUTSplineWork*)
        &Supine_SpeedGoat_v3_DW.SWork[0]);

      /* Abs: '<S3>/Abs2' */
      Supine_SpeedGoat_v3_B.AbsoluteMVCtorqueinNm = std::abs
        (Supine_SpeedGoat_v3_B.MVC_LUT);

      /* Gain: '<S3>/PERCENT_MVC' */
      Supine_SpeedGoat_v3_B.desiredtorqueinNm =
        Supine_SpeedGoat_v3_cal->PERCENT_MVC_Gain *
        Supine_SpeedGoat_v3_B.AbsoluteMVCtorqueinNm;

      /* RateTransition: '<S9>/Rate Transition1' */
      if (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_3 == 1) {
        Supine_SpeedGoat_v3_DW.RateTransition1_RdBufIdx = static_cast<int8_T>
          (Supine_SpeedGoat_v3_DW.RateTransition1_RdBufIdx == 0);
      }

      /* RateTransition: '<S9>/Rate Transition1' */
      Supine_SpeedGoat_v3_B.RateTransition1 =
        Supine_SpeedGoat_v3_DW.RateTransition1_Buf[Supine_SpeedGoat_v3_DW.RateTransition1_RdBufIdx];

      /* RateTransition: '<S9>/Rate Transition2' */
      if (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_4 == 1) {
        Supine_SpeedGoat_v3_DW.RateTransition2_RdBufIdx = static_cast<int8_T>
          (Supine_SpeedGoat_v3_DW.RateTransition2_RdBufIdx == 0);
      }

      /* RateTransition: '<S9>/Rate Transition2' */
      Supine_SpeedGoat_v3_B.RateTransition2 =
        Supine_SpeedGoat_v3_DW.RateTransition2_Buf[Supine_SpeedGoat_v3_DW.RateTransition2_RdBufIdx];

      /* RateTransition: '<S9>/Rate Transition3' */
      if (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_5 == 1) {
        Supine_SpeedGoat_v3_DW.RateTransition3_RdBufIdx = static_cast<int8_T>
          (Supine_SpeedGoat_v3_DW.RateTransition3_RdBufIdx == 0);
      }

      /* RateTransition: '<S9>/Rate Transition3' */
      Supine_SpeedGoat_v3_B.RateTransition3 =
        Supine_SpeedGoat_v3_DW.RateTransition3_Buf[Supine_SpeedGoat_v3_DW.RateTransition3_RdBufIdx];

      /* RateTransition: '<S9>/Rate Transition4' */
      if (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_6 == 1) {
        Supine_SpeedGoat_v3_DW.RateTransition4_RdBufIdx = static_cast<int8_T>
          (Supine_SpeedGoat_v3_DW.RateTransition4_RdBufIdx == 0);
      }

      /* RateTransition: '<S9>/Rate Transition4' */
      Supine_SpeedGoat_v3_B.RateTransition4 =
        Supine_SpeedGoat_v3_DW.RateTransition4_Buf[Supine_SpeedGoat_v3_DW.RateTransition4_RdBufIdx];

      /* RateTransition: '<S9>/Rate Transition5' */
      if (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_7 == 1) {
        Supine_SpeedGoat_v3_DW.RateTransition5_RdBufIdx = static_cast<int8_T>
          (Supine_SpeedGoat_v3_DW.RateTransition5_RdBufIdx == 0);
      }

      /* RateTransition: '<S9>/Rate Transition5' */
      Supine_SpeedGoat_v3_B.RateTransition5 =
        Supine_SpeedGoat_v3_DW.RateTransition5_Buf[Supine_SpeedGoat_v3_DW.RateTransition5_RdBufIdx];

      /* MultiPortSwitch: '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' incorporates:
       *  Constant: '<S9>/Viz_Fdbk_PRBS_Switch_Time_Selector'
       */
      switch (static_cast<int32_T>
              (Supine_SpeedGoat_v3_cal->Viz_Fdbk_PRBS_Switch_Time_Selec)) {
       case 1:
        /* MultiPortSwitch: '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' */
        Supine_SpeedGoat_v3_B.scope_multiport_switchFirstin_a =
          Supine_SpeedGoat_v3_B.RateTransition1;
        break;

       case 2:
        /* MultiPortSwitch: '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' */
        Supine_SpeedGoat_v3_B.scope_multiport_switchFirstin_a =
          Supine_SpeedGoat_v3_B.RateTransition2;
        break;

       case 3:
        /* MultiPortSwitch: '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' */
        Supine_SpeedGoat_v3_B.scope_multiport_switchFirstin_a =
          Supine_SpeedGoat_v3_B.RateTransition3;
        break;

       case 4:
        /* MultiPortSwitch: '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' */
        Supine_SpeedGoat_v3_B.scope_multiport_switchFirstin_a =
          Supine_SpeedGoat_v3_B.RateTransition4;
        break;

       default:
        /* MultiPortSwitch: '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' */
        Supine_SpeedGoat_v3_B.scope_multiport_switchFirstin_a =
          Supine_SpeedGoat_v3_B.RateTransition5;
        break;
      }

      /* End of MultiPortSwitch: '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' */

      /* Switch: '<S9>/Switch' */
      if (Supine_SpeedGoat_v3_B.scope_multiport_switchFirstin_a >
          Supine_SpeedGoat_v3_cal->Switch_Threshold) {
        /* Switch: '<S9>/Switch' */
        Supine_SpeedGoat_v3_B.Switch = Supine_SpeedGoat_v3_B.desiredtorqueinNm;
      } else {
        /* Switch: '<S9>/Switch' incorporates:
         *  Constant: '<S9>/Zero_MVC'
         */
        Supine_SpeedGoat_v3_B.Switch = Supine_SpeedGoat_v3_cal->Zero_MVC_Value;
      }

      /* End of Switch: '<S9>/Switch' */
    }

    /* MultiPortSwitch: '<S8>/Multiport Switch' incorporates:
     *  Constant: '<S3>/mvc_profile_selector'
     */
    if (static_cast<int32_T>(Supine_SpeedGoat_v3_cal->mvc_profile_selector_Value)
        == 1) {
      /* MultiPortSwitch: '<S8>/Multiport Switch' */
      Supine_SpeedGoat_v3_B.MultiportSwitch = Supine_SpeedGoat_v3_B.Switch;
    } else {
      /* MultiPortSwitch: '<S8>/Multiport Switch' incorporates:
       *  Inport: '<Root>/time_varying_mvc'
       */
      Supine_SpeedGoat_v3_B.MultiportSwitch =
        Supine_SpeedGoat_v3_U.time_varying_mvc;
    }

    /* End of MultiPortSwitch: '<S8>/Multiport Switch' */

    /* Product: '<S3>/Divide1' */
    Supine_SpeedGoat_v3_B.Divide1 = Supine_SpeedGoat_v3_B.MultiportSwitch /
      Supine_SpeedGoat_v3_B.AbsoluteMVCtorqueinNm;

    /* Product: '<S3>/Divide' */
    Supine_SpeedGoat_v3_B.Divide =
      Supine_SpeedGoat_v3_B.uthorderBeselwithcutoff07HzbyGo /
      Supine_SpeedGoat_v3_B.AbsoluteMVCtorqueinNm;

    /* MultiPortSwitch: '<S3>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' incorporates:
     *  Constant: '<S3>/Contraction_Profile_Selector'
     */
    if (static_cast<int32_T>
        (Supine_SpeedGoat_v3_cal->Contraction_Profile_Selector_Va) == 1) {
      /* Gain: '<S3>/decimal_to_percent' */
      Supine_SpeedGoat_v3_B.decimal_to_percent[0] =
        Supine_SpeedGoat_v3_cal->decimal_to_percent_Gain *
        Supine_SpeedGoat_v3_B.Divide1;
      Supine_SpeedGoat_v3_B.decimal_to_percent[1] =
        Supine_SpeedGoat_v3_cal->decimal_to_percent_Gain *
        Supine_SpeedGoat_v3_B.Divide;

      /* MultiPortSwitch: '<S3>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirsti_aa[0] =
        Supine_SpeedGoat_v3_B.decimal_to_percent[0];
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirsti_aa[1] =
        Supine_SpeedGoat_v3_B.decimal_to_percent[1];
    } else {
      /* MultiPortSwitch: '<S3>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirsti_aa[0] =
        Supine_SpeedGoat_v3_B.desiredtorqueinNm;
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirsti_aa[1] =
        Supine_SpeedGoat_v3_B.uthorderBeselwithcutoff07HzbyGo;
    }

    /* End of MultiPortSwitch: '<S3>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */

    /* Sum: '<S3>/Sum4' */
    Supine_SpeedGoat_v3_B.Sum4 =
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirsti_aa[0] -
      Supine_SpeedGoat_v3_B.scope_multiport_switchFirsti_aa[1];
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* Gain: '<Root>/Gain2' */
      Supine_SpeedGoat_v3_B.Gain2 = Supine_SpeedGoat_v3_cal->Gain2_Gain *
        Supine_SpeedGoat_v3_B.Torque_c;

      /* Gain: '<Root>/volt_to_tq' */
      Supine_SpeedGoat_v3_B.measuredtorqueinNm =
        Supine_SpeedGoat_v3_cal->volt_to_tq_Gain * Supine_SpeedGoat_v3_B.Gain2;
    }

    /* DeadZone: '<S2>/Dead Zone' */
    if (Supine_SpeedGoat_v3_B.Sum > Supine_SpeedGoat_v3_cal->DeadZone_End) {
      /* DeadZone: '<S2>/Dead Zone' */
      Supine_SpeedGoat_v3_B.DeadZone = Supine_SpeedGoat_v3_B.Sum -
        Supine_SpeedGoat_v3_cal->DeadZone_End;
    } else if (Supine_SpeedGoat_v3_B.Sum >=
               Supine_SpeedGoat_v3_cal->DeadZone_Start) {
      /* DeadZone: '<S2>/Dead Zone' */
      Supine_SpeedGoat_v3_B.DeadZone = 0.0;
    } else {
      /* DeadZone: '<S2>/Dead Zone' */
      Supine_SpeedGoat_v3_B.DeadZone = Supine_SpeedGoat_v3_B.Sum -
        Supine_SpeedGoat_v3_cal->DeadZone_Start;
    }

    /* End of DeadZone: '<S2>/Dead Zone' */

    /* Gain: '<S2>/I' */
    Supine_SpeedGoat_v3_B.I = Supine_SpeedGoat_v3_cal->I_Gain *
      Supine_SpeedGoat_v3_B.DeadZone;
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* S-Function (slrealtimebytepacking): '<S7>/Byte Packing' */

      /* Byte Packing: <S7>/Byte Packing */
      (void)memcpy((uint8_T*)&Supine_SpeedGoat_v3_B.BytePacking[0] + 0, (uint8_T*)
                   &Supine_SpeedGoat_v3_B.scope_multiport_switchFirsti_aa[0], 16);

      /* RateTransition: '<S7>/Rate Transition' */
      if (Supine_SpeedGoat_v3_M->Timing.RateInteraction.TID1_2 == 1) {
        Supine_SpeedGoat_v3_DW.RateTransition_WrBufIdx = static_cast<int8_T>
          (Supine_SpeedGoat_v3_DW.RateTransition_WrBufIdx == 0);
      }

      for (int32_T i = 0; i < 16; i++) {
        Supine_SpeedGoat_v3_DW.RateTransition_Buf[i +
          (Supine_SpeedGoat_v3_DW.RateTransition_WrBufIdx << 4)] =
          Supine_SpeedGoat_v3_B.BytePacking[i];
      }

      /* End of RateTransition: '<S7>/Rate Transition' */
      /* Lookup_n-D: '<S3>/Passive_LUT' incorporates:
       *  Constant: '<S3>/Constant'
       *  Constant: '<S3>/Passive_TQ'
       *  Constant: '<S3>/isometric_ankle_angles'
       */
      /*
       * About '<S3>/Passive_LUT':
       *       Table size:  8
       *    Interpolation:  Spline
       *    Extrapolation:  Linear
       *   Breakpt Search:  Binary
       *    Breakpt Cache:  OFF
       */
      Supine_SpeedGoat_v3_B.Passive_LUT = look_SplNBinXZcd(1U,
        &Supine_SpeedGoat_v3_cal->Constant_Value, (rt_LUTSplineWork*)
        &Supine_SpeedGoat_v3_DW.SWork_e[0]);

      /* Gain: '<S3>/volt_to_tq' */
      Supine_SpeedGoat_v3_B.measuredtorqueinNm_j =
        Supine_SpeedGoat_v3_cal->volt_to_tq_Gain_m * Supine_SpeedGoat_v3_B.Gain2;

      /* Sum: '<S3>/Sum5' */
      Supine_SpeedGoat_v3_B.Sum5_d = Supine_SpeedGoat_v3_B.measuredtorqueinNm_j
        - Supine_SpeedGoat_v3_B.Passive_LUT;

      /* Abs: '<S3>/Abs1' */
      Supine_SpeedGoat_v3_B.absolutetorquewithoutpassiveinN = std::abs
        (Supine_SpeedGoat_v3_B.Sum5_d);
    }

    /* Sin: '<S3>/Sine Wave' */
    lastTime = (tmp - tmp_0) / 2.0;
    u1 = (tmp + tmp_0) / 2.0;

    /* Sin: '<S3>/Sine Wave' */
    Supine_SpeedGoat_v3_B.SineWave = std::sin
      (Supine_SpeedGoat_v3_cal->SineWave_Freq * Supine_SpeedGoat_v3_M->Timing.t
       [0] + Supine_SpeedGoat_v3_cal->SineWave_Phase) * lastTime + u1;

    /* Clock: '<Root>/Clock' */
    Supine_SpeedGoat_v3_B.Clock = Supine_SpeedGoat_v3_M->Timing.t[0];
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      /* S-Function (sg_IO191_di_s): '<S1>/Digital input ' */

      /* Level2 S-Function Block: '<S1>/Digital input ' (sg_IO191_di_s) */
      {
        SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[3];
        sfcnOutputs(rts,0);
      }

      /* Gain: '<S1>/FwdFlow_Stat' */
      Supine_SpeedGoat_v3_B.FwdFlow_Stat =
        Supine_SpeedGoat_v3_cal->FwdFlow_Stat_Gain *
        Supine_SpeedGoat_v3_B.FF_stat;

      /* Memory: '<S4>/Memory7' */
      Supine_SpeedGoat_v3_B.Memory7 =
        Supine_SpeedGoat_v3_DW.Memory7_PreviousInput;

      /* S-Function (sdspnsamp2): '<S5>/N-Sample Enable' */
      /* DSP System Toolbox N-Sample Enable  (sdspnsamp2) - '<S5>/N-Sample Enable' */
      {
        {
          if (MWDSP_EPHZCFcn(EVENT_PORT_MODE_RISING,
                             (EventPortSigState *)
                             &Supine_SpeedGoat_v3_DW.NSampleEnable_EPHState,
                             EventPortSigStateFcn_D
                             (Supine_SpeedGoat_v3_B.Memory7)
                             ) != EVENT_PORT_EVENT_NONE) {
            /* Reset iteration count. */
            Supine_SpeedGoat_v3_DW.NSampleEnable_Counter = 0;
          }

          if (Supine_SpeedGoat_v3_DW.NSampleEnable_Counter ==
              Supine_SpeedGoat_v3_cal->NSampleSwitch_N) {
            Supine_SpeedGoat_v3_B.NSampleEnable = (boolean_T)(2 -
              Supine_SpeedGoat_v3_cal->NSampleEnable_ActiveLevel);
          } else {
            Supine_SpeedGoat_v3_B.NSampleEnable = (boolean_T)
              (Supine_SpeedGoat_v3_cal->NSampleEnable_ActiveLevel - 1);
            (Supine_SpeedGoat_v3_DW.NSampleEnable_Counter)++;
          }
        }
      }

      /* Switch: '<S5>/Switch' */
      if (Supine_SpeedGoat_v3_B.NSampleEnable) {
        /* Switch: '<S5>/Switch' incorporates:
         *  Constant: '<S4>/Constant1'
         */
        Supine_SpeedGoat_v3_B.Switch_g =
          Supine_SpeedGoat_v3_cal->Constant1_Value;
      } else {
        /* Switch: '<S5>/Switch' incorporates:
         *  Constant: '<S4>/Constant2'
         */
        Supine_SpeedGoat_v3_B.Switch_g =
          Supine_SpeedGoat_v3_cal->Constant2_Value;
      }

      /* End of Switch: '<S5>/Switch' */

      /* DiscretePulseGenerator: '<S4>/Hearbeat' */
      Supine_SpeedGoat_v3_B.Hearbeat = (Supine_SpeedGoat_v3_DW.clockTickCounter <
        Supine_SpeedGoat_v3_cal->Hearbeat_Duty) &&
        (Supine_SpeedGoat_v3_DW.clockTickCounter >= 0) ?
        Supine_SpeedGoat_v3_cal->Hearbeat_Amp : 0.0;

      /* DiscretePulseGenerator: '<S4>/Hearbeat' */
      if (Supine_SpeedGoat_v3_DW.clockTickCounter >=
          Supine_SpeedGoat_v3_cal->Hearbeat_Period - 1.0) {
        Supine_SpeedGoat_v3_DW.clockTickCounter = 0;
      } else {
        Supine_SpeedGoat_v3_DW.clockTickCounter++;
      }

      /* Switch: '<S4>/Switch' incorporates:
       *  Constant: '<S1>/FwdFlow_Use'
       */
      if (Supine_SpeedGoat_v3_cal->FwdFlow_Use_Value >=
          Supine_SpeedGoat_v3_cal->Switch_Threshold_i) {
        /* Switch: '<S4>/Switch' */
        Supine_SpeedGoat_v3_B.Heart_Beat = Supine_SpeedGoat_v3_B.Hearbeat;
      } else {
        /* Switch: '<S4>/Switch' */
        Supine_SpeedGoat_v3_B.Heart_Beat = 0.0;
      }

      /* End of Switch: '<S4>/Switch' */

      /* S-Function (sg_IO191_do_s): '<S4>/Digital output ' */

      /* Level2 S-Function Block: '<S4>/Digital output ' (sg_IO191_do_s) */
      {
        SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[4];
        sfcnOutputs(rts,0);
      }

      /* Product: '<S4>/Product1' incorporates:
       *  Constant: '<S1>/Sample_Trigger'
       */
      Supine_SpeedGoat_v3_B.Product1 =
        Supine_SpeedGoat_v3_cal->Sample_Trigger_Value *
        Supine_SpeedGoat_v3_B.Switch_g;

      /* Constant: '<S1>/NumActuators' */
      Supine_SpeedGoat_v3_B.NumActuators =
        Supine_SpeedGoat_v3_cal->NumActuators_Value;
    }
  }

  if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
    real_T *lastU;

    /* Update for Derivative: '<S2>/Derivative' */
    if (Supine_SpeedGoat_v3_DW.TimeStampA == (rtInf)) {
      Supine_SpeedGoat_v3_DW.TimeStampA = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeA;
    } else if (Supine_SpeedGoat_v3_DW.TimeStampB == (rtInf)) {
      Supine_SpeedGoat_v3_DW.TimeStampB = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB;
    } else if (Supine_SpeedGoat_v3_DW.TimeStampA <
               Supine_SpeedGoat_v3_DW.TimeStampB) {
      Supine_SpeedGoat_v3_DW.TimeStampA = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeA;
    } else {
      Supine_SpeedGoat_v3_DW.TimeStampB = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB;
    }

    *lastU = Supine_SpeedGoat_v3_B.D;

    /* End of Update for Derivative: '<S2>/Derivative' */

    /* Update for Derivative: '<S2>/Derivative1' */
    if (Supine_SpeedGoat_v3_DW.TimeStampA_d == (rtInf)) {
      Supine_SpeedGoat_v3_DW.TimeStampA_d = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeA_m;
    } else if (Supine_SpeedGoat_v3_DW.TimeStampB_n == (rtInf)) {
      Supine_SpeedGoat_v3_DW.TimeStampB_n = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB_m;
    } else if (Supine_SpeedGoat_v3_DW.TimeStampA_d <
               Supine_SpeedGoat_v3_DW.TimeStampB_n) {
      Supine_SpeedGoat_v3_DW.TimeStampA_d = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeA_m;
    } else {
      Supine_SpeedGoat_v3_DW.TimeStampB_n = Supine_SpeedGoat_v3_M->Timing.t[0];
      lastU = &Supine_SpeedGoat_v3_DW.LastUAtTimeB_m;
    }

    *lastU = Supine_SpeedGoat_v3_B.Sum5;

    /* End of Update for Derivative: '<S2>/Derivative1' */
    if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
      real_T tmin;

      /* Update for UniformRandomNumber: '<S2>/Uniform Random Number 0.1 ms' */
      tmin = Supine_SpeedGoat_v3_cal->UniformRandomNumber01ms_Minimum;
      Supine_SpeedGoat_v3_DW.UniformRandomNumber01ms_NextOut =
        (Supine_SpeedGoat_v3_cal->UniformRandomNumber01ms_Maximum - tmin) *
        rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed) + tmin;

      /* Update for Memory: '<S4>/Memory7' */
      Supine_SpeedGoat_v3_DW.Memory7_PreviousInput =
        Supine_SpeedGoat_v3_B.Product1;
    }
  }                                    /* end MajorTimeStep */

  if (rtmIsMajorTimeStep(Supine_SpeedGoat_v3_M)) {
    rt_ertODEUpdateContinuousStates(&Supine_SpeedGoat_v3_M->solverInfo);

    /* Update absolute time */
    /* The "clockTick0" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick0"
     * and "Timing.stepSize0". Size of "clockTick0" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick0 and the high bits
     * Timing.clockTickH0. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++Supine_SpeedGoat_v3_M->Timing.clockTick0)) {
      ++Supine_SpeedGoat_v3_M->Timing.clockTickH0;
    }

    Supine_SpeedGoat_v3_M->Timing.t[0] = rtsiGetSolverStopTime
      (&Supine_SpeedGoat_v3_M->solverInfo);

    /* Update absolute time */
    /* The "clockTick1" counts the number of times the code of this task has
     * been executed. The absolute time is the multiplication of "clockTick1"
     * and "Timing.stepSize1". Size of "clockTick1" ensures timer will not
     * overflow during the application lifespan selected.
     * Timer of this task consists of two 32 bit unsigned integers.
     * The two integers represent the low bits Timing.clockTick1 and the high bits
     * Timing.clockTickH1. When the low bit overflows to 0, the high bits increment.
     */
    if (!(++Supine_SpeedGoat_v3_M->Timing.clockTick1)) {
      ++Supine_SpeedGoat_v3_M->Timing.clockTickH1;
    }

    Supine_SpeedGoat_v3_M->Timing.t[1] =
      Supine_SpeedGoat_v3_M->Timing.clockTick1 *
      Supine_SpeedGoat_v3_M->Timing.stepSize1 +
      Supine_SpeedGoat_v3_M->Timing.clockTickH1 *
      Supine_SpeedGoat_v3_M->Timing.stepSize1 * 4294967296.0;
  }                                    /* end MajorTimeStep */
}

/* Derivatives for root system: '<Root>' */
void Supine_SpeedGoat_v3_derivatives(void)
{
  XDot_Supine_SpeedGoat_v3_T *_rtXdot;
  boolean_T lsat;
  boolean_T usat;
  _rtXdot = ((XDot_Supine_SpeedGoat_v3_T *) Supine_SpeedGoat_v3_M->derivs);

  /* Derivatives for StateSpace: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
  std::memset(&_rtXdot->uthorderBeselwithcutoff07HzbyGo[0], 0, sizeof(real_T) <<
              3U);
  for (int_T is = 0; is < 8; is++) {
    for (uint32_T ri = Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzb_de[
         static_cast<uint32_T>(is)]; ri <
         Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzb_de[is + 1U]; ri++)
    {
      _rtXdot->uthorderBeselwithcutoff07HzbyGo
        [Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_d[ri]] +=
        Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07HzbyGo[ri] *
        Supine_SpeedGoat_v3_X.uthorderBeselwithcutoff07HzbyGo
        [static_cast<uint32_T>(is)];
    }
  }

  for (uint32_T ri = Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_i[0U];
       ri < Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_i[1U]; ri++)
  {
    _rtXdot->uthorderBeselwithcutoff07HzbyGo
      [Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_a] +=
      Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_l *
      Supine_SpeedGoat_v3_B.absolutetorquewithoutpassiveinN;
  }

  /* End of Derivatives for StateSpace: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */

  /* Derivatives for StateSpace: '<S2>/Cut-off 1 Hz1' incorporates:
   *  Constant: '<Root>/POS_Offset'
   */
  _rtXdot->Cutoff1Hz1_CSTATE[0] = 0.0;
  _rtXdot->Cutoff1Hz1_CSTATE[1] = 0.0;
  for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_jc[0U]; ri <
       Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_jc[1U]; ri++) {
    _rtXdot->Cutoff1Hz1_CSTATE[Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_ir[ri]] +=
      Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_pr[ri] *
      Supine_SpeedGoat_v3_X.Cutoff1Hz1_CSTATE[0U];
  }

  for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_jc[1U]; ri <
       Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_jc[2U]; ri++) {
    _rtXdot->Cutoff1Hz1_CSTATE[Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_ir[ri]] +=
      Supine_SpeedGoat_v3_cal->Cutoff1Hz1_A_pr[ri] *
      Supine_SpeedGoat_v3_X.Cutoff1Hz1_CSTATE[1U];
  }

  for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff1Hz1_B_jc[0U]; ri <
       Supine_SpeedGoat_v3_cal->Cutoff1Hz1_B_jc[1U]; ri++) {
    _rtXdot->Cutoff1Hz1_CSTATE[Supine_SpeedGoat_v3_cal->Cutoff1Hz1_B_ir] +=
      Supine_SpeedGoat_v3_cal->Cutoff1Hz1_B_pr *
      Supine_SpeedGoat_v3_cal->POS_Offset_Value;
  }

  /* End of Derivatives for StateSpace: '<S2>/Cut-off 1 Hz1' */

  /* Derivatives for Integrator: '<S2>/Integrator' */
  lsat = (Supine_SpeedGoat_v3_X.Integrator_CSTATE <=
          Supine_SpeedGoat_v3_cal->Integrator_LowerSat);
  usat = (Supine_SpeedGoat_v3_X.Integrator_CSTATE >=
          Supine_SpeedGoat_v3_cal->Integrator_UpperSat);
  if (((!lsat) && (!usat)) || (lsat && (Supine_SpeedGoat_v3_B.I > 0.0)) || (usat
       && (Supine_SpeedGoat_v3_B.I < 0.0))) {
    _rtXdot->Integrator_CSTATE = Supine_SpeedGoat_v3_B.I;
  } else {
    /* in saturation */
    _rtXdot->Integrator_CSTATE = 0.0;
  }

  /* End of Derivatives for Integrator: '<S2>/Integrator' */

  /* Derivatives for StateSpace: '<S2>/Cut-off 35 Hz' */
  for (int_T is = 0; is < 5; is++) {
    _rtXdot->Cutoff35Hz_CSTATE[is] = 0.0;
  }

  for (int_T is = 0; is < 5; is++) {
    for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff35Hz_A_jc
         [static_cast<uint32_T>(is)]; ri <
         Supine_SpeedGoat_v3_cal->Cutoff35Hz_A_jc[is + 1U]; ri++) {
      _rtXdot->Cutoff35Hz_CSTATE[Supine_SpeedGoat_v3_cal->Cutoff35Hz_A_ir[ri]] +=
        Supine_SpeedGoat_v3_cal->Cutoff35Hz_A_pr[ri] *
        Supine_SpeedGoat_v3_X.Cutoff35Hz_CSTATE[static_cast<uint32_T>(is)];
    }
  }

  for (uint32_T ri = Supine_SpeedGoat_v3_cal->Cutoff35Hz_B_jc[0U]; ri <
       Supine_SpeedGoat_v3_cal->Cutoff35Hz_B_jc[1U]; ri++) {
    _rtXdot->Cutoff35Hz_CSTATE[Supine_SpeedGoat_v3_cal->Cutoff35Hz_B_ir[ri]] +=
      Supine_SpeedGoat_v3_cal->Cutoff35Hz_B_pr[ri] *
      Supine_SpeedGoat_v3_B.UniformRandomNumber01ms;
  }

  /* End of Derivatives for StateSpace: '<S2>/Cut-off 35 Hz' */
}

/* Model step function for TID2 */
void Supine_SpeedGoat_v3_step2(void)   /* Sample time: [0.02s, 0.0s] */
{
  int32_T tmp;

  /* RateTransition: '<S7>/Rate Transition' */
  Supine_SpeedGoat_v3_DW.RateTransition_RdBufIdx = static_cast<int8_T>
    (Supine_SpeedGoat_v3_DW.RateTransition_RdBufIdx == 0);
  tmp = Supine_SpeedGoat_v3_DW.RateTransition_RdBufIdx << 4;
  for (int32_T i = 0; i < 16; i++) {
    /* RateTransition: '<S7>/Rate Transition' */
    Supine_SpeedGoat_v3_B.RateTransition[i] =
      Supine_SpeedGoat_v3_DW.RateTransition_Buf[i + tmp];
  }

  /* S-Function (slrealtimeUDPSend): '<S7>/UDP Send1' incorporates:
   *  Constant: '<S7>/position_switch4'
   */
  {
    try {
      slrealtime::ip::udp::Socket *udpSock = reinterpret_cast<slrealtime::ip::
        udp::Socket*>(Supine_SpeedGoat_v3_DW.UDPSend1_PWORK);
      uint8_t *remoteAddress = (uint8_t *)
        Supine_SpeedGoat_v3_cal->UDPSend1_toAddress;
      uint16_t *remotePort = (uint16_t *)
        &Supine_SpeedGoat_v3_cal->UDPSend1_toPort;
      udpSock->resetRemoteEndpoint(remoteAddress, remotePort);
      int dataLen = Supine_SpeedGoat_v3_cal->position_switch4_Value_l;
      dataLen = (dataLen <= Supine_SpeedGoat_v3_DW.UDPSend1_IWORK[0]) ? dataLen :
        Supine_SpeedGoat_v3_DW.UDPSend1_IWORK[0];
      dataLen = (dataLen <= -1) ? 0 : dataLen;
      void *dataPort = &Supine_SpeedGoat_v3_B.RateTransition[0];
      size_t numBytesSend = udpSock->send((const char*)dataPort,dataLen);
    } catch (std::exception& e) {
      std::string tmp = std::string(e.what());
      static std::string eMsg = tmp;
      rtmSetErrorStatus(Supine_SpeedGoat_v3_M, eMsg.c_str());
      rtmSetStopRequested(Supine_SpeedGoat_v3_M, 1);
      ;
    }
  }

  /* Update absolute time */
  /* The "clockTick2" counts the number of times the code of this task has
   * been executed. The resolution of this integer timer is 0.02, which is the step size
   * of the task. Size of "clockTick2" ensures timer will not overflow during the
   * application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick2 and the high bits
   * Timing.clockTickH2. When the low bit overflows to 0, the high bits increment.
   */
  Supine_SpeedGoat_v3_M->Timing.clockTick2++;
  if (!Supine_SpeedGoat_v3_M->Timing.clockTick2) {
    Supine_SpeedGoat_v3_M->Timing.clockTickH2++;
  }
}

/* Model step function for TID3 */
void Supine_SpeedGoat_v3_step3(void)   /* Sample time: [0.2s, 0.0s] */
{
  real_T tmin;

  /* UniformRandomNumber: '<S9>/For PRBS with 0.2s base switch time' */
  Supine_SpeedGoat_v3_B.ForPRBSwith02sbaseswitchtime =
    Supine_SpeedGoat_v3_DW.ForPRBSwith02sbaseswitchtime_Ne;

  /* RateTransition: '<S9>/Rate Transition1' */
  Supine_SpeedGoat_v3_DW.RateTransition1_WrBufIdx = static_cast<int8_T>
    (Supine_SpeedGoat_v3_DW.RateTransition1_WrBufIdx == 0);
  Supine_SpeedGoat_v3_DW.RateTransition1_Buf[Supine_SpeedGoat_v3_DW.RateTransition1_WrBufIdx]
    = Supine_SpeedGoat_v3_B.ForPRBSwith02sbaseswitchtime;

  /* Update for UniformRandomNumber: '<S9>/For PRBS with 0.2s base switch time' */
  tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith02sbaseswitchtime_Mi;
  Supine_SpeedGoat_v3_DW.ForPRBSwith02sbaseswitchtime_Ne =
    (Supine_SpeedGoat_v3_cal->ForPRBSwith02sbaseswitchtime_Ma - tmin) *
    rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_n) + tmin;

  /* Update absolute time */
  /* The "clockTick3" counts the number of times the code of this task has
   * been executed. The resolution of this integer timer is 0.2, which is the step size
   * of the task. Size of "clockTick3" ensures timer will not overflow during the
   * application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick3 and the high bits
   * Timing.clockTickH3. When the low bit overflows to 0, the high bits increment.
   */
  Supine_SpeedGoat_v3_M->Timing.clockTick3++;
  if (!Supine_SpeedGoat_v3_M->Timing.clockTick3) {
    Supine_SpeedGoat_v3_M->Timing.clockTickH3++;
  }
}

/* Model step function for TID4 */
void Supine_SpeedGoat_v3_step4(void)   /* Sample time: [0.5s, 0.0s] */
{
  real_T tmin;

  /* UniformRandomNumber: '<S9>/For PRBS with 0.5s base switch time' */
  Supine_SpeedGoat_v3_B.ForPRBSwith05sbaseswitchtime =
    Supine_SpeedGoat_v3_DW.ForPRBSwith05sbaseswitchtime_Ne;

  /* RateTransition: '<S9>/Rate Transition2' */
  Supine_SpeedGoat_v3_DW.RateTransition2_WrBufIdx = static_cast<int8_T>
    (Supine_SpeedGoat_v3_DW.RateTransition2_WrBufIdx == 0);
  Supine_SpeedGoat_v3_DW.RateTransition2_Buf[Supine_SpeedGoat_v3_DW.RateTransition2_WrBufIdx]
    = Supine_SpeedGoat_v3_B.ForPRBSwith05sbaseswitchtime;

  /* Update for UniformRandomNumber: '<S9>/For PRBS with 0.5s base switch time' */
  tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith05sbaseswitchtime_Mi;
  Supine_SpeedGoat_v3_DW.ForPRBSwith05sbaseswitchtime_Ne =
    (Supine_SpeedGoat_v3_cal->ForPRBSwith05sbaseswitchtime_Ma - tmin) *
    rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_h) + tmin;

  /* Update absolute time */
  /* The "clockTick4" counts the number of times the code of this task has
   * been executed. The resolution of this integer timer is 0.5, which is the step size
   * of the task. Size of "clockTick4" ensures timer will not overflow during the
   * application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick4 and the high bits
   * Timing.clockTickH4. When the low bit overflows to 0, the high bits increment.
   */
  Supine_SpeedGoat_v3_M->Timing.clockTick4++;
  if (!Supine_SpeedGoat_v3_M->Timing.clockTick4) {
    Supine_SpeedGoat_v3_M->Timing.clockTickH4++;
  }
}

/* Model step function for TID5 */
void Supine_SpeedGoat_v3_step5(void)   /* Sample time: [0.7s, 0.0s] */
{
  real_T tmin;

  /* UniformRandomNumber: '<S9>/For PRBS with 0.7s base switch time' */
  Supine_SpeedGoat_v3_B.ForPRBSwith07sbaseswitchtime =
    Supine_SpeedGoat_v3_DW.ForPRBSwith07sbaseswitchtime_Ne;

  /* RateTransition: '<S9>/Rate Transition3' */
  Supine_SpeedGoat_v3_DW.RateTransition3_WrBufIdx = static_cast<int8_T>
    (Supine_SpeedGoat_v3_DW.RateTransition3_WrBufIdx == 0);
  Supine_SpeedGoat_v3_DW.RateTransition3_Buf[Supine_SpeedGoat_v3_DW.RateTransition3_WrBufIdx]
    = Supine_SpeedGoat_v3_B.ForPRBSwith07sbaseswitchtime;

  /* Update for UniformRandomNumber: '<S9>/For PRBS with 0.7s base switch time' */
  tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith07sbaseswitchtime_Mi;
  Supine_SpeedGoat_v3_DW.ForPRBSwith07sbaseswitchtime_Ne =
    (Supine_SpeedGoat_v3_cal->ForPRBSwith07sbaseswitchtime_Ma - tmin) *
    rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_g) + tmin;

  /* Update absolute time */
  /* The "clockTick5" counts the number of times the code of this task has
   * been executed. The resolution of this integer timer is 0.7, which is the step size
   * of the task. Size of "clockTick5" ensures timer will not overflow during the
   * application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick5 and the high bits
   * Timing.clockTickH5. When the low bit overflows to 0, the high bits increment.
   */
  Supine_SpeedGoat_v3_M->Timing.clockTick5++;
  if (!Supine_SpeedGoat_v3_M->Timing.clockTick5) {
    Supine_SpeedGoat_v3_M->Timing.clockTickH5++;
  }
}

/* Model step function for TID6 */
void Supine_SpeedGoat_v3_step6(void)   /* Sample time: [1.0s, 0.0s] */
{
  real_T tmin;

  /* UniformRandomNumber: '<S9>/For PRBS with 1s base switch time' */
  Supine_SpeedGoat_v3_B.ForPRBSwith1sbaseswitchtime =
    Supine_SpeedGoat_v3_DW.ForPRBSwith1sbaseswitchtime_Nex;

  /* RateTransition: '<S9>/Rate Transition4' */
  Supine_SpeedGoat_v3_DW.RateTransition4_WrBufIdx = static_cast<int8_T>
    (Supine_SpeedGoat_v3_DW.RateTransition4_WrBufIdx == 0);
  Supine_SpeedGoat_v3_DW.RateTransition4_Buf[Supine_SpeedGoat_v3_DW.RateTransition4_WrBufIdx]
    = Supine_SpeedGoat_v3_B.ForPRBSwith1sbaseswitchtime;

  /* Update for UniformRandomNumber: '<S9>/For PRBS with 1s base switch time' */
  tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith1sbaseswitchtime_Min;
  Supine_SpeedGoat_v3_DW.ForPRBSwith1sbaseswitchtime_Nex =
    (Supine_SpeedGoat_v3_cal->ForPRBSwith1sbaseswitchtime_Max - tmin) *
    rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_j) + tmin;

  /* Update absolute time */
  /* The "clockTick6" counts the number of times the code of this task has
   * been executed. The resolution of this integer timer is 1.0, which is the step size
   * of the task. Size of "clockTick6" ensures timer will not overflow during the
   * application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick6 and the high bits
   * Timing.clockTickH6. When the low bit overflows to 0, the high bits increment.
   */
  Supine_SpeedGoat_v3_M->Timing.clockTick6++;
  if (!Supine_SpeedGoat_v3_M->Timing.clockTick6) {
    Supine_SpeedGoat_v3_M->Timing.clockTickH6++;
  }
}

/* Model step function for TID7 */
void Supine_SpeedGoat_v3_step7(void)   /* Sample time: [2.0s, 0.0s] */
{
  real_T tmin;

  /* UniformRandomNumber: '<S9>/For PRBS with 2s base switch time' */
  Supine_SpeedGoat_v3_B.ForPRBSwith2sbaseswitchtime =
    Supine_SpeedGoat_v3_DW.ForPRBSwith2sbaseswitchtime_Nex;

  /* RateTransition: '<S9>/Rate Transition5' */
  Supine_SpeedGoat_v3_DW.RateTransition5_WrBufIdx = static_cast<int8_T>
    (Supine_SpeedGoat_v3_DW.RateTransition5_WrBufIdx == 0);
  Supine_SpeedGoat_v3_DW.RateTransition5_Buf[Supine_SpeedGoat_v3_DW.RateTransition5_WrBufIdx]
    = Supine_SpeedGoat_v3_B.ForPRBSwith2sbaseswitchtime;

  /* Update for UniformRandomNumber: '<S9>/For PRBS with 2s base switch time' */
  tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith2sbaseswitchtime_Min;
  Supine_SpeedGoat_v3_DW.ForPRBSwith2sbaseswitchtime_Nex =
    (Supine_SpeedGoat_v3_cal->ForPRBSwith2sbaseswitchtime_Max - tmin) *
    rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_b) + tmin;

  /* Update absolute time */
  /* The "clockTick7" counts the number of times the code of this task has
   * been executed. The resolution of this integer timer is 2.0, which is the step size
   * of the task. Size of "clockTick7" ensures timer will not overflow during the
   * application lifespan selected.
   * Timer of this task consists of two 32 bit unsigned integers.
   * The two integers represent the low bits Timing.clockTick7 and the high bits
   * Timing.clockTickH7. When the low bit overflows to 0, the high bits increment.
   */
  Supine_SpeedGoat_v3_M->Timing.clockTick7++;
  if (!Supine_SpeedGoat_v3_M->Timing.clockTick7) {
    Supine_SpeedGoat_v3_M->Timing.clockTickH7++;
  }
}

/* Model initialize function */
void Supine_SpeedGoat_v3_initialize(void)
{
  /* Registration code */

  /* initialize non-finites */
  rt_InitInfAndNaN(sizeof(real_T));

  /* Set task counter limit used by the static main program */
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[0] = 1;
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[1] = 1;
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[2] = 20;
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[3] = 200;
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[4] = 500;
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[5] = 700;
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[6] = 1000;
  (Supine_SpeedGoat_v3_M)->Timing.TaskCounters.cLimit[7] = 2000;

  {
    /* Setup solver object */
    rtsiSetSimTimeStepPtr(&Supine_SpeedGoat_v3_M->solverInfo,
                          &Supine_SpeedGoat_v3_M->Timing.simTimeStep);
    rtsiSetTPtr(&Supine_SpeedGoat_v3_M->solverInfo, &rtmGetTPtr
                (Supine_SpeedGoat_v3_M));
    rtsiSetStepSizePtr(&Supine_SpeedGoat_v3_M->solverInfo,
                       &Supine_SpeedGoat_v3_M->Timing.stepSize0);
    rtsiSetdXPtr(&Supine_SpeedGoat_v3_M->solverInfo,
                 &Supine_SpeedGoat_v3_M->derivs);
    rtsiSetContStatesPtr(&Supine_SpeedGoat_v3_M->solverInfo, (real_T **)
                         &Supine_SpeedGoat_v3_M->contStates);
    rtsiSetNumContStatesPtr(&Supine_SpeedGoat_v3_M->solverInfo,
      &Supine_SpeedGoat_v3_M->Sizes.numContStates);
    rtsiSetNumPeriodicContStatesPtr(&Supine_SpeedGoat_v3_M->solverInfo,
      &Supine_SpeedGoat_v3_M->Sizes.numPeriodicContStates);
    rtsiSetPeriodicContStateIndicesPtr(&Supine_SpeedGoat_v3_M->solverInfo,
      &Supine_SpeedGoat_v3_M->periodicContStateIndices);
    rtsiSetPeriodicContStateRangesPtr(&Supine_SpeedGoat_v3_M->solverInfo,
      &Supine_SpeedGoat_v3_M->periodicContStateRanges);
    rtsiSetErrorStatusPtr(&Supine_SpeedGoat_v3_M->solverInfo,
                          (&rtmGetErrorStatus(Supine_SpeedGoat_v3_M)));
    rtsiSetRTModelPtr(&Supine_SpeedGoat_v3_M->solverInfo, Supine_SpeedGoat_v3_M);
  }

  rtsiSetSimTimeStep(&Supine_SpeedGoat_v3_M->solverInfo, MAJOR_TIME_STEP);
  Supine_SpeedGoat_v3_M->intgData.y = Supine_SpeedGoat_v3_M->odeY;
  Supine_SpeedGoat_v3_M->intgData.f[0] = Supine_SpeedGoat_v3_M->odeF[0];
  Supine_SpeedGoat_v3_M->intgData.f[1] = Supine_SpeedGoat_v3_M->odeF[1];
  Supine_SpeedGoat_v3_M->intgData.f[2] = Supine_SpeedGoat_v3_M->odeF[2];
  Supine_SpeedGoat_v3_M->intgData.f[3] = Supine_SpeedGoat_v3_M->odeF[3];
  Supine_SpeedGoat_v3_M->intgData.f[4] = Supine_SpeedGoat_v3_M->odeF[4];
  Supine_SpeedGoat_v3_M->intgData.f[5] = Supine_SpeedGoat_v3_M->odeF[5];
  Supine_SpeedGoat_v3_M->contStates = ((X_Supine_SpeedGoat_v3_T *)
    &Supine_SpeedGoat_v3_X);
  rtsiSetSolverData(&Supine_SpeedGoat_v3_M->solverInfo, static_cast<void *>
                    (&Supine_SpeedGoat_v3_M->intgData));
  rtsiSetIsMinorTimeStepWithModeChange(&Supine_SpeedGoat_v3_M->solverInfo, false);
  rtsiSetSolverName(&Supine_SpeedGoat_v3_M->solverInfo,"ode5");
  Supine_SpeedGoat_v3_M->solverInfoPtr = (&Supine_SpeedGoat_v3_M->solverInfo);

  /* Initialize timing info */
  {
    int_T *mdlTsMap = Supine_SpeedGoat_v3_M->Timing.sampleTimeTaskIDArray;
    int_T i;
    for (i = 0; i < 8; i++) {
      mdlTsMap[i] = i;
    }

    /* polyspace +2 MISRA2012:D4.1 [Justified:Low] "Supine_SpeedGoat_v3_M points to
       static memory which is guaranteed to be non-NULL" */
    Supine_SpeedGoat_v3_M->Timing.sampleTimeTaskIDPtr = (&mdlTsMap[0]);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes =
      (&Supine_SpeedGoat_v3_M->Timing.sampleTimesArray[0]);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes =
      (&Supine_SpeedGoat_v3_M->Timing.offsetTimesArray[0]);

    /* task periods */
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[0] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[1] = (0.001);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[2] = (0.02);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[3] = (0.2);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[4] = (0.5);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[5] = (0.7);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[6] = (1.0);
    Supine_SpeedGoat_v3_M->Timing.sampleTimes[7] = (2.0);

    /* task offsets */
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[0] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[1] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[2] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[3] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[4] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[5] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[6] = (0.0);
    Supine_SpeedGoat_v3_M->Timing.offsetTimes[7] = (0.0);
  }

  rtmSetTPtr(Supine_SpeedGoat_v3_M, &Supine_SpeedGoat_v3_M->Timing.tArray[0]);

  {
    int_T *mdlSampleHits = Supine_SpeedGoat_v3_M->Timing.sampleHitArray;
    int_T *mdlPerTaskSampleHits =
      Supine_SpeedGoat_v3_M->Timing.perTaskSampleHitsArray;
    Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits = (&mdlPerTaskSampleHits[0]);
    mdlSampleHits[0] = 1;
    Supine_SpeedGoat_v3_M->Timing.sampleHits = (&mdlSampleHits[0]);
  }

  rtmSetTFinal(Supine_SpeedGoat_v3_M, -1);
  Supine_SpeedGoat_v3_M->Timing.stepSize0 = 0.001;
  Supine_SpeedGoat_v3_M->Timing.stepSize1 = 0.001;
  Supine_SpeedGoat_v3_M->solverInfoPtr = (&Supine_SpeedGoat_v3_M->solverInfo);
  Supine_SpeedGoat_v3_M->Timing.stepSize = (0.001);
  rtsiSetFixedStepSize(&Supine_SpeedGoat_v3_M->solverInfo, 0.001);
  rtsiSetSolverMode(&Supine_SpeedGoat_v3_M->solverInfo, SOLVER_MODE_MULTITASKING);

  /* block I/O */
  (void) std::memset((static_cast<void *>(&Supine_SpeedGoat_v3_B)), 0,
                     sizeof(B_Supine_SpeedGoat_v3_T));

  /* states (continuous) */
  {
    (void) std::memset(static_cast<void *>(&Supine_SpeedGoat_v3_X), 0,
                       sizeof(X_Supine_SpeedGoat_v3_T));
  }

  /* states (dwork) */
  (void) std::memset(static_cast<void *>(&Supine_SpeedGoat_v3_DW), 0,
                     sizeof(DW_Supine_SpeedGoat_v3_T));

  /* external inputs */
  (void)std::memset(&Supine_SpeedGoat_v3_U, 0, sizeof(ExtU_Supine_SpeedGoat_v3_T));

  /* child S-Function registration */
  {
    RTWSfcnInfo *sfcnInfo = &Supine_SpeedGoat_v3_M->NonInlinedSFcns.sfcnInfo;
    Supine_SpeedGoat_v3_M->sfcnInfo = (sfcnInfo);
    rtssSetErrorStatusPtr(sfcnInfo, (&rtmGetErrorStatus(Supine_SpeedGoat_v3_M)));
    Supine_SpeedGoat_v3_M->Sizes.numSampTimes = (8);
    rtssSetNumRootSampTimesPtr(sfcnInfo,
      &Supine_SpeedGoat_v3_M->Sizes.numSampTimes);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[0] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[0]);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[1] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[1]);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[2] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[2]);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[3] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[3]);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[4] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[4]);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[5] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[5]);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[6] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[6]);
    Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs[7] = &(rtmGetTPtr
      (Supine_SpeedGoat_v3_M)[7]);
    rtssSetTPtrPtr(sfcnInfo,Supine_SpeedGoat_v3_M->NonInlinedSFcns.taskTimePtrs);
    rtssSetTStartPtr(sfcnInfo, &rtmGetTStart(Supine_SpeedGoat_v3_M));
    rtssSetTFinalPtr(sfcnInfo, &rtmGetTFinal(Supine_SpeedGoat_v3_M));
    rtssSetTimeOfLastOutputPtr(sfcnInfo, &rtmGetTimeOfLastOutput
      (Supine_SpeedGoat_v3_M));
    rtssSetStepSizePtr(sfcnInfo, &Supine_SpeedGoat_v3_M->Timing.stepSize);
    rtssSetStopRequestedPtr(sfcnInfo, &rtmGetStopRequested(Supine_SpeedGoat_v3_M));
    rtssSetDerivCacheNeedsResetPtr(sfcnInfo,
      &Supine_SpeedGoat_v3_M->derivCacheNeedsReset);
    rtssSetZCCacheNeedsResetPtr(sfcnInfo,
      &Supine_SpeedGoat_v3_M->zCCacheNeedsReset);
    rtssSetContTimeOutputInconsistentWithStateAtMajorStepPtr(sfcnInfo,
      &Supine_SpeedGoat_v3_M->CTOutputIncnstWithState);
    rtssSetSampleHitsPtr(sfcnInfo, &Supine_SpeedGoat_v3_M->Timing.sampleHits);
    rtssSetPerTaskSampleHitsPtr(sfcnInfo,
      &Supine_SpeedGoat_v3_M->Timing.perTaskSampleHits);
    rtssSetSimModePtr(sfcnInfo, &Supine_SpeedGoat_v3_M->simMode);
    rtssSetSolverInfoPtr(sfcnInfo, &Supine_SpeedGoat_v3_M->solverInfoPtr);
  }

  Supine_SpeedGoat_v3_M->Sizes.numSFcns = (5);

  /* register each child */
  {
    (void) std::memset(static_cast<void *>
                       (&Supine_SpeedGoat_v3_M->NonInlinedSFcns.childSFunctions
                        [0]), 0,
                       5*sizeof(SimStruct));
    Supine_SpeedGoat_v3_M->childSfunctions =
      (&Supine_SpeedGoat_v3_M->NonInlinedSFcns.childSFunctionPtrs[0]);

    {
      int_T i;
      for (i = 0; i < 5; i++) {
        Supine_SpeedGoat_v3_M->childSfunctions[i] =
          (&Supine_SpeedGoat_v3_M->NonInlinedSFcns.childSFunctions[i]);
      }
    }

    /* Level2 S-Function Block: Supine_SpeedGoat_v3/<Root>/Setup  (sg_IO191_setup_s) */
    {
      SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[0];

      /* timing info */
      time_T *sfcnPeriod =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn0.sfcnPeriod;
      time_T *sfcnOffset =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn0.sfcnOffset;
      int_T *sfcnTsMap = Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn0.sfcnTsMap;
      (void) std::memset(static_cast<void*>(sfcnPeriod), 0,
                         sizeof(time_T)*1);
      (void) std::memset(static_cast<void*>(sfcnOffset), 0,
                         sizeof(time_T)*1);
      ssSetSampleTimePtr(rts, &sfcnPeriod[0]);
      ssSetOffsetTimePtr(rts, &sfcnOffset[0]);
      ssSetSampleTimeTaskIDPtr(rts, sfcnTsMap);

      {
        ssSetBlkInfo2Ptr(rts, &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfo2[0]);
        ssSetBlkInfoSLSizePtr(rts,
                              &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfoSLSize
                              [0]);
      }

      _ssSetBlkInfo2PortInfo2Ptr(rts,
        &Supine_SpeedGoat_v3_M->NonInlinedSFcns.inputOutputPortInfo2[0]);

      /* Set up the mdlInfo pointer */
      ssSetRTWSfcnInfo(rts, Supine_SpeedGoat_v3_M->sfcnInfo);

      /* Allocate memory of model methods 2 */
      {
        ssSetModelMethods2(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods2[0]);
      }

      /* Allocate memory of model methods 3 */
      {
        ssSetModelMethods3(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods3[0]);
      }

      /* Allocate memory of model methods 4 */
      {
        ssSetModelMethods4(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods4[0]);
      }

      /* Allocate memory for states auxilliary information */
      {
        ssSetStatesInfo2(rts,
                         &Supine_SpeedGoat_v3_M->NonInlinedSFcns.statesInfo2[0]);
        ssSetPeriodicStatesInfo(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.periodicStatesInfo[0]);
      }

      /* path info */
      ssSetModelName(rts, "Setup ");
      ssSetPath(rts, "Supine_SpeedGoat_v3/Setup ");
      ssSetRTModel(rts,Supine_SpeedGoat_v3_M);
      ssSetParentSS(rts, (NULL));
      ssSetRootSS(rts, rts);
      ssSetVersion(rts, SIMSTRUCT_VERSION_LEVEL2);

      /* parameters */
      {
        mxArray **sfcnParams = (mxArray **)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn0.params;
        ssSetSFcnParamsCount(rts, 9);
        ssSetSFcnParamsPtr(rts, &sfcnParams[0]);
        ssSetSFcnParam(rts, 0, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P1_Size);
        ssSetSFcnParam(rts, 1, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P2_Size);
        ssSetSFcnParam(rts, 2, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P3_Size);
        ssSetSFcnParam(rts, 3, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P4_Size);
        ssSetSFcnParam(rts, 4, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P5_Size);
        ssSetSFcnParam(rts, 5, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P6_Size);
        ssSetSFcnParam(rts, 6, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P7_Size);
        ssSetSFcnParam(rts, 7, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P8_Size);
        ssSetSFcnParam(rts, 8, (mxArray*)Supine_SpeedGoat_v3_cal->Setup_P9_Size);
      }

      /* work vectors */
      ssSetRWork(rts, (real_T *) &Supine_SpeedGoat_v3_DW.Setup_RWORK[0]);
      ssSetPWork(rts, (void **) &Supine_SpeedGoat_v3_DW.Setup_PWORK);

      {
        struct _ssDWorkRecord *dWorkRecord = (struct _ssDWorkRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn0.dWork;
        struct _ssDWorkAuxRecord *dWorkAuxRecord = (struct _ssDWorkAuxRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn0.dWorkAux;
        ssSetSFcnDWork(rts, dWorkRecord);
        ssSetSFcnDWorkAux(rts, dWorkAuxRecord);
        _ssSetNumDWork(rts, 2);

        /* RWORK */
        ssSetDWorkWidthAsInt(rts, 0, 2);
        ssSetDWorkDataType(rts, 0,SS_DOUBLE);
        ssSetDWorkComplexSignal(rts, 0, 0);
        ssSetDWork(rts, 0, &Supine_SpeedGoat_v3_DW.Setup_RWORK[0]);

        /* PWORK */
        ssSetDWorkWidthAsInt(rts, 1, 1);
        ssSetDWorkDataType(rts, 1,SS_POINTER);
        ssSetDWorkComplexSignal(rts, 1, 0);
        ssSetDWork(rts, 1, &Supine_SpeedGoat_v3_DW.Setup_PWORK);
      }

      /* registration */
      sg_IO191_setup_s(rts);
      sfcnInitializeSizes(rts);
      sfcnInitializeSampleTimes(rts);

      /* adjust sample time */
      ssSetSampleTime(rts, 0, 0.001);
      ssSetOffsetTime(rts, 0, 0.0);
      sfcnTsMap[0] = 1;

      /* set compiled values of dynamic vector attributes */
      ssSetNumNonsampledZCsAsInt(rts, 0);

      /* Update connectivity flags for each port */
      /* Update the BufferDstPort flags for each input port */
    }

    /* Level2 S-Function Block: Supine_SpeedGoat_v3/<Root>/Analog input  (sg_IO191_ad_s) */
    {
      SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[1];

      /* timing info */
      time_T *sfcnPeriod =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.sfcnPeriod;
      time_T *sfcnOffset =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.sfcnOffset;
      int_T *sfcnTsMap = Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.sfcnTsMap;
      (void) std::memset(static_cast<void*>(sfcnPeriod), 0,
                         sizeof(time_T)*1);
      (void) std::memset(static_cast<void*>(sfcnOffset), 0,
                         sizeof(time_T)*1);
      ssSetSampleTimePtr(rts, &sfcnPeriod[0]);
      ssSetOffsetTimePtr(rts, &sfcnOffset[0]);
      ssSetSampleTimeTaskIDPtr(rts, sfcnTsMap);

      {
        ssSetBlkInfo2Ptr(rts, &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfo2[1]);
        ssSetBlkInfoSLSizePtr(rts,
                              &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfoSLSize
                              [1]);
      }

      _ssSetBlkInfo2PortInfo2Ptr(rts,
        &Supine_SpeedGoat_v3_M->NonInlinedSFcns.inputOutputPortInfo2[1]);

      /* Set up the mdlInfo pointer */
      ssSetRTWSfcnInfo(rts, Supine_SpeedGoat_v3_M->sfcnInfo);

      /* Allocate memory of model methods 2 */
      {
        ssSetModelMethods2(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods2[1]);
      }

      /* Allocate memory of model methods 3 */
      {
        ssSetModelMethods3(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods3[1]);
      }

      /* Allocate memory of model methods 4 */
      {
        ssSetModelMethods4(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods4[1]);
      }

      /* Allocate memory for states auxilliary information */
      {
        ssSetStatesInfo2(rts,
                         &Supine_SpeedGoat_v3_M->NonInlinedSFcns.statesInfo2[1]);
        ssSetPeriodicStatesInfo(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.periodicStatesInfo[1]);
      }

      /* outputs */
      {
        ssSetPortInfoForOutputs(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.outputPortInfo[0]);
        rts->blkInfo.blkInfo2->blkInfoSLSize->outputs =
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.outputPortInfoSLSize[0];
        _ssSetNumOutputPorts(rts, 8);
        _ssSetPortInfo2ForOutputUnits(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.outputPortUnits[0]);
        ssSetOutputPortUnit(rts, 0, 0);
        ssSetOutputPortUnit(rts, 1, 0);
        ssSetOutputPortUnit(rts, 2, 0);
        ssSetOutputPortUnit(rts, 3, 0);
        ssSetOutputPortUnit(rts, 4, 0);
        ssSetOutputPortUnit(rts, 5, 0);
        ssSetOutputPortUnit(rts, 6, 0);
        ssSetOutputPortUnit(rts, 7, 0);
        _ssSetPortInfo2ForOutputCoSimAttribute(rts,
          &Supine_SpeedGoat_v3_M->
          NonInlinedSFcns.Sfcn1.outputPortCoSimAttribute[0]);
        ssSetOutputPortIsContinuousQuantity(rts, 0, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 1, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 2, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 3, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 4, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 5, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 6, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 7, 0);

        /* port 0 */
        {
          _ssSetOutputPortNumDimensions(rts, 0, 1);
          ssSetOutputPortWidthAsInt(rts, 0, 1);
          ssSetOutputPortSignal(rts, 0, ((real_T *)
            &Supine_SpeedGoat_v3_B.Torque));
        }

        /* port 1 */
        {
          _ssSetOutputPortNumDimensions(rts, 1, 1);
          ssSetOutputPortWidthAsInt(rts, 1, 1);
          ssSetOutputPortSignal(rts, 1, ((real_T *)
            &Supine_SpeedGoat_v3_B.Analoginput_o2));
        }

        /* port 2 */
        {
          _ssSetOutputPortNumDimensions(rts, 2, 1);
          ssSetOutputPortWidthAsInt(rts, 2, 1);
          ssSetOutputPortSignal(rts, 2, ((real_T *)
            &Supine_SpeedGoat_v3_B.Analoginput_o3));
        }

        /* port 3 */
        {
          _ssSetOutputPortNumDimensions(rts, 3, 1);
          ssSetOutputPortWidthAsInt(rts, 3, 1);
          ssSetOutputPortSignal(rts, 3, ((real_T *)
            &Supine_SpeedGoat_v3_B.Analoginput_o4));
        }

        /* port 4 */
        {
          _ssSetOutputPortNumDimensions(rts, 4, 1);
          ssSetOutputPortWidthAsInt(rts, 4, 1);
          ssSetOutputPortSignal(rts, 4, ((real_T *)
            &Supine_SpeedGoat_v3_B.Possition));
        }

        /* port 5 */
        {
          _ssSetOutputPortNumDimensions(rts, 5, 1);
          ssSetOutputPortWidthAsInt(rts, 5, 1);
          ssSetOutputPortSignal(rts, 5, ((real_T *)
            &Supine_SpeedGoat_v3_B.Torque_c));
        }

        /* port 6 */
        {
          _ssSetOutputPortNumDimensions(rts, 6, 1);
          ssSetOutputPortWidthAsInt(rts, 6, 1);
          ssSetOutputPortSignal(rts, 6, ((real_T *)
            &Supine_SpeedGoat_v3_B.Analoginput_o7));
        }

        /* port 7 */
        {
          _ssSetOutputPortNumDimensions(rts, 7, 1);
          ssSetOutputPortWidthAsInt(rts, 7, 1);
          ssSetOutputPortSignal(rts, 7, ((real_T *)
            &Supine_SpeedGoat_v3_B.Analoginput_o8));
        }
      }

      /* path info */
      ssSetModelName(rts, "Analog input ");
      ssSetPath(rts, "Supine_SpeedGoat_v3/Analog input ");
      ssSetRTModel(rts,Supine_SpeedGoat_v3_M);
      ssSetParentSS(rts, (NULL));
      ssSetRootSS(rts, rts);
      ssSetVersion(rts, SIMSTRUCT_VERSION_LEVEL2);

      /* parameters */
      {
        mxArray **sfcnParams = (mxArray **)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.params;
        ssSetSFcnParamsCount(rts, 9);
        ssSetSFcnParamsPtr(rts, &sfcnParams[0]);
        ssSetSFcnParam(rts, 0, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P1_Size);
        ssSetSFcnParam(rts, 1, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P2_Size);
        ssSetSFcnParam(rts, 2, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P3_Size);
        ssSetSFcnParam(rts, 3, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P4_Size);
        ssSetSFcnParam(rts, 4, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P5_Size);
        ssSetSFcnParam(rts, 5, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P6_Size);
        ssSetSFcnParam(rts, 6, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P7_Size);
        ssSetSFcnParam(rts, 7, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P8_Size);
        ssSetSFcnParam(rts, 8, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analoginput_P9_Size);
      }

      /* work vectors */
      ssSetIWork(rts, (int_T *) &Supine_SpeedGoat_v3_DW.Analoginput_IWORK[0]);
      ssSetPWork(rts, (void **) &Supine_SpeedGoat_v3_DW.Analoginput_PWORK);

      {
        struct _ssDWorkRecord *dWorkRecord = (struct _ssDWorkRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.dWork;
        struct _ssDWorkAuxRecord *dWorkAuxRecord = (struct _ssDWorkAuxRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn1.dWorkAux;
        ssSetSFcnDWork(rts, dWorkRecord);
        ssSetSFcnDWorkAux(rts, dWorkAuxRecord);
        _ssSetNumDWork(rts, 2);

        /* IWORK */
        ssSetDWorkWidthAsInt(rts, 0, 2);
        ssSetDWorkDataType(rts, 0,SS_INTEGER);
        ssSetDWorkComplexSignal(rts, 0, 0);
        ssSetDWork(rts, 0, &Supine_SpeedGoat_v3_DW.Analoginput_IWORK[0]);

        /* PWORK */
        ssSetDWorkWidthAsInt(rts, 1, 1);
        ssSetDWorkDataType(rts, 1,SS_POINTER);
        ssSetDWorkComplexSignal(rts, 1, 0);
        ssSetDWork(rts, 1, &Supine_SpeedGoat_v3_DW.Analoginput_PWORK);
      }

      /* registration */
      sg_IO191_ad_s(rts);
      sfcnInitializeSizes(rts);
      sfcnInitializeSampleTimes(rts);

      /* adjust sample time */
      ssSetSampleTime(rts, 0, 0.001);
      ssSetOffsetTime(rts, 0, 0.0);
      sfcnTsMap[0] = 1;

      /* set compiled values of dynamic vector attributes */
      ssSetNumNonsampledZCsAsInt(rts, 0);

      /* Update connectivity flags for each port */
      _ssSetOutputPortConnected(rts, 0, 0);
      _ssSetOutputPortConnected(rts, 1, 0);
      _ssSetOutputPortConnected(rts, 2, 0);
      _ssSetOutputPortConnected(rts, 3, 0);
      _ssSetOutputPortConnected(rts, 4, 1);
      _ssSetOutputPortConnected(rts, 5, 1);
      _ssSetOutputPortConnected(rts, 6, 0);
      _ssSetOutputPortConnected(rts, 7, 0);
      _ssSetOutputPortBeingMerged(rts, 0, 0);
      _ssSetOutputPortBeingMerged(rts, 1, 0);
      _ssSetOutputPortBeingMerged(rts, 2, 0);
      _ssSetOutputPortBeingMerged(rts, 3, 0);
      _ssSetOutputPortBeingMerged(rts, 4, 0);
      _ssSetOutputPortBeingMerged(rts, 5, 0);
      _ssSetOutputPortBeingMerged(rts, 6, 0);
      _ssSetOutputPortBeingMerged(rts, 7, 0);

      /* Update the BufferDstPort flags for each input port */
    }

    /* Level2 S-Function Block: Supine_SpeedGoat_v3/<Root>/Analog output  (sg_IO191_da_s) */
    {
      SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[2];

      /* timing info */
      time_T *sfcnPeriod =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.sfcnPeriod;
      time_T *sfcnOffset =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.sfcnOffset;
      int_T *sfcnTsMap = Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.sfcnTsMap;
      (void) std::memset(static_cast<void*>(sfcnPeriod), 0,
                         sizeof(time_T)*1);
      (void) std::memset(static_cast<void*>(sfcnOffset), 0,
                         sizeof(time_T)*1);
      ssSetSampleTimePtr(rts, &sfcnPeriod[0]);
      ssSetOffsetTimePtr(rts, &sfcnOffset[0]);
      ssSetSampleTimeTaskIDPtr(rts, sfcnTsMap);

      {
        ssSetBlkInfo2Ptr(rts, &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfo2[2]);
        ssSetBlkInfoSLSizePtr(rts,
                              &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfoSLSize
                              [2]);
      }

      _ssSetBlkInfo2PortInfo2Ptr(rts,
        &Supine_SpeedGoat_v3_M->NonInlinedSFcns.inputOutputPortInfo2[2]);

      /* Set up the mdlInfo pointer */
      ssSetRTWSfcnInfo(rts, Supine_SpeedGoat_v3_M->sfcnInfo);

      /* Allocate memory of model methods 2 */
      {
        ssSetModelMethods2(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods2[2]);
      }

      /* Allocate memory of model methods 3 */
      {
        ssSetModelMethods3(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods3[2]);
      }

      /* Allocate memory of model methods 4 */
      {
        ssSetModelMethods4(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods4[2]);
      }

      /* Allocate memory for states auxilliary information */
      {
        ssSetStatesInfo2(rts,
                         &Supine_SpeedGoat_v3_M->NonInlinedSFcns.statesInfo2[2]);
        ssSetPeriodicStatesInfo(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.periodicStatesInfo[2]);
      }

      /* inputs */
      {
        _ssSetNumInputPorts(rts, 4);
        ssSetPortInfoForInputs(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.inputPortInfo[0]);
        rts->blkInfo.blkInfo2->blkInfoSLSize->inputs =
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.inputPortInfoSLSize[0];
        _ssSetPortInfo2ForInputUnits(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.inputPortUnits[0]);
        ssSetInputPortUnit(rts, 0, 0);
        ssSetInputPortUnit(rts, 1, 0);
        ssSetInputPortUnit(rts, 2, 0);
        ssSetInputPortUnit(rts, 3, 0);
        _ssSetPortInfo2ForInputCoSimAttribute(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.inputPortCoSimAttribute
          [0]);
        ssSetInputPortIsContinuousQuantity(rts, 0, 0);
        ssSetInputPortIsContinuousQuantity(rts, 1, 0);
        ssSetInputPortIsContinuousQuantity(rts, 2, 0);
        ssSetInputPortIsContinuousQuantity(rts, 3, 0);

        /* port 0 */
        {
          ssSetInputPortRequiredContiguous(rts, 0, 1);
          ssSetInputPortSignal(rts, 0, &Supine_SpeedGoat_v3_B.Gain);
          _ssSetInputPortNumDimensions(rts, 0, 1);
          ssSetInputPortWidthAsInt(rts, 0, 1);
        }

        /* port 1 */
        {
          ssSetInputPortRequiredContiguous(rts, 1, 1);
          ssSetInputPortSignal(rts, 1, &Supine_SpeedGoat_v3_B.Gain3);
          _ssSetInputPortNumDimensions(rts, 1, 1);
          ssSetInputPortWidthAsInt(rts, 1, 1);
        }

        /* port 2 */
        {
          ssSetInputPortRequiredContiguous(rts, 2, 1);
          ssSetInputPortSignal(rts, 2, &Supine_SpeedGoat_v3_B.Gain1);
          _ssSetInputPortNumDimensions(rts, 2, 1);
          ssSetInputPortWidthAsInt(rts, 2, 1);
        }

        /* port 3 */
        {
          ssSetInputPortRequiredContiguous(rts, 3, 1);
          ssSetInputPortSignal(rts, 3, &Supine_SpeedGoat_v3_B.Possition);
          _ssSetInputPortNumDimensions(rts, 3, 1);
          ssSetInputPortWidthAsInt(rts, 3, 1);
        }
      }

      /* path info */
      ssSetModelName(rts, "Analog output ");
      ssSetPath(rts, "Supine_SpeedGoat_v3/Analog output ");
      ssSetRTModel(rts,Supine_SpeedGoat_v3_M);
      ssSetParentSS(rts, (NULL));
      ssSetRootSS(rts, rts);
      ssSetVersion(rts, SIMSTRUCT_VERSION_LEVEL2);

      /* parameters */
      {
        mxArray **sfcnParams = (mxArray **)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.params;
        ssSetSFcnParamsCount(rts, 7);
        ssSetSFcnParamsPtr(rts, &sfcnParams[0]);
        ssSetSFcnParam(rts, 0, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analogoutput_P1_Size);
        ssSetSFcnParam(rts, 1, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analogoutput_P2_Size);
        ssSetSFcnParam(rts, 2, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analogoutput_P3_Size);
        ssSetSFcnParam(rts, 3, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analogoutput_P4_Size);
        ssSetSFcnParam(rts, 4, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analogoutput_P5_Size);
        ssSetSFcnParam(rts, 5, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analogoutput_P6_Size);
        ssSetSFcnParam(rts, 6, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Analogoutput_P7_Size);
      }

      /* work vectors */
      ssSetPWork(rts, (void **) &Supine_SpeedGoat_v3_DW.Analogoutput_PWORK);

      {
        struct _ssDWorkRecord *dWorkRecord = (struct _ssDWorkRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.dWork;
        struct _ssDWorkAuxRecord *dWorkAuxRecord = (struct _ssDWorkAuxRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn2.dWorkAux;
        ssSetSFcnDWork(rts, dWorkRecord);
        ssSetSFcnDWorkAux(rts, dWorkAuxRecord);
        _ssSetNumDWork(rts, 1);

        /* PWORK */
        ssSetDWorkWidthAsInt(rts, 0, 1);
        ssSetDWorkDataType(rts, 0,SS_POINTER);
        ssSetDWorkComplexSignal(rts, 0, 0);
        ssSetDWork(rts, 0, &Supine_SpeedGoat_v3_DW.Analogoutput_PWORK);
      }

      /* registration */
      sg_IO191_da_s(rts);
      sfcnInitializeSizes(rts);
      sfcnInitializeSampleTimes(rts);

      /* adjust sample time */
      ssSetSampleTime(rts, 0, 0.001);
      ssSetOffsetTime(rts, 0, 0.0);
      sfcnTsMap[0] = 1;

      /* set compiled values of dynamic vector attributes */
      ssSetNumNonsampledZCsAsInt(rts, 0);

      /* Update connectivity flags for each port */
      _ssSetInputPortConnected(rts, 0, 1);
      _ssSetInputPortConnected(rts, 1, 1);
      _ssSetInputPortConnected(rts, 2, 1);
      _ssSetInputPortConnected(rts, 3, 1);

      /* Update the BufferDstPort flags for each input port */
      ssSetInputPortBufferDstPort(rts, 0, -1);
      ssSetInputPortBufferDstPort(rts, 1, -1);
      ssSetInputPortBufferDstPort(rts, 2, -1);
      ssSetInputPortBufferDstPort(rts, 3, -1);
    }

    /* Level2 S-Function Block: Supine_SpeedGoat_v3/<S1>/Digital input  (sg_IO191_di_s) */
    {
      SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[3];

      /* timing info */
      time_T *sfcnPeriod =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.sfcnPeriod;
      time_T *sfcnOffset =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.sfcnOffset;
      int_T *sfcnTsMap = Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.sfcnTsMap;
      (void) std::memset(static_cast<void*>(sfcnPeriod), 0,
                         sizeof(time_T)*1);
      (void) std::memset(static_cast<void*>(sfcnOffset), 0,
                         sizeof(time_T)*1);
      ssSetSampleTimePtr(rts, &sfcnPeriod[0]);
      ssSetOffsetTimePtr(rts, &sfcnOffset[0]);
      ssSetSampleTimeTaskIDPtr(rts, sfcnTsMap);

      {
        ssSetBlkInfo2Ptr(rts, &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfo2[3]);
        ssSetBlkInfoSLSizePtr(rts,
                              &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfoSLSize
                              [3]);
      }

      _ssSetBlkInfo2PortInfo2Ptr(rts,
        &Supine_SpeedGoat_v3_M->NonInlinedSFcns.inputOutputPortInfo2[3]);

      /* Set up the mdlInfo pointer */
      ssSetRTWSfcnInfo(rts, Supine_SpeedGoat_v3_M->sfcnInfo);

      /* Allocate memory of model methods 2 */
      {
        ssSetModelMethods2(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods2[3]);
      }

      /* Allocate memory of model methods 3 */
      {
        ssSetModelMethods3(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods3[3]);
      }

      /* Allocate memory of model methods 4 */
      {
        ssSetModelMethods4(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods4[3]);
      }

      /* Allocate memory for states auxilliary information */
      {
        ssSetStatesInfo2(rts,
                         &Supine_SpeedGoat_v3_M->NonInlinedSFcns.statesInfo2[3]);
        ssSetPeriodicStatesInfo(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.periodicStatesInfo[3]);
      }

      /* outputs */
      {
        ssSetPortInfoForOutputs(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.outputPortInfo[0]);
        rts->blkInfo.blkInfo2->blkInfoSLSize->outputs =
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.outputPortInfoSLSize[0];
        _ssSetNumOutputPorts(rts, 2);
        _ssSetPortInfo2ForOutputUnits(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.outputPortUnits[0]);
        ssSetOutputPortUnit(rts, 0, 0);
        ssSetOutputPortUnit(rts, 1, 0);
        _ssSetPortInfo2ForOutputCoSimAttribute(rts,
          &Supine_SpeedGoat_v3_M->
          NonInlinedSFcns.Sfcn3.outputPortCoSimAttribute[0]);
        ssSetOutputPortIsContinuousQuantity(rts, 0, 0);
        ssSetOutputPortIsContinuousQuantity(rts, 1, 0);

        /* port 0 */
        {
          _ssSetOutputPortNumDimensions(rts, 0, 1);
          ssSetOutputPortWidthAsInt(rts, 0, 1);
          ssSetOutputPortSignal(rts, 0, ((real_T *)
            &Supine_SpeedGoat_v3_B.FF_stat));
        }

        /* port 1 */
        {
          _ssSetOutputPortNumDimensions(rts, 1, 1);
          ssSetOutputPortWidthAsInt(rts, 1, 1);
          ssSetOutputPortSignal(rts, 1, ((real_T *)
            &Supine_SpeedGoat_v3_B.Digitalinput_o2));
        }
      }

      /* path info */
      ssSetModelName(rts, "Digital input ");
      ssSetPath(rts, "Supine_SpeedGoat_v3/Digital Outputs/Digital input ");
      ssSetRTModel(rts,Supine_SpeedGoat_v3_M);
      ssSetParentSS(rts, (NULL));
      ssSetRootSS(rts, rts);
      ssSetVersion(rts, SIMSTRUCT_VERSION_LEVEL2);

      /* parameters */
      {
        mxArray **sfcnParams = (mxArray **)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.params;
        ssSetSFcnParamsCount(rts, 4);
        ssSetSFcnParamsPtr(rts, &sfcnParams[0]);
        ssSetSFcnParam(rts, 0, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitalinput_P1_Size);
        ssSetSFcnParam(rts, 1, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitalinput_P2_Size);
        ssSetSFcnParam(rts, 2, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitalinput_P3_Size);
        ssSetSFcnParam(rts, 3, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitalinput_P4_Size);
      }

      /* work vectors */
      ssSetPWork(rts, (void **) &Supine_SpeedGoat_v3_DW.Digitalinput_PWORK);

      {
        struct _ssDWorkRecord *dWorkRecord = (struct _ssDWorkRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.dWork;
        struct _ssDWorkAuxRecord *dWorkAuxRecord = (struct _ssDWorkAuxRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn3.dWorkAux;
        ssSetSFcnDWork(rts, dWorkRecord);
        ssSetSFcnDWorkAux(rts, dWorkAuxRecord);
        _ssSetNumDWork(rts, 1);

        /* PWORK */
        ssSetDWorkWidthAsInt(rts, 0, 1);
        ssSetDWorkDataType(rts, 0,SS_POINTER);
        ssSetDWorkComplexSignal(rts, 0, 0);
        ssSetDWork(rts, 0, &Supine_SpeedGoat_v3_DW.Digitalinput_PWORK);
      }

      /* registration */
      sg_IO191_di_s(rts);
      sfcnInitializeSizes(rts);
      sfcnInitializeSampleTimes(rts);

      /* adjust sample time */
      ssSetSampleTime(rts, 0, 0.001);
      ssSetOffsetTime(rts, 0, 0.0);
      sfcnTsMap[0] = 1;

      /* set compiled values of dynamic vector attributes */
      ssSetNumNonsampledZCsAsInt(rts, 0);

      /* Update connectivity flags for each port */
      _ssSetOutputPortConnected(rts, 0, 1);
      _ssSetOutputPortConnected(rts, 1, 0);
      _ssSetOutputPortBeingMerged(rts, 0, 0);
      _ssSetOutputPortBeingMerged(rts, 1, 0);

      /* Update the BufferDstPort flags for each input port */
    }

    /* Level2 S-Function Block: Supine_SpeedGoat_v3/<S4>/Digital output  (sg_IO191_do_s) */
    {
      SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[4];

      /* timing info */
      time_T *sfcnPeriod =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.sfcnPeriod;
      time_T *sfcnOffset =
        Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.sfcnOffset;
      int_T *sfcnTsMap = Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.sfcnTsMap;
      (void) std::memset(static_cast<void*>(sfcnPeriod), 0,
                         sizeof(time_T)*1);
      (void) std::memset(static_cast<void*>(sfcnOffset), 0,
                         sizeof(time_T)*1);
      ssSetSampleTimePtr(rts, &sfcnPeriod[0]);
      ssSetOffsetTimePtr(rts, &sfcnOffset[0]);
      ssSetSampleTimeTaskIDPtr(rts, sfcnTsMap);

      {
        ssSetBlkInfo2Ptr(rts, &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfo2[4]);
        ssSetBlkInfoSLSizePtr(rts,
                              &Supine_SpeedGoat_v3_M->NonInlinedSFcns.blkInfoSLSize
                              [4]);
      }

      _ssSetBlkInfo2PortInfo2Ptr(rts,
        &Supine_SpeedGoat_v3_M->NonInlinedSFcns.inputOutputPortInfo2[4]);

      /* Set up the mdlInfo pointer */
      ssSetRTWSfcnInfo(rts, Supine_SpeedGoat_v3_M->sfcnInfo);

      /* Allocate memory of model methods 2 */
      {
        ssSetModelMethods2(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods2[4]);
      }

      /* Allocate memory of model methods 3 */
      {
        ssSetModelMethods3(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods3[4]);
      }

      /* Allocate memory of model methods 4 */
      {
        ssSetModelMethods4(rts, &Supine_SpeedGoat_v3_M->
                           NonInlinedSFcns.methods4[4]);
      }

      /* Allocate memory for states auxilliary information */
      {
        ssSetStatesInfo2(rts,
                         &Supine_SpeedGoat_v3_M->NonInlinedSFcns.statesInfo2[4]);
        ssSetPeriodicStatesInfo(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.periodicStatesInfo[4]);
      }

      /* inputs */
      {
        _ssSetNumInputPorts(rts, 14);
        ssSetPortInfoForInputs(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.inputPortInfo[0]);
        rts->blkInfo.blkInfo2->blkInfoSLSize->inputs =
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.inputPortInfoSLSize[0];
        _ssSetPortInfo2ForInputUnits(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.inputPortUnits[0]);
        ssSetInputPortUnit(rts, 0, 0);
        ssSetInputPortUnit(rts, 1, 0);
        ssSetInputPortUnit(rts, 2, 0);
        ssSetInputPortUnit(rts, 3, 0);
        ssSetInputPortUnit(rts, 4, 0);
        ssSetInputPortUnit(rts, 5, 0);
        ssSetInputPortUnit(rts, 6, 0);
        ssSetInputPortUnit(rts, 7, 0);
        ssSetInputPortUnit(rts, 8, 0);
        ssSetInputPortUnit(rts, 9, 0);
        ssSetInputPortUnit(rts, 10, 0);
        ssSetInputPortUnit(rts, 11, 0);
        ssSetInputPortUnit(rts, 12, 0);
        ssSetInputPortUnit(rts, 13, 0);
        _ssSetPortInfo2ForInputCoSimAttribute(rts,
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.inputPortCoSimAttribute
          [0]);
        ssSetInputPortIsContinuousQuantity(rts, 0, 0);
        ssSetInputPortIsContinuousQuantity(rts, 1, 0);
        ssSetInputPortIsContinuousQuantity(rts, 2, 0);
        ssSetInputPortIsContinuousQuantity(rts, 3, 0);
        ssSetInputPortIsContinuousQuantity(rts, 4, 0);
        ssSetInputPortIsContinuousQuantity(rts, 5, 0);
        ssSetInputPortIsContinuousQuantity(rts, 6, 0);
        ssSetInputPortIsContinuousQuantity(rts, 7, 0);
        ssSetInputPortIsContinuousQuantity(rts, 8, 0);
        ssSetInputPortIsContinuousQuantity(rts, 9, 0);
        ssSetInputPortIsContinuousQuantity(rts, 10, 0);
        ssSetInputPortIsContinuousQuantity(rts, 11, 0);
        ssSetInputPortIsContinuousQuantity(rts, 12, 0);
        ssSetInputPortIsContinuousQuantity(rts, 13, 0);

        /* port 0 */
        {
          ssSetInputPortRequiredContiguous(rts, 0, 1);
          ssSetInputPortSignal(rts, 0, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 0, 1);
          ssSetInputPortWidthAsInt(rts, 0, 1);
        }

        /* port 1 */
        {
          ssSetInputPortRequiredContiguous(rts, 1, 1);
          ssSetInputPortSignal(rts, 1, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 1, 1);
          ssSetInputPortWidthAsInt(rts, 1, 1);
        }

        /* port 2 */
        {
          ssSetInputPortRequiredContiguous(rts, 2, 1);
          ssSetInputPortSignal(rts, 2, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 2, 1);
          ssSetInputPortWidthAsInt(rts, 2, 1);
        }

        /* port 3 */
        {
          ssSetInputPortRequiredContiguous(rts, 3, 1);
          ssSetInputPortSignal(rts, 3, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 3, 1);
          ssSetInputPortWidthAsInt(rts, 3, 1);
        }

        /* port 4 */
        {
          ssSetInputPortRequiredContiguous(rts, 4, 1);
          ssSetInputPortSignal(rts, 4, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 4, 1);
          ssSetInputPortWidthAsInt(rts, 4, 1);
        }

        /* port 5 */
        {
          ssSetInputPortRequiredContiguous(rts, 5, 1);
          ssSetInputPortSignal(rts, 5, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 5, 1);
          ssSetInputPortWidthAsInt(rts, 5, 1);
        }

        /* port 6 */
        {
          ssSetInputPortRequiredContiguous(rts, 6, 1);
          ssSetInputPortSignal(rts, 6, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 6, 1);
          ssSetInputPortWidthAsInt(rts, 6, 1);
        }

        /* port 7 */
        {
          ssSetInputPortRequiredContiguous(rts, 7, 1);
          ssSetInputPortSignal(rts, 7, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 7, 1);
          ssSetInputPortWidthAsInt(rts, 7, 1);
        }

        /* port 8 */
        {
          ssSetInputPortRequiredContiguous(rts, 8, 1);
          ssSetInputPortSignal(rts, 8, &Supine_SpeedGoat_v3_B.Switch_g);
          _ssSetInputPortNumDimensions(rts, 8, 1);
          ssSetInputPortWidthAsInt(rts, 8, 1);
        }

        /* port 9 */
        {
          ssSetInputPortRequiredContiguous(rts, 9, 1);
          ssSetInputPortSignal(rts, 9, &Supine_SpeedGoat_v3_B.Heart_Beat);
          _ssSetInputPortNumDimensions(rts, 9, 1);
          ssSetInputPortWidthAsInt(rts, 9, 1);
        }

        /* port 10 */
        {
          ssSetInputPortRequiredContiguous(rts, 10, 1);
          ssSetInputPortSignal(rts, 10, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 10, 1);
          ssSetInputPortWidthAsInt(rts, 10, 1);
        }

        /* port 11 */
        {
          ssSetInputPortRequiredContiguous(rts, 11, 1);
          ssSetInputPortSignal(rts, 11, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 11, 1);
          ssSetInputPortWidthAsInt(rts, 11, 1);
        }

        /* port 12 */
        {
          ssSetInputPortRequiredContiguous(rts, 12, 1);
          ssSetInputPortSignal(rts, 12, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 12, 1);
          ssSetInputPortWidthAsInt(rts, 12, 1);
        }

        /* port 13 */
        {
          ssSetInputPortRequiredContiguous(rts, 13, 1);
          ssSetInputPortSignal(rts, 13, (const_cast<real_T*>
            (&Supine_SpeedGoat_v3_RGND)));
          _ssSetInputPortNumDimensions(rts, 13, 1);
          ssSetInputPortWidthAsInt(rts, 13, 1);
        }
      }

      /* path info */
      ssSetModelName(rts, "Digital output ");
      ssSetPath(rts,
                "Supine_SpeedGoat_v3/Digital Outputs/Trigger Data Acquisition/Digital output ");
      ssSetRTModel(rts,Supine_SpeedGoat_v3_M);
      ssSetParentSS(rts, (NULL));
      ssSetRootSS(rts, rts);
      ssSetVersion(rts, SIMSTRUCT_VERSION_LEVEL2);

      /* parameters */
      {
        mxArray **sfcnParams = (mxArray **)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.params;
        ssSetSFcnParamsCount(rts, 6);
        ssSetSFcnParamsPtr(rts, &sfcnParams[0]);
        ssSetSFcnParam(rts, 0, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitaloutput_P1_Size);
        ssSetSFcnParam(rts, 1, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitaloutput_P2_Size);
        ssSetSFcnParam(rts, 2, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitaloutput_P3_Size);
        ssSetSFcnParam(rts, 3, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitaloutput_P4_Size);
        ssSetSFcnParam(rts, 4, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitaloutput_P5_Size);
        ssSetSFcnParam(rts, 5, (mxArray*)
                       Supine_SpeedGoat_v3_cal->Digitaloutput_P6_Size);
      }

      /* work vectors */
      ssSetPWork(rts, (void **) &Supine_SpeedGoat_v3_DW.Digitaloutput_PWORK);

      {
        struct _ssDWorkRecord *dWorkRecord = (struct _ssDWorkRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.dWork;
        struct _ssDWorkAuxRecord *dWorkAuxRecord = (struct _ssDWorkAuxRecord *)
          &Supine_SpeedGoat_v3_M->NonInlinedSFcns.Sfcn4.dWorkAux;
        ssSetSFcnDWork(rts, dWorkRecord);
        ssSetSFcnDWorkAux(rts, dWorkAuxRecord);
        _ssSetNumDWork(rts, 1);

        /* PWORK */
        ssSetDWorkWidthAsInt(rts, 0, 1);
        ssSetDWorkDataType(rts, 0,SS_POINTER);
        ssSetDWorkComplexSignal(rts, 0, 0);
        ssSetDWork(rts, 0, &Supine_SpeedGoat_v3_DW.Digitaloutput_PWORK);
      }

      /* registration */
      sg_IO191_do_s(rts);
      sfcnInitializeSizes(rts);
      sfcnInitializeSampleTimes(rts);

      /* adjust sample time */
      ssSetSampleTime(rts, 0, 0.001);
      ssSetOffsetTime(rts, 0, 0.0);
      sfcnTsMap[0] = 1;

      /* set compiled values of dynamic vector attributes */
      ssSetNumNonsampledZCsAsInt(rts, 0);

      /* Update connectivity flags for each port */
      _ssSetInputPortConnected(rts, 0, 0);
      _ssSetInputPortConnected(rts, 1, 0);
      _ssSetInputPortConnected(rts, 2, 0);
      _ssSetInputPortConnected(rts, 3, 0);
      _ssSetInputPortConnected(rts, 4, 0);
      _ssSetInputPortConnected(rts, 5, 0);
      _ssSetInputPortConnected(rts, 6, 0);
      _ssSetInputPortConnected(rts, 7, 0);
      _ssSetInputPortConnected(rts, 8, 1);
      _ssSetInputPortConnected(rts, 9, 1);
      _ssSetInputPortConnected(rts, 10, 0);
      _ssSetInputPortConnected(rts, 11, 0);
      _ssSetInputPortConnected(rts, 12, 0);
      _ssSetInputPortConnected(rts, 13, 0);

      /* Update the BufferDstPort flags for each input port */
      ssSetInputPortBufferDstPort(rts, 0, -1);
      ssSetInputPortBufferDstPort(rts, 1, -1);
      ssSetInputPortBufferDstPort(rts, 2, -1);
      ssSetInputPortBufferDstPort(rts, 3, -1);
      ssSetInputPortBufferDstPort(rts, 4, -1);
      ssSetInputPortBufferDstPort(rts, 5, -1);
      ssSetInputPortBufferDstPort(rts, 6, -1);
      ssSetInputPortBufferDstPort(rts, 7, -1);
      ssSetInputPortBufferDstPort(rts, 8, -1);
      ssSetInputPortBufferDstPort(rts, 9, -1);
      ssSetInputPortBufferDstPort(rts, 10, -1);
      ssSetInputPortBufferDstPort(rts, 11, -1);
      ssSetInputPortBufferDstPort(rts, 12, -1);
      ssSetInputPortBufferDstPort(rts, 13, -1);
    }
  }

  /* Start for S-Function (sg_IO191_setup_s): '<Root>/Setup ' */
  /* Level2 S-Function Block: '<Root>/Setup ' (sg_IO191_setup_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[0];
    sfcnStart(rts);
    if (ssGetErrorStatus(rts) != (NULL))
      return;
  }

  /* Start for S-Function (sg_IO191_ad_s): '<Root>/Analog input ' */
  /* Level2 S-Function Block: '<Root>/Analog input ' (sg_IO191_ad_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[1];
    sfcnStart(rts);
    if (ssGetErrorStatus(rts) != (NULL))
      return;
  }

  /* Start for S-Function (sg_IO191_da_s): '<Root>/Analog output ' */
  /* Level2 S-Function Block: '<Root>/Analog output ' (sg_IO191_da_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[2];
    sfcnStart(rts);
    if (ssGetErrorStatus(rts) != (NULL))
      return;
  }

  /* Start for Lookup_n-D: '<S3>/MVC_LUT' incorporates:
   *  Constant: '<S3>/Constant'
   *  Constant: '<S3>/MVC_TQ'
   *  Constant: '<S3>/isometric_ankle_angles'
   */
  {
    rt_LUTnWork *TWork_start = (rt_LUTnWork *) &Supine_SpeedGoat_v3_DW.TWork[0];
    void **bpDataSet = static_cast<void **>(&Supine_SpeedGoat_v3_DW.m_bpDataSet);
    TWork_start->m_dimSizes = static_cast<const uint32_T *>
      (&Supine_SpeedGoat_v3_cal->MVC_LUT_dimSizes);
    TWork_start->m_tableData = (void *) &Supine_SpeedGoat_v3_cal->MVC_TQ_Value[0];
    TWork_start->m_bpDataSet = bpDataSet;
    TWork_start->m_bpIndex = &Supine_SpeedGoat_v3_DW.m_bpIndex;
    TWork_start->m_bpLambda = &Supine_SpeedGoat_v3_DW.m_bpLambda;
    TWork_start->m_maxIndex = static_cast<const uint32_T *>
      (&Supine_SpeedGoat_v3_cal->MVC_LUT_maxIndex);
    bpDataSet[0] = (void *)
      &Supine_SpeedGoat_v3_cal->isometric_ankle_angles_Value[0];
  }

  {
    const real_T **bpDataSet;
    const real_T *xp, *yp;
    real_T *dp;
    uint32_T len;
    const rt_LUTnWork *TWork_interp;
    rt_LUTSplineWork *rt_SplWk = (rt_LUTSplineWork*)
      &Supine_SpeedGoat_v3_DW.SWork[0];
    rt_SplWk->m_TWork = (rt_LUTnWork*)&Supine_SpeedGoat_v3_DW.TWork[0];
    rt_SplWk->m_yyA = &Supine_SpeedGoat_v3_DW.m_yyA;
    rt_SplWk->m_yyB = &Supine_SpeedGoat_v3_DW.m_yyB;
    rt_SplWk->m_yy2 = &Supine_SpeedGoat_v3_DW.m_yy2;
    rt_SplWk->m_up = &Supine_SpeedGoat_v3_DW.m_up[0];
    rt_SplWk->m_y2 = &Supine_SpeedGoat_v3_DW.m_y2[0];
    rt_SplWk->m_numYWorkElts = Supine_SpeedGoat_v3_cal->MVC_LUT_numYWorkElts;
    rt_SplWk->m_reCalc = &Supine_SpeedGoat_v3_DW.reCalcSecDerivFirstDimCoeffs;
    rt_SplWk->m_preBp0AndTable = &Supine_SpeedGoat_v3_DW.prevBp0AndTableData[0];
    *rt_SplWk->m_reCalc = 1;

    /* cache table data and first breakpoint data */
    TWork_interp = static_cast<const rt_LUTnWork *>(rt_SplWk->m_TWork);
    bpDataSet = (const real_T **) TWork_interp->m_bpDataSet;
    xp = bpDataSet[0U];
    len = TWork_interp->m_maxIndex[0U] + 1U;
    dp = (real_T *) rt_SplWk->m_preBp0AndTable;
    yp = (real_T *) TWork_interp->m_tableData;
    (void) std::memcpy(dp, xp,
                       len * sizeof(real_T));
    dp = &(dp[len]);

    /* save the table data */
    (void) std::memcpy(dp, yp,
                       len * rt_SplWk->m_numYWorkElts[0U] * sizeof(real_T));
  }

  /* Start for Lookup_n-D: '<S3>/Passive_LUT' incorporates:
   *  Constant: '<S3>/Constant'
   *  Constant: '<S3>/Passive_TQ'
   *  Constant: '<S3>/isometric_ankle_angles'
   */
  {
    rt_LUTnWork *TWork_start = (rt_LUTnWork *) &Supine_SpeedGoat_v3_DW.TWork_e[0];
    void **bpDataSet = static_cast<void **>
      (&Supine_SpeedGoat_v3_DW.m_bpDataSet_e);
    TWork_start->m_dimSizes = static_cast<const uint32_T *>
      (&Supine_SpeedGoat_v3_cal->Passive_LUT_dimSizes);
    TWork_start->m_tableData = (void *)
      &Supine_SpeedGoat_v3_cal->Passive_TQ_Value[0];
    TWork_start->m_bpDataSet = bpDataSet;
    TWork_start->m_bpIndex = &Supine_SpeedGoat_v3_DW.m_bpIndex_d;
    TWork_start->m_bpLambda = &Supine_SpeedGoat_v3_DW.m_bpLambda_p;
    TWork_start->m_maxIndex = static_cast<const uint32_T *>
      (&Supine_SpeedGoat_v3_cal->Passive_LUT_maxIndex);
    bpDataSet[0] = (void *)
      &Supine_SpeedGoat_v3_cal->isometric_ankle_angles_Value[0];
  }

  {
    const real_T **bpDataSet;
    const real_T *xp, *yp;
    real_T *dp;
    uint32_T len;
    const rt_LUTnWork *TWork_interp;
    rt_LUTSplineWork *rt_SplWk = (rt_LUTSplineWork*)
      &Supine_SpeedGoat_v3_DW.SWork_e[0];
    rt_SplWk->m_TWork = (rt_LUTnWork*)&Supine_SpeedGoat_v3_DW.TWork_e[0];
    rt_SplWk->m_yyA = &Supine_SpeedGoat_v3_DW.m_yyA_h;
    rt_SplWk->m_yyB = &Supine_SpeedGoat_v3_DW.m_yyB_b;
    rt_SplWk->m_yy2 = &Supine_SpeedGoat_v3_DW.m_yy2_j;
    rt_SplWk->m_up = &Supine_SpeedGoat_v3_DW.m_up_k[0];
    rt_SplWk->m_y2 = &Supine_SpeedGoat_v3_DW.m_y2_k[0];
    rt_SplWk->m_numYWorkElts = Supine_SpeedGoat_v3_cal->Passive_LUT_numYWorkElts;
    rt_SplWk->m_reCalc = &Supine_SpeedGoat_v3_DW.reCalcSecDerivFirstDimCoeffs_b;
    rt_SplWk->m_preBp0AndTable = &Supine_SpeedGoat_v3_DW.prevBp0AndTableData_i[0];
    *rt_SplWk->m_reCalc = 1;

    /* cache table data and first breakpoint data */
    TWork_interp = static_cast<const rt_LUTnWork *>(rt_SplWk->m_TWork);
    bpDataSet = (const real_T **) TWork_interp->m_bpDataSet;
    xp = bpDataSet[0U];
    len = TWork_interp->m_maxIndex[0U] + 1U;
    dp = (real_T *) rt_SplWk->m_preBp0AndTable;
    yp = (real_T *) TWork_interp->m_tableData;
    (void) std::memcpy(dp, xp,
                       len * sizeof(real_T));
    dp = &(dp[len]);

    /* save the table data */
    (void) std::memcpy(dp, yp,
                       len * rt_SplWk->m_numYWorkElts[0U] * sizeof(real_T));
  }

  /* Start for S-Function (sg_IO191_di_s): '<S1>/Digital input ' */
  /* Level2 S-Function Block: '<S1>/Digital input ' (sg_IO191_di_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[3];
    sfcnStart(rts);
    if (ssGetErrorStatus(rts) != (NULL))
      return;
  }

  /* Start for DiscretePulseGenerator: '<S4>/Hearbeat' */
  Supine_SpeedGoat_v3_DW.clockTickCounter = 0;

  /* Start for S-Function (sg_IO191_do_s): '<S4>/Digital output ' */
  /* Level2 S-Function Block: '<S4>/Digital output ' (sg_IO191_do_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[4];
    sfcnStart(rts);
    if (ssGetErrorStatus(rts) != (NULL))
      return;
  }

  /* Start for S-Function (slrealtimeUDPSend): '<S7>/UDP Send1' incorporates:
   *  Constant: '<S7>/position_switch4'
   */
  {
    try {
      slrealtime::ip::udp::Socket *udpSock = getUDPSocket("10.68.3.237",5005U);
      uint8_t *remoteAddress = (uint8_t *)
        Supine_SpeedGoat_v3_cal->UDPSend1_toAddress;
      uint16_t *remotePort = (uint16_t *)
        &Supine_SpeedGoat_v3_cal->UDPSend1_toPort;
      udpSock->setRemoteEndpoint(remoteAddress, remotePort[0]);
      Supine_SpeedGoat_v3_DW.UDPSend1_IWORK[0] = 16;
      Supine_SpeedGoat_v3_DW.UDPSend1_IWORK[1] = 5005U;
      Supine_SpeedGoat_v3_DW.UDPSend1_PWORK = reinterpret_cast<void*>(udpSock);
    } catch (std::exception& e) {
      std::string tmp = std::string(e.what());
      static std::string eMsg = tmp;
      rtmSetErrorStatus(Supine_SpeedGoat_v3_M, eMsg.c_str());
      rtmSetStopRequested(Supine_SpeedGoat_v3_M, 1);
      ;
    }
  }

  Supine_SpeedGoat_v3_PrevZCX.SampleandHold_Trig_ZCE = ZERO_ZCSIG;

  {
    real_T tmin;
    int32_T i;
    int32_T t;
    uint32_T tseed;

    /* InitializeConditions for StateSpace: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
    for (i = 0; i < 8; i++) {
      Supine_SpeedGoat_v3_X.uthorderBeselwithcutoff07HzbyGo[i] =
        Supine_SpeedGoat_v3_cal->uthorderBeselwithcutoff07Hzby_g;
    }

    /* End of InitializeConditions for StateSpace: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */

    /* InitializeConditions for S-Function (sdspnsamp2): '<S2>/N-Sample Enable1' */

    /* DSP System Toolbox N-Sample Enable  (sdspnsamp2) - '<S2>/N-Sample Enable1' */
    Supine_SpeedGoat_v3_DW.NSampleEnable1_Counter = (uint32_T) 0;

    /* InitializeConditions for StateSpace: '<S2>/Cut-off 1 Hz1' */
    Supine_SpeedGoat_v3_X.Cutoff1Hz1_CSTATE[0] =
      Supine_SpeedGoat_v3_cal->Cutoff1Hz1_InitialCondition;
    Supine_SpeedGoat_v3_X.Cutoff1Hz1_CSTATE[1] =
      Supine_SpeedGoat_v3_cal->Cutoff1Hz1_InitialCondition;

    /* InitializeConditions for Derivative: '<S2>/Derivative' */
    Supine_SpeedGoat_v3_DW.TimeStampA = (rtInf);
    Supine_SpeedGoat_v3_DW.TimeStampB = (rtInf);

    /* InitializeConditions for Integrator: '<S2>/Integrator' */
    Supine_SpeedGoat_v3_X.Integrator_CSTATE =
      Supine_SpeedGoat_v3_cal->Integrator_IC;

    /* InitializeConditions for Derivative: '<S2>/Derivative1' */
    Supine_SpeedGoat_v3_DW.TimeStampA_d = (rtInf);
    Supine_SpeedGoat_v3_DW.TimeStampB_n = (rtInf);

    /* InitializeConditions for UniformRandomNumber: '<S2>/Uniform Random Number 0.1 ms' */
    tmin = std::floor(Supine_SpeedGoat_v3_cal->UniformRandomNumber01ms_Seed);
    if (rtIsNaN(tmin) || rtIsInf(tmin)) {
      tmin = 0.0;
    } else {
      tmin = std::fmod(tmin, 4.294967296E+9);
    }

    tseed = tmin < 0.0 ? static_cast<uint32_T>(-static_cast<int32_T>(
      static_cast<uint32_T>(-tmin))) : static_cast<uint32_T>(tmin);
    i = static_cast<int32_T>(tseed >> 16U);
    t = static_cast<int32_T>(tseed & 32768U);
    tseed = ((((tseed - (static_cast<uint32_T>(i) << 16U)) + t) << 16U) + t) + i;
    if (tseed < 1U) {
      tseed = 1144108930U;
    } else if (tseed > 2147483646U) {
      tseed = 2147483646U;
    }

    Supine_SpeedGoat_v3_DW.RandSeed = tseed;
    tmin = Supine_SpeedGoat_v3_cal->UniformRandomNumber01ms_Minimum;
    Supine_SpeedGoat_v3_DW.UniformRandomNumber01ms_NextOut =
      (Supine_SpeedGoat_v3_cal->UniformRandomNumber01ms_Maximum - tmin) *
      rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed) + tmin;

    /* End of InitializeConditions for UniformRandomNumber: '<S2>/Uniform Random Number 0.1 ms' */

    /* InitializeConditions for StateSpace: '<S2>/Cut-off 35 Hz' */
    for (i = 0; i < 5; i++) {
      Supine_SpeedGoat_v3_X.Cutoff35Hz_CSTATE[i] =
        Supine_SpeedGoat_v3_cal->Cutoff35Hz_InitialCondition;
    }

    /* End of InitializeConditions for StateSpace: '<S2>/Cut-off 35 Hz' */

    /* InitializeConditions for RateTransition: '<S9>/Rate Transition1' */
    Supine_SpeedGoat_v3_DW.RateTransition1_Buf[0] =
      Supine_SpeedGoat_v3_cal->RateTransition1_InitialConditio;
    Supine_SpeedGoat_v3_DW.RateTransition1_WrBufIdx = 0;
    Supine_SpeedGoat_v3_DW.RateTransition1_RdBufIdx = 1;

    /* InitializeConditions for RateTransition: '<S9>/Rate Transition2' */
    Supine_SpeedGoat_v3_DW.RateTransition2_Buf[0] =
      Supine_SpeedGoat_v3_cal->RateTransition2_InitialConditio;
    Supine_SpeedGoat_v3_DW.RateTransition2_WrBufIdx = 0;
    Supine_SpeedGoat_v3_DW.RateTransition2_RdBufIdx = 1;

    /* InitializeConditions for RateTransition: '<S9>/Rate Transition3' */
    Supine_SpeedGoat_v3_DW.RateTransition3_Buf[0] =
      Supine_SpeedGoat_v3_cal->RateTransition3_InitialConditio;
    Supine_SpeedGoat_v3_DW.RateTransition3_WrBufIdx = 0;
    Supine_SpeedGoat_v3_DW.RateTransition3_RdBufIdx = 1;

    /* InitializeConditions for RateTransition: '<S9>/Rate Transition4' */
    Supine_SpeedGoat_v3_DW.RateTransition4_Buf[0] =
      Supine_SpeedGoat_v3_cal->RateTransition4_InitialConditio;
    Supine_SpeedGoat_v3_DW.RateTransition4_WrBufIdx = 0;
    Supine_SpeedGoat_v3_DW.RateTransition4_RdBufIdx = 1;

    /* InitializeConditions for RateTransition: '<S9>/Rate Transition5' */
    Supine_SpeedGoat_v3_DW.RateTransition5_Buf[0] =
      Supine_SpeedGoat_v3_cal->RateTransition5_InitialConditio;
    Supine_SpeedGoat_v3_DW.RateTransition5_WrBufIdx = 0;
    Supine_SpeedGoat_v3_DW.RateTransition5_RdBufIdx = 1;

    /* InitializeConditions for RateTransition: '<S7>/Rate Transition' */
    for (i = 0; i < 16; i++) {
      Supine_SpeedGoat_v3_DW.RateTransition_Buf[i] =
        Supine_SpeedGoat_v3_cal->RateTransition_InitialCondition;
    }

    Supine_SpeedGoat_v3_DW.RateTransition_WrBufIdx = 0;
    Supine_SpeedGoat_v3_DW.RateTransition_RdBufIdx = 1;

    /* End of InitializeConditions for RateTransition: '<S7>/Rate Transition' */

    /* InitializeConditions for Memory: '<S4>/Memory7' */
    Supine_SpeedGoat_v3_DW.Memory7_PreviousInput =
      Supine_SpeedGoat_v3_cal->Memory7_InitialCondition;

    /* InitializeConditions for S-Function (sdspnsamp2): '<S5>/N-Sample Enable' */

    /* DSP System Toolbox N-Sample Enable  (sdspnsamp2) - '<S5>/N-Sample Enable' */
    Supine_SpeedGoat_v3_DW.NSampleEnable_Counter = (uint32_T) 0;

    /* Initialize event port handler previous signal state: */
    Supine_SpeedGoat_v3_DW.NSampleEnable_EPHState = EVENT_PORT_STATE_UNINIT;

    /* InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 0.2s base switch time' */
    tmin = std::floor(Supine_SpeedGoat_v3_cal->ForPRBSwith02sbaseswitchtime_Se);
    if (rtIsNaN(tmin) || rtIsInf(tmin)) {
      tmin = 0.0;
    } else {
      tmin = std::fmod(tmin, 4.294967296E+9);
    }

    tseed = tmin < 0.0 ? static_cast<uint32_T>(-static_cast<int32_T>(
      static_cast<uint32_T>(-tmin))) : static_cast<uint32_T>(tmin);
    i = static_cast<int32_T>(tseed >> 16U);
    t = static_cast<int32_T>(tseed & 32768U);
    tseed = ((((tseed - (static_cast<uint32_T>(i) << 16U)) + t) << 16U) + t) + i;
    if (tseed < 1U) {
      tseed = 1144108930U;
    } else if (tseed > 2147483646U) {
      tseed = 2147483646U;
    }

    Supine_SpeedGoat_v3_DW.RandSeed_n = tseed;
    tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith02sbaseswitchtime_Mi;
    Supine_SpeedGoat_v3_DW.ForPRBSwith02sbaseswitchtime_Ne =
      (Supine_SpeedGoat_v3_cal->ForPRBSwith02sbaseswitchtime_Ma - tmin) *
      rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_n) + tmin;

    /* End of InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 0.2s base switch time' */

    /* InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 0.5s base switch time' */
    tmin = std::floor(Supine_SpeedGoat_v3_cal->ForPRBSwith05sbaseswitchtime_Se);
    if (rtIsNaN(tmin) || rtIsInf(tmin)) {
      tmin = 0.0;
    } else {
      tmin = std::fmod(tmin, 4.294967296E+9);
    }

    tseed = tmin < 0.0 ? static_cast<uint32_T>(-static_cast<int32_T>(
      static_cast<uint32_T>(-tmin))) : static_cast<uint32_T>(tmin);
    i = static_cast<int32_T>(tseed >> 16U);
    t = static_cast<int32_T>(tseed & 32768U);
    tseed = ((((tseed - (static_cast<uint32_T>(i) << 16U)) + t) << 16U) + t) + i;
    if (tseed < 1U) {
      tseed = 1144108930U;
    } else if (tseed > 2147483646U) {
      tseed = 2147483646U;
    }

    Supine_SpeedGoat_v3_DW.RandSeed_h = tseed;
    tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith05sbaseswitchtime_Mi;
    Supine_SpeedGoat_v3_DW.ForPRBSwith05sbaseswitchtime_Ne =
      (Supine_SpeedGoat_v3_cal->ForPRBSwith05sbaseswitchtime_Ma - tmin) *
      rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_h) + tmin;

    /* End of InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 0.5s base switch time' */

    /* InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 0.7s base switch time' */
    tmin = std::floor(Supine_SpeedGoat_v3_cal->ForPRBSwith07sbaseswitchtime_Se);
    if (rtIsNaN(tmin) || rtIsInf(tmin)) {
      tmin = 0.0;
    } else {
      tmin = std::fmod(tmin, 4.294967296E+9);
    }

    tseed = tmin < 0.0 ? static_cast<uint32_T>(-static_cast<int32_T>(
      static_cast<uint32_T>(-tmin))) : static_cast<uint32_T>(tmin);
    i = static_cast<int32_T>(tseed >> 16U);
    t = static_cast<int32_T>(tseed & 32768U);
    tseed = ((((tseed - (static_cast<uint32_T>(i) << 16U)) + t) << 16U) + t) + i;
    if (tseed < 1U) {
      tseed = 1144108930U;
    } else if (tseed > 2147483646U) {
      tseed = 2147483646U;
    }

    Supine_SpeedGoat_v3_DW.RandSeed_g = tseed;
    tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith07sbaseswitchtime_Mi;
    Supine_SpeedGoat_v3_DW.ForPRBSwith07sbaseswitchtime_Ne =
      (Supine_SpeedGoat_v3_cal->ForPRBSwith07sbaseswitchtime_Ma - tmin) *
      rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_g) + tmin;

    /* End of InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 0.7s base switch time' */

    /* InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 1s base switch time' */
    tmin = std::floor(Supine_SpeedGoat_v3_cal->ForPRBSwith1sbaseswitchtime_See);
    if (rtIsNaN(tmin) || rtIsInf(tmin)) {
      tmin = 0.0;
    } else {
      tmin = std::fmod(tmin, 4.294967296E+9);
    }

    tseed = tmin < 0.0 ? static_cast<uint32_T>(-static_cast<int32_T>(
      static_cast<uint32_T>(-tmin))) : static_cast<uint32_T>(tmin);
    i = static_cast<int32_T>(tseed >> 16U);
    t = static_cast<int32_T>(tseed & 32768U);
    tseed = ((((tseed - (static_cast<uint32_T>(i) << 16U)) + t) << 16U) + t) + i;
    if (tseed < 1U) {
      tseed = 1144108930U;
    } else if (tseed > 2147483646U) {
      tseed = 2147483646U;
    }

    Supine_SpeedGoat_v3_DW.RandSeed_j = tseed;
    tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith1sbaseswitchtime_Min;
    Supine_SpeedGoat_v3_DW.ForPRBSwith1sbaseswitchtime_Nex =
      (Supine_SpeedGoat_v3_cal->ForPRBSwith1sbaseswitchtime_Max - tmin) *
      rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_j) + tmin;

    /* End of InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 1s base switch time' */

    /* InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 2s base switch time' */
    tmin = std::floor(Supine_SpeedGoat_v3_cal->ForPRBSwith2sbaseswitchtime_See);
    if (rtIsNaN(tmin) || rtIsInf(tmin)) {
      tmin = 0.0;
    } else {
      tmin = std::fmod(tmin, 4.294967296E+9);
    }

    tseed = tmin < 0.0 ? static_cast<uint32_T>(-static_cast<int32_T>(
      static_cast<uint32_T>(-tmin))) : static_cast<uint32_T>(tmin);
    i = static_cast<int32_T>(tseed >> 16U);
    t = static_cast<int32_T>(tseed & 32768U);
    tseed = ((((tseed - (static_cast<uint32_T>(i) << 16U)) + t) << 16U) + t) + i;
    if (tseed < 1U) {
      tseed = 1144108930U;
    } else if (tseed > 2147483646U) {
      tseed = 2147483646U;
    }

    Supine_SpeedGoat_v3_DW.RandSeed_b = tseed;
    tmin = Supine_SpeedGoat_v3_cal->ForPRBSwith2sbaseswitchtime_Min;
    Supine_SpeedGoat_v3_DW.ForPRBSwith2sbaseswitchtime_Nex =
      (Supine_SpeedGoat_v3_cal->ForPRBSwith2sbaseswitchtime_Max - tmin) *
      rt_urand_Upu32_Yd_f_pw_snf(&Supine_SpeedGoat_v3_DW.RandSeed_b) + tmin;

    /* End of InitializeConditions for UniformRandomNumber: '<S9>/For PRBS with 2s base switch time' */

    /* SystemInitialize for Triggered SubSystem: '<S2>/Sample and Hold' */
    /* SystemInitialize for SignalConversion generated from: '<S6>/In' incorporates:
     *  Outport: '<S6>/ '
     */
    Supine_SpeedGoat_v3_B.In = Supine_SpeedGoat_v3_cal->_Y0;

    /* End of SystemInitialize for SubSystem: '<S2>/Sample and Hold' */
  }
}

/* Model terminate function */
void Supine_SpeedGoat_v3_terminate(void)
{
  /* Terminate for S-Function (sg_IO191_setup_s): '<Root>/Setup ' */
  /* Level2 S-Function Block: '<Root>/Setup ' (sg_IO191_setup_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[0];
    sfcnTerminate(rts);
  }

  /* Terminate for S-Function (sg_IO191_ad_s): '<Root>/Analog input ' */
  /* Level2 S-Function Block: '<Root>/Analog input ' (sg_IO191_ad_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[1];
    sfcnTerminate(rts);
  }

  /* Terminate for S-Function (sg_IO191_da_s): '<Root>/Analog output ' */
  /* Level2 S-Function Block: '<Root>/Analog output ' (sg_IO191_da_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[2];
    sfcnTerminate(rts);
  }

  /* Terminate for S-Function (sg_IO191_di_s): '<S1>/Digital input ' */
  /* Level2 S-Function Block: '<S1>/Digital input ' (sg_IO191_di_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[3];
    sfcnTerminate(rts);
  }

  /* Terminate for S-Function (sg_IO191_do_s): '<S4>/Digital output ' */
  /* Level2 S-Function Block: '<S4>/Digital output ' (sg_IO191_do_s) */
  {
    SimStruct *rts = Supine_SpeedGoat_v3_M->childSfunctions[4];
    sfcnTerminate(rts);
  }

  /* Terminate for S-Function (slrealtimeUDPSend): '<S7>/UDP Send1' incorporates:
   *  Constant: '<S7>/position_switch4'
   */
  {
    slrealtime::ip::udp::Socket *udpSock = getUDPSocket("10.68.3.237",5005U);
    if (udpSock)
      delete udpSock;
  }
}
