/*
 * Supine_SpeedGoat_v3_private.h
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

#ifndef RTW_HEADER_Supine_SpeedGoat_v3_private_h_
#define RTW_HEADER_Supine_SpeedGoat_v3_private_h_
#include "rtwtypes.h"
#include "multiword_types.h"
#include "zero_crossing_types.h"
#include "Supine_SpeedGoat_v3.h"

/* Private macros used by the generated code to access rtModel */
#ifndef rtmIsMajorTimeStep
#define rtmIsMajorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MAJOR_TIME_STEP)
#endif

#ifndef rtmIsMinorTimeStep
#define rtmIsMinorTimeStep(rtm)        (((rtm)->Timing.simTimeStep) == MINOR_TIME_STEP)
#endif

#ifndef rtmSetTFinal
#define rtmSetTFinal(rtm, val)         ((rtm)->Timing.tFinal = (val))
#endif

#ifndef rtmSetTPtr
#define rtmSetTPtr(rtm, val)           ((rtm)->Timing.t = (val))
#endif

#include "dsp_rt.h"  /* DSP System Toolbox general run time support functions */
#include "dspeph_rt.h"         /* DSP System Toolbox run time support library */

extern real_T rt_urand_Upu32_Yd_f_pw_snf(uint32_T *u);
extern void* slrtRegisterSignalToLoggingService(uintptr_t sigAddr);
real_T look_SplNBinXZcd(uint32_T numDims, const real_T* u, const
  rt_LUTSplineWork * const SWork);
void rt_Spline2Derivd(const real_T *x, const real_T *y, uint32_T n, real_T *u,
                      real_T *y2);
real_T intrp_NSplcd(uint32_T numDims, const rt_LUTSplineWork * const splWork,
                    uint32_T extrapMethod);
extern uint32_T plook_binx(real_T u, const real_T bp[], uint32_T maxIndex,
  real_T *fraction);
extern uint32_T binsearch_u32d(real_T u, const real_T bp[], uint32_T startIndex,
  uint32_T maxIndex);
extern "C" void sg_IO191_setup_s(SimStruct *rts);
extern "C" void sg_IO191_ad_s(SimStruct *rts);
extern "C" void sg_IO191_da_s(SimStruct *rts);
extern "C" void sg_IO191_di_s(SimStruct *rts);
extern "C" void sg_IO191_do_s(SimStruct *rts);

/* private model entry point functions */
extern void Supine_SpeedGoat_v3_derivatives(void);

#endif                           /* RTW_HEADER_Supine_SpeedGoat_v3_private_h_ */
