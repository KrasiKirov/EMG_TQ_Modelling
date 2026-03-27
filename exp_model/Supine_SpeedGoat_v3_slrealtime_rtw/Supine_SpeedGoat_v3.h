/*
 * Supine_SpeedGoat_v3.h
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

#ifndef RTW_HEADER_Supine_SpeedGoat_v3_h_
#define RTW_HEADER_Supine_SpeedGoat_v3_h_
#include <logsrv.h>
#include <cstring>
#include "rtwtypes.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "slrealtime/libsrc/IP/udp.hpp"
#include "Supine_SpeedGoat_v3_types.h"

extern "C" {

#include "rtGetInf.h"

}
  extern "C"
{

#include "rt_nonfinite.h"

}

#include <stddef.h>
#include "rtsplntypes.h"
#include "Supine_SpeedGoat_v3_cal.h"
#include "zero_crossing_types.h"

/* Macros for accessing real-time model data structure */
#ifndef rtmGetContStateDisabled
#define rtmGetContStateDisabled(rtm)   ((rtm)->contStateDisabled)
#endif

#ifndef rtmSetContStateDisabled
#define rtmSetContStateDisabled(rtm, val) ((rtm)->contStateDisabled = (val))
#endif

#ifndef rtmGetContStates
#define rtmGetContStates(rtm)          ((rtm)->contStates)
#endif

#ifndef rtmSetContStates
#define rtmSetContStates(rtm, val)     ((rtm)->contStates = (val))
#endif

#ifndef rtmGetContTimeOutputInconsistentWithStateAtMajorStepFlag
#define rtmGetContTimeOutputInconsistentWithStateAtMajorStepFlag(rtm) ((rtm)->CTOutputIncnstWithState)
#endif

#ifndef rtmSetContTimeOutputInconsistentWithStateAtMajorStepFlag
#define rtmSetContTimeOutputInconsistentWithStateAtMajorStepFlag(rtm, val) ((rtm)->CTOutputIncnstWithState = (val))
#endif

#ifndef rtmGetDerivCacheNeedsReset
#define rtmGetDerivCacheNeedsReset(rtm) ((rtm)->derivCacheNeedsReset)
#endif

#ifndef rtmSetDerivCacheNeedsReset
#define rtmSetDerivCacheNeedsReset(rtm, val) ((rtm)->derivCacheNeedsReset = (val))
#endif

#ifndef rtmGetFinalTime
#define rtmGetFinalTime(rtm)           ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetIntgData
#define rtmGetIntgData(rtm)            ((rtm)->intgData)
#endif

#ifndef rtmSetIntgData
#define rtmSetIntgData(rtm, val)       ((rtm)->intgData = (val))
#endif

#ifndef rtmGetOdeF
#define rtmGetOdeF(rtm)                ((rtm)->odeF)
#endif

#ifndef rtmSetOdeF
#define rtmSetOdeF(rtm, val)           ((rtm)->odeF = (val))
#endif

#ifndef rtmGetOdeY
#define rtmGetOdeY(rtm)                ((rtm)->odeY)
#endif

#ifndef rtmSetOdeY
#define rtmSetOdeY(rtm, val)           ((rtm)->odeY = (val))
#endif

#ifndef rtmGetPeriodicContStateIndices
#define rtmGetPeriodicContStateIndices(rtm) ((rtm)->periodicContStateIndices)
#endif

#ifndef rtmSetPeriodicContStateIndices
#define rtmSetPeriodicContStateIndices(rtm, val) ((rtm)->periodicContStateIndices = (val))
#endif

#ifndef rtmGetPeriodicContStateRanges
#define rtmGetPeriodicContStateRanges(rtm) ((rtm)->periodicContStateRanges)
#endif

#ifndef rtmSetPeriodicContStateRanges
#define rtmSetPeriodicContStateRanges(rtm, val) ((rtm)->periodicContStateRanges = (val))
#endif

#ifndef rtmGetSampleHitArray
#define rtmGetSampleHitArray(rtm)      ((rtm)->Timing.sampleHitArray)
#endif

#ifndef rtmGetStepSize
#define rtmGetStepSize(rtm)            ((rtm)->Timing.stepSize)
#endif

#ifndef rtmGetZCCacheNeedsReset
#define rtmGetZCCacheNeedsReset(rtm)   ((rtm)->zCCacheNeedsReset)
#endif

#ifndef rtmSetZCCacheNeedsReset
#define rtmSetZCCacheNeedsReset(rtm, val) ((rtm)->zCCacheNeedsReset = (val))
#endif

#ifndef rtmGet_TimeOfLastOutput
#define rtmGet_TimeOfLastOutput(rtm)   ((rtm)->Timing.timeOfLastOutput)
#endif

#ifndef rtmGetdX
#define rtmGetdX(rtm)                  ((rtm)->derivs)
#endif

#ifndef rtmSetdX
#define rtmSetdX(rtm, val)             ((rtm)->derivs = (val))
#endif

#ifndef rtmCounterLimit
#define rtmCounterLimit(rtm, idx)      ((rtm)->Timing.TaskCounters.cLimit[(idx)])
#endif

#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmStepTask
#define rtmStepTask(rtm, idx)          ((rtm)->Timing.TaskCounters.TID[(idx)] == 0)
#endif

#ifndef rtmGetStopRequested
#define rtmGetStopRequested(rtm)       ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
#define rtmSetStopRequested(rtm, val)  ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
#define rtmGetStopRequestedPtr(rtm)    (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
#define rtmGetT(rtm)                   (rtmGetTPtr((rtm))[0])
#endif

#ifndef rtmGetTFinal
#define rtmGetTFinal(rtm)              ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                ((rtm)->Timing.t)
#endif

#ifndef rtmGetTStart
#define rtmGetTStart(rtm)              ((rtm)->Timing.tStart)
#endif

#ifndef rtmTaskCounter
#define rtmTaskCounter(rtm, idx)       ((rtm)->Timing.TaskCounters.TID[(idx)])
#endif

#ifndef rtmGetTimeOfLastOutput
#define rtmGetTimeOfLastOutput(rtm)    ((rtm)->Timing.timeOfLastOutput)
#endif

/* Block signals (default storage) */
struct B_Supine_SpeedGoat_v3_T {
  real_T uthorderBeselwithcutoff07HzbyGo;
  /* '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
  real_T Torque;                       /* '<Root>/Analog input ' */
  real_T Analoginput_o2;               /* '<Root>/Analog input ' */
  real_T Analoginput_o3;               /* '<Root>/Analog input ' */
  real_T Analoginput_o4;               /* '<Root>/Analog input ' */
  real_T Possition;                    /* '<Root>/Analog input ' */
  real_T Torque_c;                     /* '<Root>/Analog input ' */
  real_T Analoginput_o7;               /* '<Root>/Analog input ' */
  real_T Analoginput_o8;               /* '<Root>/Analog input ' */
  real_T Gain1;                        /* '<Root>/Gain1' */
  real_T Sum2;                         /* '<S2>/Sum2' */
  real_T Cutoff1Hz1;                   /* '<S2>/Cut-off 1 Hz1' */
  real_T Sum2_m;                       /* '<Root>/Sum2' */
  real_T em_null;                      /* '<Root>/em_null' */
  real_T Sum5;                         /* '<S2>/Sum5' */
  real_T Sum;                          /* '<S2>/Sum' */
  real_T P;                            /* '<S2>/P' */
  real_T D;                            /* '<S2>/D' */
  real_T Derivative;                   /* '<S2>/Derivative' */
  real_T Saturation;                   /* '<S2>/Saturation' */
  real_T Integrator;                   /* '<S2>/Integrator' */
  real_T Saturation1;                  /* '<S2>/Saturation1' */
  real_T Derivative1;                  /* '<S2>/Derivative1' */
  real_T ddt_gain;                     /* '<S2>/ddt_gain' */
  real_T Saturation2;                  /* '<S2>/Saturation2' */
  real_T Sum1;                         /* '<S2>/Sum1' */
  real_T UniformRandomNumber01ms;      /* '<S2>/Uniform Random Number 0.1 ms' */
  real_T Cutoff35Hz;                   /* '<S2>/Cut-off 35 Hz' */
  real_T didder_gain;                  /* '<S2>/didder_gain' */
  real_T Sum3;                         /* '<S2>/Sum3' */
  real_T Gain;                         /* '<Root>/Gain' */
  real_T Gain3;                        /* '<Root>/Gain3' */
  real_T position_switch4;             /* '<Root>/position_switch4' */
  real_T scope_multiport_switchFirstinpu;
  /* '<Root>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */
  real_T Sum1_l;                       /* '<Root>/Sum1' */
  real_T Scope_gain;                   /* '<Root>/Scope_gain' */
  real_T MVC_LUT;                      /* '<S3>/MVC_LUT' */
  real_T AbsoluteMVCtorqueinNm;        /* '<S3>/Abs2' */
  real_T desiredtorqueinNm;            /* '<S3>/PERCENT_MVC' */
  real_T RateTransition1;              /* '<S9>/Rate Transition1' */
  real_T RateTransition2;              /* '<S9>/Rate Transition2' */
  real_T RateTransition3;              /* '<S9>/Rate Transition3' */
  real_T RateTransition4;              /* '<S9>/Rate Transition4' */
  real_T RateTransition5;              /* '<S9>/Rate Transition5' */
  real_T scope_multiport_switchFirstin_a;
  /* '<S9>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal1' */
  real_T Switch;                       /* '<S9>/Switch' */
  real_T MultiportSwitch;              /* '<S8>/Multiport Switch' */
  real_T Divide1;                      /* '<S3>/Divide1' */
  real_T Divide;                       /* '<S3>/Divide' */
  real_T scope_multiport_switchFirsti_aa[2];
  /* '<S3>/scope_multiport_switch First input is the desired position Second inpt is the measured position Third input is measured Torque Fourth input is visual feedback signal' */
  real_T Sum4;                         /* '<S3>/Sum4' */
  real_T Gain2;                        /* '<Root>/Gain2' */
  real_T measuredtorqueinNm;           /* '<Root>/volt_to_tq' */
  real_T DeadZone;                     /* '<S2>/Dead Zone' */
  real_T I;                            /* '<S2>/I' */
  real_T Passive_LUT;                  /* '<S3>/Passive_LUT' */
  real_T measuredtorqueinNm_j;         /* '<S3>/volt_to_tq' */
  real_T Sum5_d;                       /* '<S3>/Sum5' */
  real_T absolutetorquewithoutpassiveinN;/* '<S3>/Abs1' */
  real_T SineWave;                     /* '<S3>/Sine Wave' */
  real_T Clock;                        /* '<Root>/Clock' */
  real_T FF_stat;                      /* '<S1>/Digital input ' */
  real_T Digitalinput_o2;              /* '<S1>/Digital input ' */
  real_T FwdFlow_Stat;                 /* '<S1>/FwdFlow_Stat' */
  real_T Memory7;                      /* '<S4>/Memory7' */
  real_T Switch_g;                     /* '<S5>/Switch' */
  real_T Hearbeat;                     /* '<S4>/Hearbeat' */
  real_T Heart_Beat;                   /* '<S4>/Switch' */
  real_T Product1;                     /* '<S4>/Product1' */
  real_T NumActuators;                 /* '<S1>/NumActuators' */
  real_T ForPRBSwith02sbaseswitchtime;
                                /* '<S9>/For PRBS with 0.2s base switch time' */
  real_T ForPRBSwith05sbaseswitchtime;
                                /* '<S9>/For PRBS with 0.5s base switch time' */
  real_T ForPRBSwith07sbaseswitchtime;
                                /* '<S9>/For PRBS with 0.7s base switch time' */
  real_T ForPRBSwith1sbaseswitchtime;
                                  /* '<S9>/For PRBS with 1s base switch time' */
  real_T ForPRBSwith2sbaseswitchtime;
                                  /* '<S9>/For PRBS with 2s base switch time' */
  real_T decimal_to_percent[2];        /* '<S3>/decimal_to_percent' */
  real_T In;                           /* '<S6>/In' */
  uint8_T BytePacking[16];             /* '<S7>/Byte Packing' */
  uint8_T RateTransition[16];          /* '<S7>/Rate Transition' */
  boolean_T NSampleEnable1;            /* '<S2>/N-Sample Enable1' */
  boolean_T NSampleEnable;             /* '<S5>/N-Sample Enable' */
};

/* Block states (default storage) for system '<Root>' */
struct DW_Supine_SpeedGoat_v3_T {
  real_T TimeStampA;                   /* '<S2>/Derivative' */
  real_T LastUAtTimeA;                 /* '<S2>/Derivative' */
  real_T TimeStampB;                   /* '<S2>/Derivative' */
  real_T LastUAtTimeB;                 /* '<S2>/Derivative' */
  real_T TimeStampA_d;                 /* '<S2>/Derivative1' */
  real_T LastUAtTimeA_m;               /* '<S2>/Derivative1' */
  real_T TimeStampB_n;                 /* '<S2>/Derivative1' */
  real_T LastUAtTimeB_m;               /* '<S2>/Derivative1' */
  real_T UniformRandomNumber01ms_NextOut;/* '<S2>/Uniform Random Number 0.1 ms' */
  real_T m_bpLambda;                   /* '<S3>/MVC_LUT' */
  real_T m_yyA;                        /* '<S3>/MVC_LUT' */
  real_T m_yyB;                        /* '<S3>/MVC_LUT' */
  real_T m_yy2;                        /* '<S3>/MVC_LUT' */
  real_T m_up[8];                      /* '<S3>/MVC_LUT' */
  real_T m_y2[8];                      /* '<S3>/MVC_LUT' */
  real_T prevBp0AndTableData[16];      /* '<S3>/MVC_LUT' */
  real_T RateTransition1_Buf[2];       /* '<S9>/Rate Transition1' */
  real_T RateTransition2_Buf[2];       /* '<S9>/Rate Transition2' */
  real_T RateTransition3_Buf[2];       /* '<S9>/Rate Transition3' */
  real_T RateTransition4_Buf[2];       /* '<S9>/Rate Transition4' */
  real_T RateTransition5_Buf[2];       /* '<S9>/Rate Transition5' */
  real_T m_bpLambda_p;                 /* '<S3>/Passive_LUT' */
  real_T m_yyA_h;                      /* '<S3>/Passive_LUT' */
  real_T m_yyB_b;                      /* '<S3>/Passive_LUT' */
  real_T m_yy2_j;                      /* '<S3>/Passive_LUT' */
  real_T m_up_k[8];                    /* '<S3>/Passive_LUT' */
  real_T m_y2_k[8];                    /* '<S3>/Passive_LUT' */
  real_T prevBp0AndTableData_i[16];    /* '<S3>/Passive_LUT' */
  real_T Memory7_PreviousInput;        /* '<S4>/Memory7' */
  real_T ForPRBSwith02sbaseswitchtime_Ne;
                                /* '<S9>/For PRBS with 0.2s base switch time' */
  real_T ForPRBSwith05sbaseswitchtime_Ne;
                                /* '<S9>/For PRBS with 0.5s base switch time' */
  real_T ForPRBSwith07sbaseswitchtime_Ne;
                                /* '<S9>/For PRBS with 0.7s base switch time' */
  real_T ForPRBSwith1sbaseswitchtime_Nex;
                                  /* '<S9>/For PRBS with 1s base switch time' */
  real_T ForPRBSwith2sbaseswitchtime_Nex;
                                  /* '<S9>/For PRBS with 2s base switch time' */
  real_T Setup_RWORK[2];               /* '<Root>/Setup ' */
  void *Setup_PWORK;                   /* '<Root>/Setup ' */
  struct {
    void *LoggedData;
  } AbsoluteVoluntaryTorquewithoutP;
             /* '<Root>/Absolute Voluntary Torque, without Passive TQ, in Nm' */

  void *Analoginput_PWORK;             /* '<Root>/Analog input ' */
  void *Analogoutput_PWORK;            /* '<Root>/Analog output ' */
  struct {
    void *LoggedData;
  } Scope_PWORK;                       /* '<Root>/Scope' */

  struct {
    void *LoggedData;
  } Scope1_PWORK;                      /* '<Root>/Scope1' */

  struct {
    void *LoggedData;
  } Scope2_PWORK;                      /* '<Root>/Scope2' */

  struct {
    void *LoggedData;
  } Scope3_PWORK;                      /* '<Root>/Scope3' */

  struct {
    void *LoggedData;
  } Scope4_PWORK;                      /* '<Root>/Scope4' */

  struct {
    void *LoggedData[2];
  } Scope6_PWORK;                      /* '<Root>/Scope6' */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_Analo;   /* synthesized block */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_Ana_e;   /* synthesized block */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_From4;   /* synthesized block */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_Servo;   /* synthesized block */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_Ser_n;   /* synthesized block */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_em_nu;   /* synthesized block */

  void* m_bpDataSet;                   /* '<S3>/MVC_LUT' */
  void* TWork[6];                      /* '<S3>/MVC_LUT' */
  void* SWork[9];                      /* '<S3>/MVC_LUT' */
  struct {
    void *LoggedData;
  } TQTrackingErrorMVC_PWORK;          /* '<Root>/TQ Tracking Error (%MVC)' */

  struct {
    void *LoggedData;
  } measuredtorqueinNm_PWORK;          /* '<Root>/measured torque in Nm' */

  struct {
    void *LoggedData;
  } DesiredTQAbsoluteNm_PWORK;         /* '<S3>/Desired TQ Absolute (Nm)' */

  struct {
    void *LoggedData;
  } Scope_PWORK_m;                     /* '<S3>/Scope' */

  void* m_bpDataSet_e;                 /* '<S3>/Passive_LUT' */
  void* TWork_e[6];                    /* '<S3>/Passive_LUT' */
  void* SWork_e[9];                    /* '<S3>/Passive_LUT' */
  struct {
    void *LoggedData;
  } Scope1_PWORK_f;                    /* '<S3>/Scope1' */

  struct {
    void *LoggedData;
  } ToWorkspace2_PWORK;                /* '<Root>/To Workspace2' */

  void *Digitalinput_PWORK;            /* '<S1>/Digital input ' */
  struct {
    void *LoggedData;
  } Scope_PWORK_a;                     /* '<S1>/Scope' */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_Digit;   /* synthesized block */

  void *Digitaloutput_PWORK;           /* '<S4>/Digital output ' */
  struct {
    void *LoggedData;
  } Scope_PWORK_i;                     /* '<S4>/Scope' */

  struct {
    void *AQHandles;
  } TAQSigLogging_InsertedFor_Switc;   /* synthesized block */

  void* UDPSend1_DWORK1;               /* '<S7>/UDP Send1' */
  void *UDPSend1_PWORK;                /* '<S7>/UDP Send1' */
  int32_T clockTickCounter;            /* '<S4>/Hearbeat' */
  uint32_T NSampleEnable1_Counter;     /* '<S2>/N-Sample Enable1' */
  uint32_T RandSeed;                   /* '<S2>/Uniform Random Number 0.1 ms' */
  uint32_T m_bpIndex;                  /* '<S3>/MVC_LUT' */
  uint32_T m_bpIndex_d;                /* '<S3>/Passive_LUT' */
  uint32_T NSampleEnable_Counter;      /* '<S5>/N-Sample Enable' */
  uint32_T NSampleEnable_EPHState;     /* '<S5>/N-Sample Enable' */
  uint32_T RandSeed_n;          /* '<S9>/For PRBS with 0.2s base switch time' */
  uint32_T RandSeed_h;          /* '<S9>/For PRBS with 0.5s base switch time' */
  uint32_T RandSeed_g;          /* '<S9>/For PRBS with 0.7s base switch time' */
  uint32_T RandSeed_j;            /* '<S9>/For PRBS with 1s base switch time' */
  uint32_T RandSeed_b;            /* '<S9>/For PRBS with 2s base switch time' */
  int_T Analoginput_IWORK[2];          /* '<Root>/Analog input ' */
  int_T BytePacking_IWORK[2];          /* '<S7>/Byte Packing' */
  int_T UDPSend1_IWORK[2];             /* '<S7>/UDP Send1' */
  uint16_T UDPSend1_DWORK2;            /* '<S7>/UDP Send1' */
  int8_T RateTransition1_RdBufIdx;     /* '<S9>/Rate Transition1' */
  int8_T RateTransition1_WrBufIdx;     /* '<S9>/Rate Transition1' */
  int8_T RateTransition2_RdBufIdx;     /* '<S9>/Rate Transition2' */
  int8_T RateTransition2_WrBufIdx;     /* '<S9>/Rate Transition2' */
  int8_T RateTransition3_RdBufIdx;     /* '<S9>/Rate Transition3' */
  int8_T RateTransition3_WrBufIdx;     /* '<S9>/Rate Transition3' */
  int8_T RateTransition4_RdBufIdx;     /* '<S9>/Rate Transition4' */
  int8_T RateTransition4_WrBufIdx;     /* '<S9>/Rate Transition4' */
  int8_T RateTransition5_RdBufIdx;     /* '<S9>/Rate Transition5' */
  int8_T RateTransition5_WrBufIdx;     /* '<S9>/Rate Transition5' */
  int8_T RateTransition_RdBufIdx;      /* '<S7>/Rate Transition' */
  int8_T RateTransition_WrBufIdx;      /* '<S7>/Rate Transition' */
  int8_T SampleandHold_SubsysRanBC;    /* '<S2>/Sample and Hold' */
  uint8_T reCalcSecDerivFirstDimCoeffs;/* '<S3>/MVC_LUT' */
  uint8_T RateTransition_Buf[32];      /* '<S7>/Rate Transition' */
  uint8_T reCalcSecDerivFirstDimCoeffs_b;/* '<S3>/Passive_LUT' */
};

/* Continuous states (default storage) */
struct X_Supine_SpeedGoat_v3_T {
  real_T uthorderBeselwithcutoff07HzbyGo[8];
  /* '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
  real_T Cutoff1Hz1_CSTATE[2];         /* '<S2>/Cut-off 1 Hz1' */
  real_T Integrator_CSTATE;            /* '<S2>/Integrator' */
  real_T Cutoff35Hz_CSTATE[5];         /* '<S2>/Cut-off 35 Hz' */
};

/* State derivatives (default storage) */
struct XDot_Supine_SpeedGoat_v3_T {
  real_T uthorderBeselwithcutoff07HzbyGo[8];
  /* '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
  real_T Cutoff1Hz1_CSTATE[2];         /* '<S2>/Cut-off 1 Hz1' */
  real_T Integrator_CSTATE;            /* '<S2>/Integrator' */
  real_T Cutoff35Hz_CSTATE[5];         /* '<S2>/Cut-off 35 Hz' */
};

/* State disabled  */
struct XDis_Supine_SpeedGoat_v3_T {
  boolean_T uthorderBeselwithcutoff07HzbyGo[8];
  /* '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  ' */
  boolean_T Cutoff1Hz1_CSTATE[2];      /* '<S2>/Cut-off 1 Hz1' */
  boolean_T Integrator_CSTATE;         /* '<S2>/Integrator' */
  boolean_T Cutoff35Hz_CSTATE[5];      /* '<S2>/Cut-off 35 Hz' */
};

/* Zero-crossing (trigger) state */
struct PrevZCX_Supine_SpeedGoat_v3_T {
  ZCSigState SampleandHold_Trig_ZCE;   /* '<S2>/Sample and Hold' */
};

#ifndef ODE5_INTG
#define ODE5_INTG

/* ODE5 Integration Data */
struct ODE5_IntgData {
  real_T *y;                           /* output */
  real_T *f[6];                        /* derivatives */
};

#endif

/* External inputs (root inport signals with default storage) */
struct ExtU_Supine_SpeedGoat_v3_T {
  real_T In1;                          /* '<Root>/In1' */
  real_T time_varying_mvc;             /* '<Root>/time_varying_mvc' */
};

/* Real-time Model Data Structure */
struct tag_RTM_Supine_SpeedGoat_v3_T {
  struct SimStruct_tag * *childSfunctions;
  const char_T *errorStatus;
  SS_SimMode simMode;
  RTWSolverInfo solverInfo;
  RTWSolverInfo *solverInfoPtr;
  void *sfcnInfo;

  /*
   * NonInlinedSFcns:
   * The following substructure contains information regarding
   * non-inlined s-functions used in the model.
   */
  struct {
    RTWSfcnInfo sfcnInfo;
    time_T *taskTimePtrs[8];
    SimStruct childSFunctions[5];
    SimStruct *childSFunctionPtrs[5];
    struct _ssBlkInfo2 blkInfo2[5];
    struct _ssBlkInfoSLSize blkInfoSLSize[5];
    struct _ssSFcnModelMethods2 methods2[5];
    struct _ssSFcnModelMethods3 methods3[5];
    struct _ssSFcnModelMethods4 methods4[5];
    struct _ssStatesInfo2 statesInfo2[5];
    ssPeriodicStatesInfo periodicStatesInfo[5];
    struct _ssPortInfo2 inputOutputPortInfo2[5];
    struct {
      time_T sfcnPeriod[1];
      time_T sfcnOffset[1];
      int_T sfcnTsMap[1];
      uint_T attribs[9];
      mxArray *params[9];
      struct _ssDWorkRecord dWork[2];
      struct _ssDWorkAuxRecord dWorkAux[2];
    } Sfcn0;

    struct {
      time_T sfcnPeriod[1];
      time_T sfcnOffset[1];
      int_T sfcnTsMap[1];
      struct _ssPortOutputs outputPortInfo[8];
      struct _ssPortOutputsSLSize outputPortInfoSLSize[8];
      struct _ssOutPortUnit outputPortUnits[8];
      struct _ssOutPortCoSimAttribute outputPortCoSimAttribute[8];
      uint_T attribs[9];
      mxArray *params[9];
      struct _ssDWorkRecord dWork[2];
      struct _ssDWorkAuxRecord dWorkAux[2];
    } Sfcn1;

    struct {
      time_T sfcnPeriod[1];
      time_T sfcnOffset[1];
      int_T sfcnTsMap[1];
      struct _ssPortInputs inputPortInfo[4];
      struct _ssPortInputsSLSize inputPortInfoSLSize[4];
      struct _ssInPortUnit inputPortUnits[4];
      struct _ssInPortCoSimAttribute inputPortCoSimAttribute[4];
      uint_T attribs[7];
      mxArray *params[7];
      struct _ssDWorkRecord dWork[1];
      struct _ssDWorkAuxRecord dWorkAux[1];
    } Sfcn2;

    struct {
      time_T sfcnPeriod[1];
      time_T sfcnOffset[1];
      int_T sfcnTsMap[1];
      struct _ssPortOutputs outputPortInfo[2];
      struct _ssPortOutputsSLSize outputPortInfoSLSize[2];
      struct _ssOutPortUnit outputPortUnits[2];
      struct _ssOutPortCoSimAttribute outputPortCoSimAttribute[2];
      uint_T attribs[4];
      mxArray *params[4];
      struct _ssDWorkRecord dWork[1];
      struct _ssDWorkAuxRecord dWorkAux[1];
    } Sfcn3;

    struct {
      time_T sfcnPeriod[1];
      time_T sfcnOffset[1];
      int_T sfcnTsMap[1];
      struct _ssPortInputs inputPortInfo[14];
      struct _ssPortInputsSLSize inputPortInfoSLSize[14];
      struct _ssInPortUnit inputPortUnits[14];
      struct _ssInPortCoSimAttribute inputPortCoSimAttribute[14];
      uint_T attribs[6];
      mxArray *params[6];
      struct _ssDWorkRecord dWork[1];
      struct _ssDWorkAuxRecord dWorkAux[1];
    } Sfcn4;
  } NonInlinedSFcns;

  X_Supine_SpeedGoat_v3_T *contStates;
  int_T *periodicContStateIndices;
  real_T *periodicContStateRanges;
  real_T *derivs;
  boolean_T *contStateDisabled;
  boolean_T zCCacheNeedsReset;
  boolean_T derivCacheNeedsReset;
  boolean_T CTOutputIncnstWithState;
  real_T odeY[16];
  real_T odeF[6][16];
  ODE5_IntgData intgData;

  /*
   * Sizes:
   * The following substructure contains sizes information
   * for many of the model attributes such as inputs, outputs,
   * dwork, sample times, etc.
   */
  struct {
    uint32_T options;
    int_T numContStates;
    int_T numPeriodicContStates;
    int_T numU;
    int_T numY;
    int_T numSampTimes;
    int_T numBlocks;
    int_T numBlockIO;
    int_T numBlockPrms;
    int_T numDwork;
    int_T numSFcnPrms;
    int_T numSFcns;
    int_T numIports;
    int_T numOports;
    int_T numNonSampZCs;
    int_T sysDirFeedThru;
    int_T rtwGenSfcn;
  } Sizes;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    time_T stepSize;
    uint32_T clockTick0;
    uint32_T clockTickH0;
    time_T stepSize0;
    uint32_T clockTick1;
    uint32_T clockTickH1;
    time_T stepSize1;
    uint32_T clockTick2;
    uint32_T clockTickH2;
    uint32_T clockTick3;
    uint32_T clockTickH3;
    uint32_T clockTick4;
    uint32_T clockTickH4;
    uint32_T clockTick5;
    uint32_T clockTickH5;
    uint32_T clockTick6;
    uint32_T clockTickH6;
    uint32_T clockTick7;
    uint32_T clockTickH7;
    struct {
      uint16_T TID[8];
      uint16_T cLimit[8];
    } TaskCounters;

    struct {
      uint16_T TID1_2;
      uint16_T TID1_3;
      uint16_T TID1_4;
      uint16_T TID1_5;
      uint16_T TID1_6;
      uint16_T TID1_7;
    } RateInteraction;

    time_T tStart;
    time_T tFinal;
    time_T timeOfLastOutput;
    SimTimeStep simTimeStep;
    boolean_T stopRequestedFlag;
    time_T *sampleTimes;
    time_T *offsetTimes;
    int_T *sampleTimeTaskIDPtr;
    int_T *sampleHits;
    int_T *perTaskSampleHits;
    time_T *t;
    time_T sampleTimesArray[8];
    time_T offsetTimesArray[8];
    int_T sampleTimeTaskIDArray[8];
    int_T sampleHitArray[8];
    int_T perTaskSampleHitsArray[64];
    time_T tArray[8];
  } Timing;
};

/* Block signals (default storage) */
#ifdef __cplusplus

extern "C" {

#endif

  extern struct B_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_B;

#ifdef __cplusplus

}
#endif

/* Continuous states (default storage) */
extern X_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_X;

/* Block states (default storage) */
extern struct DW_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_DW;

/* Zero-crossing (trigger) state */
extern PrevZCX_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_PrevZCX;

#ifdef __cplusplus

extern "C" {

#endif

  /* External inputs (root inport signals with default storage) */
  extern struct ExtU_Supine_SpeedGoat_v3_T Supine_SpeedGoat_v3_U;

#ifdef __cplusplus

}
#endif

/* External data declarations for dependent source files */
extern const real_T Supine_SpeedGoat_v3_RGND;/* real_T ground */

#ifdef __cplusplus

extern "C" {

#endif

  /* Model entry point functions */
  extern void Supine_SpeedGoat_v3_initialize(void);
  extern void Supine_SpeedGoat_v3_step0(void);
  extern void Supine_SpeedGoat_v3_step2(void);
  extern void Supine_SpeedGoat_v3_step3(void);
  extern void Supine_SpeedGoat_v3_step4(void);
  extern void Supine_SpeedGoat_v3_step5(void);
  extern void Supine_SpeedGoat_v3_step6(void);
  extern void Supine_SpeedGoat_v3_step7(void);
  extern void Supine_SpeedGoat_v3_terminate(void);

#ifdef __cplusplus

}
#endif

/* Real-time Model object */
#ifdef __cplusplus

extern "C" {

#endif

  extern RT_MODEL_Supine_SpeedGoat_v3_T *const Supine_SpeedGoat_v3_M;

#ifdef __cplusplus

}
#endif

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'Supine_SpeedGoat_v3'
 * '<S1>'   : 'Supine_SpeedGoat_v3/Digital Outputs'
 * '<S2>'   : 'Supine_SpeedGoat_v3/Servo Controller'
 * '<S3>'   : 'Supine_SpeedGoat_v3/Visual Feedback Controller'
 * '<S4>'   : 'Supine_SpeedGoat_v3/Digital Outputs/Trigger Data Acquisition'
 * '<S5>'   : 'Supine_SpeedGoat_v3/Digital Outputs/Trigger Data Acquisition/N-Sample Switch'
 * '<S6>'   : 'Supine_SpeedGoat_v3/Servo Controller/Sample and Hold'
 * '<S7>'   : 'Supine_SpeedGoat_v3/Visual Feedback Controller/Data to TWITCH (measured and desired torque)'
 * '<S8>'   : 'Supine_SpeedGoat_v3/Visual Feedback Controller/target_mvc_generator'
 * '<S9>'   : 'Supine_SpeedGoat_v3/Visual Feedback Controller/target_mvc_generator/PRBS GENERATOR'
 */
#endif                                 /* RTW_HEADER_Supine_SpeedGoat_v3_h_ */
