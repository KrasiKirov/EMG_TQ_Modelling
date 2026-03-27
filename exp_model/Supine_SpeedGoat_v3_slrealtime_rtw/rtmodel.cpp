/*
 *  rtmodel.cpp:
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

#include "rtmodel.h"

/* Use this function only if you need to maintain compatibility with an existing static main program. */
void Supine_SpeedGoat_v3_step(int_T tid)
{
  switch (tid) {
   case 0 :
    Supine_SpeedGoat_v3_step0();
    break;

   case 1 :
    Supine_SpeedGoat_v3_step2();
    break;

   case 2 :
    Supine_SpeedGoat_v3_step3();
    break;

   case 3 :
    Supine_SpeedGoat_v3_step4();
    break;

   case 4 :
    Supine_SpeedGoat_v3_step5();
    break;

   case 5 :
    Supine_SpeedGoat_v3_step6();
    break;

   case 6 :
    Supine_SpeedGoat_v3_step7();
    break;

   default :
    /* do nothing */
    break;
  }
}
