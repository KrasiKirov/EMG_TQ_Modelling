#ifndef _RTE_SUPINE_SPEEDGOAT_V__PARAMETERS_H
#define _RTE_SUPINE_SPEEDGOAT_V__PARAMETERS_H
#include "rtwtypes.h"
#include "SegmentInfo.hpp"
#include "multiword_types.h"
#include "zero_crossing_types.h"
#include "Supine_SpeedGoat_v3_types.h"

struct RTE_Param_Service_T {
  real_T maxDF;
  real_T maxPF;
};

extern RTE_Param_Service_T RTE_Param_Service;
extern RTE_Param_Service_T *RTE_Param_Service_ptr;
real_T* get_maxDF(void);
real_T* get_maxPF(void);
namespace slrealtime
{
  SegmentVector &getSegmentVector(void);
}                                      // slrealtime

#endif
