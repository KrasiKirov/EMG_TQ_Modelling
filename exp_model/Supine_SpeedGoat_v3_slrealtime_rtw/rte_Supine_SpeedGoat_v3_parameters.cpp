#include "rte_Supine_SpeedGoat_v3_parameters.h"
#include "Supine_SpeedGoat_v3.h"
#include "Supine_SpeedGoat_v3_cal.h"

RTE_Param_Service_T RTE_Param_Service = {
  0.2,
  -0.5
};

RTE_Param_Service_T *RTE_Param_Service_ptr = &RTE_Param_Service;
real_T* get_maxDF(void)
{
  return &RTE_Param_Service_ptr->maxDF;
}

real_T* get_maxPF(void)
{
  return &RTE_Param_Service_ptr->maxPF;
}

extern Supine_SpeedGoat_v3_cal_type Supine_SpeedGoat_v3_cal_impl;
extern RTE_Param_Service_T RTE_Param_Service;
namespace slrealtime
{
  /* Description of SEGMENTS */
  SegmentVector segmentInfo {
    { (void*)&RTE_Param_Service, (void**)&RTE_Param_Service_ptr, sizeof
      (RTE_Param_Service_T), 2 },

    { (void*)&Supine_SpeedGoat_v3_cal_impl, (void**)&Supine_SpeedGoat_v3_cal,
      sizeof(Supine_SpeedGoat_v3_cal_type), 2 }
  };

  SegmentVector &getSegmentVector(void)
  {
    return segmentInfo;
  }
}                                      // slrealtime
