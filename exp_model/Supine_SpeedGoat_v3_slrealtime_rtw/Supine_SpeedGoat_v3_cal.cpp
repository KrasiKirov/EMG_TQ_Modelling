#include "Supine_SpeedGoat_v3_cal.h"
#include "Supine_SpeedGoat_v3.h"

/* Storage class 'PageSwitching' */
Supine_SpeedGoat_v3_cal_type Supine_SpeedGoat_v3_cal_impl = {
  /* Mask Parameter: NSampleEnable1_ActiveLevel
   * Referenced by: '<S2>/N-Sample Enable1'
   */
  2U,

  /* Mask Parameter: NSampleEnable_ActiveLevel
   * Referenced by: '<S5>/N-Sample Enable'
   */
  2U,

  /* Mask Parameter: NSampleEnable1_N
   * Referenced by: '<S2>/N-Sample Enable1'
   */
  100U,

  /* Mask Parameter: NSampleSwitch_N
   * Referenced by: '<S5>/N-Sample Enable'
   */
  1000U,

  /* Expression: 0
   * Referenced by: '<S4>/Constant1'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<S4>/Constant2'
   */
  1.0,

  /* Expression: initCond
   * Referenced by: '<S6>/ '
   */
  0.0,

  /* Expression: 100
   * Referenced by: '<S3>/decimal_to_percent'
   */
  100.0,

  /* Expression: 0
   * Referenced by: '<S9>/Zero_MVC'
   */
  0.0,

  /* Computed Parameter: Setup_P1_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 1.0 },

  /* Expression: parPciSlot
   * Referenced by: '<Root>/Setup '
   */
  -1.0,

  /* Computed Parameter: Setup_P2_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 1.0 },

  /* Expression: parModuleId
   * Referenced by: '<Root>/Setup '
   */
  1.0,

  /* Computed Parameter: Setup_P3_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 1.0 },

  /* Expression: parTriggerSignal
   * Referenced by: '<Root>/Setup '
   */
  1.0,

  /* Computed Parameter: Setup_P4_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 8.0 },

  /* Expression: parAdcChannels
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 },

  /* Computed Parameter: Setup_P5_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 1.0 },

  /* Expression: parAdcMode
   * Referenced by: '<Root>/Setup '
   */
  2.0,

  /* Computed Parameter: Setup_P6_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 8.0 },

  /* Expression: parAdcRanges
   * Referenced by: '<Root>/Setup '
   */
  { 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },

  /* Computed Parameter: Setup_P7_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 4.0 },

  /* Expression: parDacChannels
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 2.0, 3.0, 4.0 },

  /* Computed Parameter: Setup_P8_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 4.0 },

  /* Expression: parDacRanges
   * Referenced by: '<Root>/Setup '
   */
  { 4.0, 4.0, 4.0, 4.0 },

  /* Computed Parameter: Setup_P9_Size
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 7.0 },

  /* Expression: parDioFirstControl
   * Referenced by: '<Root>/Setup '
   */
  { 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0 },

  /* Computed Parameter: uthorderBeselwithcutoff07HzbyGo
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  { -228.62831326326926, 115.68388472262141, -115.68388472262139,
    136.50446714861272, -212.95601978172724, 119.17214036828834,
    -119.17214036828837, 132.50888163073597, -178.72851877127863,
    127.04953833045536, -127.04953833045536, 124.2929903505016,
    -116.1570058144798, 142.370090573767, -142.370090573767 },

  /* Computed Parameter: uthorderBeselwithcutoff07Hzby_l
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  125.66370614359172,

  /* Computed Parameter: uthorderBeselwithcutoff07Hzby_o
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  0.88265523774799393,

  /* Expression: 0
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  0.0,

  /* Computed Parameter: Analoginput_P1_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 1.0 },

  /* Expression: parModuleId
   * Referenced by: '<Root>/Analog input '
   */
  1.0,

  /* Computed Parameter: Analoginput_P2_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 1.0 },

  /* Expression: parSampleTime
   * Referenced by: '<Root>/Analog input '
   */
  0.001,

  /* Computed Parameter: Analoginput_P3_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 1.0 },

  /* Expression: parPciSlot
   * Referenced by: '<Root>/Analog input '
   */
  -1.0,

  /* Computed Parameter: Analoginput_P4_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 8.0 },

  /* Expression: parAdcChannels
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 },

  /* Computed Parameter: Analoginput_P5_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 1.0 },

  /* Expression: parAdcMode
   * Referenced by: '<Root>/Analog input '
   */
  2.0,

  /* Computed Parameter: Analoginput_P6_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 1.0 },

  /* Expression: parAdcRate
   * Referenced by: '<Root>/Analog input '
   */
  100000.0,

  /* Computed Parameter: Analoginput_P7_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 8.0 },

  /* Expression: parAdcRanges
   * Referenced by: '<Root>/Analog input '
   */
  { 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0 },

  /* Computed Parameter: Analoginput_P8_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 2.0 },

  /* Expression: parAdcInitValues
   * Referenced by: '<Root>/Analog input '
   */
  { 0.0, 0.0 },

  /* Computed Parameter: Analoginput_P9_Size
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 2.0 },

  /* Expression: parAdcResets
   * Referenced by: '<Root>/Analog input '
   */
  { 1.0, 1.0 },

  /* Expression: 0.1
   * Referenced by: '<Root>/Gain1'
   */
  0.1,

  /* Computed Parameter: Cutoff1Hz1_A_pr
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  { -10.882796185405306, 6.2831853071795845, -6.2831853071795862 },

  /* Computed Parameter: Cutoff1Hz1_B_pr
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  6.2831853071795862,

  /* Computed Parameter: Cutoff1Hz1_C_pr
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  1.0000000000000002,

  /* Expression: 0
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<Root>/em_null'
   */
  1.0,

  /* Expression: 67
   * Referenced by: '<S2>/P'
   */
  67.0,

  /* Expression: 0.01
   * Referenced by: '<S2>/D'
   */
  0.01,

  /* Expression: 0.5
   * Referenced by: '<S2>/Saturation'
   */
  0.5,

  /* Expression: -0.5
   * Referenced by: '<S2>/Saturation'
   */
  -0.5,

  /* Expression: 0
   * Referenced by: '<S2>/Integrator'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<S2>/Integrator'
   */
  1.0,

  /* Expression: -1
   * Referenced by: '<S2>/Integrator'
   */
  -1.0,

  /* Expression: 1
   * Referenced by: '<S2>/Saturation1'
   */
  1.0,

  /* Expression: -1
   * Referenced by: '<S2>/Saturation1'
   */
  -1.0,

  /* Expression: 0
   * Referenced by: '<S2>/ddt_gain'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<S2>/Saturation2'
   */
  1.0,

  /* Expression: -1
   * Referenced by: '<S2>/Saturation2'
   */
  -1.0,

  /* Expression: -1
   * Referenced by: '<S2>/Uniform Random Number 0.1 ms'
   */
  -1.0,

  /* Expression: 1
   * Referenced by: '<S2>/Uniform Random Number 0.1 ms'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S2>/Uniform Random Number 0.1 ms'
   */
  0.0,

  /* Computed Parameter: Cutoff35Hz_A_pr
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { -271.28238064902894, 2.7902947984069054E-14, -282.65600751005906,
    -272.061667866601, -5.5805895968138108E-14, 2.5850465095603549E-14,
    -261.8644187837495, -252.04937675591268, -5.1589978753168567E-14,
    261.86441878374939, -464.68121817764353, -447.26432088726983,
    -3.097849530683156E-30, -2.7902947984069054E-14, -232.17334754392377,
    -6.1956990613663119E-30, -5.5805895968138108E-14, 232.17334754392374,
    -253.332328919049 },

  /* Computed Parameter: Cutoff35Hz_B_pr
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 271.28238064902888, 282.65600751005894, 272.06166786660094 },

  /* Computed Parameter: Cutoff35Hz_C_pr
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { -1.0000000000000002, -0.92644207738776041, -1.64398139728591,
    -0.85338500408578133, -0.93115774414521535 },

  /* Computed Parameter: Cutoff35Hz_D_pr
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  1.0000000000000002,

  /* Expression: 0
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  0.0,

  /* Expression: 0
   * Referenced by: '<S2>/didder_gain'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<Root>/Gain'
   */
  1.0,

  /* Expression: 10
   * Referenced by: '<Root>/Gain3'
   */
  10.0,

  /* Computed Parameter: Analogoutput_P1_Size
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 1.0 },

  /* Expression: parModuleId
   * Referenced by: '<Root>/Analog output '
   */
  1.0,

  /* Computed Parameter: Analogoutput_P2_Size
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 1.0 },

  /* Expression: parSampleTime
   * Referenced by: '<Root>/Analog output '
   */
  0.001,

  /* Computed Parameter: Analogoutput_P3_Size
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 1.0 },

  /* Expression: parPciSlot
   * Referenced by: '<Root>/Analog output '
   */
  -1.0,

  /* Computed Parameter: Analogoutput_P4_Size
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 4.0 },

  /* Expression: parDacChannels
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 2.0, 3.0, 4.0 },

  /* Computed Parameter: Analogoutput_P5_Size
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 4.0 },

  /* Expression: parDacRanges
   * Referenced by: '<Root>/Analog output '
   */
  { 4.0, 4.0, 4.0, 4.0 },

  /* Computed Parameter: Analogoutput_P6_Size
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 4.0 },

  /* Expression: parDacInitValues
   * Referenced by: '<Root>/Analog output '
   */
  { 0.0, 0.0, 0.0, 0.0 },

  /* Computed Parameter: Analogoutput_P7_Size
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 4.0 },

  /* Expression: parDacResets
   * Referenced by: '<Root>/Analog output '
   */
  { 1.0, 1.0, 1.0, 1.0 },

  /* Expression: 0
   * Referenced by: '<Root>/Scope_offset'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<Root>/scope_switch_signal'
   */
  1.0,

  /* Expression: 1
   * Referenced by: '<Root>/position_switch4'
   */
  1.0,

  /* Expression: 1
   * Referenced by: '<Root>/Scope_gain'
   */
  1.0,

  /* Expression: 1
   * Referenced by: '<S3>/Contraction_Profile_Selector'
   */
  1.0,

  /* Expression: 1
   * Referenced by: '<S3>/mvc_profile_selector'
   */
  1.0,

  /* Expression: 0.18
   * Referenced by: '<S3>/Constant'
   */
  0.18,

  /* Expression: -ones(n_angles,1)
   * Referenced by: '<S3>/MVC_TQ'
   */
  { -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0 },

  /* Expression: linspace(maxPF,maxDF,n_angles)
   * Referenced by: '<S3>/isometric_ankle_angles'
   */
  { -0.5, -0.4, -0.30000000000000004, -0.20000000000000007, -0.10000000000000003,
    0.0, 0.099999999999999867, 0.2 },

  /* Expression: 0.4
   * Referenced by: '<S3>/PERCENT_MVC'
   */
  0.4,

  /* Expression: 4
   * Referenced by: '<S9>/Viz_Fdbk_PRBS_Switch_Time_Selector'
   */
  4.0,

  /* Expression: 0
   * Referenced by: '<S9>/Rate Transition1'
   */
  0.0,

  /* Expression: 0
   * Referenced by: '<S9>/Rate Transition2'
   */
  0.0,

  /* Expression: 0
   * Referenced by: '<S9>/Rate Transition3'
   */
  0.0,

  /* Expression: 0
   * Referenced by: '<S9>/Rate Transition4'
   */
  0.0,

  /* Expression: 0
   * Referenced by: '<S9>/Rate Transition5'
   */
  0.0,

  /* Expression: 0
   * Referenced by: '<S9>/Switch'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<Root>/Gain2'
   */
  1.0,

  /* Expression: 20
   * Referenced by: '<Root>/volt_to_tq'
   */
  20.0,

  /* Expression: 0
   * Referenced by: '<Root>/POS_Offset'
   */
  0.0,

  /* Expression: -0.01
   * Referenced by: '<S2>/Dead Zone'
   */
  -0.01,

  /* Expression: 0.01
   * Referenced by: '<S2>/Dead Zone'
   */
  0.01,

  /* Expression: 0.02
   * Referenced by: '<S2>/I'
   */
  0.02,

  /* Expression: zeros(n_angles,1)
   * Referenced by: '<S3>/Passive_TQ'
   */
  { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },

  /* Expression: 20
   * Referenced by: '<S3>/volt_to_tq'
   */
  20.0,

  /* Expression: 1
   * Referenced by: '<S3>/Sine Wave'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S3>/Sine Wave'
   */
  0.0,

  /* Computed Parameter: Digitalinput_P1_Size
   * Referenced by: '<S1>/Digital input '
   */
  { 1.0, 1.0 },

  /* Expression: parModuleId
   * Referenced by: '<S1>/Digital input '
   */
  1.0,

  /* Computed Parameter: Digitalinput_P2_Size
   * Referenced by: '<S1>/Digital input '
   */
  { 1.0, 1.0 },

  /* Expression: parSampleTime
   * Referenced by: '<S1>/Digital input '
   */
  -1.0,

  /* Computed Parameter: Digitalinput_P3_Size
   * Referenced by: '<S1>/Digital input '
   */
  { 1.0, 1.0 },

  /* Expression: parPciSlot
   * Referenced by: '<S1>/Digital input '
   */
  -1.0,

  /* Computed Parameter: Digitalinput_P4_Size
   * Referenced by: '<S1>/Digital input '
   */
  { 1.0, 2.0 },

  /* Expression: parDiChannels
   * Referenced by: '<S1>/Digital input '
   */
  { 11.0, 12.0 },

  /* Expression: 1
   * Referenced by: '<S1>/FwdFlow_Stat'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S4>/Memory7'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<S4>/Hearbeat'
   */
  1.0,

  /* Computed Parameter: Hearbeat_Period
   * Referenced by: '<S4>/Hearbeat'
   */
  2.0,

  /* Computed Parameter: Hearbeat_Duty
   * Referenced by: '<S4>/Hearbeat'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S4>/Hearbeat'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<S1>/FwdFlow_Use'
   */
  1.0,

  /* Expression: 0.5
   * Referenced by: '<S4>/Switch'
   */
  0.5,

  /* Computed Parameter: Digitaloutput_P1_Size
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 1.0 },

  /* Expression: parModuleId
   * Referenced by: '<S4>/Digital output '
   */
  1.0,

  /* Computed Parameter: Digitaloutput_P2_Size
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 1.0 },

  /* Expression: parSampleTime
   * Referenced by: '<S4>/Digital output '
   */
  0.001,

  /* Computed Parameter: Digitaloutput_P3_Size
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 1.0 },

  /* Expression: parPciSlot
   * Referenced by: '<S4>/Digital output '
   */
  -1.0,

  /* Computed Parameter: Digitaloutput_P4_Size
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 14.0 },

  /* Expression: parDoChannels
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 13.0, 14.0, 15.0, 16.0 },

  /* Computed Parameter: Digitaloutput_P5_Size
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 14.0 },

  /* Expression: parDoInitValues
   * Referenced by: '<S4>/Digital output '
   */
  { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },

  /* Computed Parameter: Digitaloutput_P6_Size
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 14.0 },

  /* Expression: parDoResets
   * Referenced by: '<S4>/Digital output '
   */
  { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 },

  /* Expression: 0
   * Referenced by: '<S1>/Sample_Trigger'
   */
  0.0,

  /* Expression: 1
   * Referenced by: '<S1>/NumActuators'
   */
  1.0,

  /* Expression: 16
   * Referenced by: '<S7>/position_switch4'
   */
  16.0,

  /* Expression: -1
   * Referenced by: '<S9>/For PRBS with 0.2s base switch time'
   */
  -1.0,

  /* Expression: 1
   * Referenced by: '<S9>/For PRBS with 0.2s base switch time'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S9>/For PRBS with 0.2s base switch time'
   */
  0.0,

  /* Expression: -1
   * Referenced by: '<S9>/For PRBS with 0.5s base switch time'
   */
  -1.0,

  /* Expression: 1
   * Referenced by: '<S9>/For PRBS with 0.5s base switch time'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S9>/For PRBS with 0.5s base switch time'
   */
  0.0,

  /* Expression: -1
   * Referenced by: '<S9>/For PRBS with 0.7s base switch time'
   */
  -1.0,

  /* Expression: 1
   * Referenced by: '<S9>/For PRBS with 0.7s base switch time'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S9>/For PRBS with 0.7s base switch time'
   */
  0.0,

  /* Expression: -1
   * Referenced by: '<S9>/For PRBS with 1s base switch time'
   */
  -1.0,

  /* Expression: 1
   * Referenced by: '<S9>/For PRBS with 1s base switch time'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S9>/For PRBS with 1s base switch time'
   */
  0.0,

  /* Expression: -1
   * Referenced by: '<S9>/For PRBS with 2s base switch time'
   */
  -1.0,

  /* Expression: 1
   * Referenced by: '<S9>/For PRBS with 2s base switch time'
   */
  1.0,

  /* Expression: 0
   * Referenced by: '<S9>/For PRBS with 2s base switch time'
   */
  0.0,

  /* Computed Parameter: uthorderBeselwithcutoff07Hzby_d
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  { 0U, 1U, 0U, 2U, 2U, 3U, 2U, 4U, 4U, 5U, 4U, 6U, 6U, 7U, 6U },

  /* Computed Parameter: uthorderBeselwithcutoff07Hzb_de
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  { 0U, 2U, 4U, 6U, 8U, 10U, 12U, 14U, 15U },

  /* Computed Parameter: uthorderBeselwithcutoff07Hzby_a
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  0U,

  /* Computed Parameter: uthorderBeselwithcutoff07Hzby_i
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  { 0U, 1U },

  /* Computed Parameter: uthorderBeselwithcutoff07Hzb_ok
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  0U,

  /* Computed Parameter: uthorderBeselwithcutoff07Hzby_c
   * Referenced by: '<S3>/8th order Besel with  cut-off 0.7 Hz by Golkar et al. (2018). Ehsan used 20Hz beacuse Mahsa  used slow torque modulation but in  Ehsan experiments, voluntary torque is modulated fast (following PRBS).  '
   */
  { 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 1U },

  /* Computed Parameter: Cutoff1Hz1_A_ir
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  { 0U, 1U, 0U },

  /* Computed Parameter: Cutoff1Hz1_A_jc
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  { 0U, 2U, 3U },

  /* Computed Parameter: Cutoff1Hz1_B_ir
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  0U,

  /* Computed Parameter: Cutoff1Hz1_B_jc
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  { 0U, 1U },

  /* Computed Parameter: Cutoff1Hz1_C_ir
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  0U,

  /* Computed Parameter: Cutoff1Hz1_C_jc
   * Referenced by: '<S2>/Cut-off 1 Hz1'
   */
  { 0U, 0U, 1U },

  /* Computed Parameter: Cutoff35Hz_A_ir
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 0U, 1U, 2U, 4U, 0U, 1U, 2U, 4U, 0U, 1U, 2U, 4U, 0U, 2U, 4U, 0U, 2U, 3U, 4U },

  /* Computed Parameter: Cutoff35Hz_A_jc
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 0U, 4U, 8U, 12U, 15U, 19U },

  /* Computed Parameter: Cutoff35Hz_B_ir
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 0U, 2U, 4U },

  /* Computed Parameter: Cutoff35Hz_B_jc
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 0U, 3U },

  /* Computed Parameter: Cutoff35Hz_C_ir
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 0U, 0U, 0U, 0U, 0U },

  /* Computed Parameter: Cutoff35Hz_C_jc
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 0U, 1U, 2U, 3U, 4U, 5U },

  /* Computed Parameter: Cutoff35Hz_D_ir
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  0U,

  /* Computed Parameter: Cutoff35Hz_D_jc
   * Referenced by: '<S2>/Cut-off 35 Hz'
   */
  { 0U, 1U },

  /* Computed Parameter: MVC_LUT_maxIndex
   * Referenced by: '<S3>/MVC_LUT'
   */
  7U,

  /* Computed Parameter: MVC_LUT_dimSizes
   * Referenced by: '<S3>/MVC_LUT'
   */
  1U,

  /* Computed Parameter: MVC_LUT_numYWorkElts
   * Referenced by: '<S3>/MVC_LUT'
   */
  { 1U, 0U },

  /* Computed Parameter: Passive_LUT_maxIndex
   * Referenced by: '<S3>/Passive_LUT'
   */
  7U,

  /* Computed Parameter: Passive_LUT_dimSizes
   * Referenced by: '<S3>/Passive_LUT'
   */
  1U,

  /* Computed Parameter: Passive_LUT_numYWorkElts
   * Referenced by: '<S3>/Passive_LUT'
   */
  { 1U, 0U },

  /* Computed Parameter: UDPSend1_toPort
   * Referenced by: '<S7>/UDP Send1'
   */
  5005U,

  /* Computed Parameter: RateTransition_InitialCondition
   * Referenced by: '<S7>/Rate Transition'
   */
  0U,

  /* Computed Parameter: UDPSend1_toAddress
   * Referenced by: '<S7>/UDP Send1'
   */
  { 10U, 68U, 3U, 239U }
};

Supine_SpeedGoat_v3_cal_type *Supine_SpeedGoat_v3_cal =
  &Supine_SpeedGoat_v3_cal_impl;
