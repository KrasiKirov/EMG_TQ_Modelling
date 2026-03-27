/* Main generated for Simulink Real-Time model Supine_SpeedGoat_v3 */
#include <ModelInfo.hpp>
#include <utilities.hpp>
#include "Supine_SpeedGoat_v3.h"
#include "rte_Supine_SpeedGoat_v3_parameters.h"

/* Task descriptors */
slrealtime::TaskInfo task_1( 0u, std::bind(Supine_SpeedGoat_v3_step0), slrealtime::TaskInfo::PERIODIC, 0.001, 0, 40);
slrealtime::TaskInfo task_2( 1u, std::bind(Supine_SpeedGoat_v3_step2), slrealtime::TaskInfo::PERIODIC, 0.02, 0, 39);
slrealtime::TaskInfo task_3( 2u, std::bind(Supine_SpeedGoat_v3_step3), slrealtime::TaskInfo::PERIODIC, 0.2, 0, 38);
slrealtime::TaskInfo task_4( 3u, std::bind(Supine_SpeedGoat_v3_step4), slrealtime::TaskInfo::PERIODIC, 0.5, 0, 37);
slrealtime::TaskInfo task_5( 4u, std::bind(Supine_SpeedGoat_v3_step5), slrealtime::TaskInfo::PERIODIC, 0.7, 0, 36);
slrealtime::TaskInfo task_6( 5u, std::bind(Supine_SpeedGoat_v3_step6), slrealtime::TaskInfo::PERIODIC, 1, 0, 35);
slrealtime::TaskInfo task_7( 6u, std::bind(Supine_SpeedGoat_v3_step7), slrealtime::TaskInfo::PERIODIC, 2, 0, 34);

/* Executable base address for XCP */
#ifdef __linux__
extern char __executable_start;
static uintptr_t const base_address = reinterpret_cast<uintptr_t>(&__executable_start);
#else
/* Set 0 as placeholder, to be parsed later from /proc filesystem */
static uintptr_t const base_address = 0;
#endif

/* Model descriptor */
slrealtime::ModelInfo Supine_SpeedGoat_v3_Info =
{
    "Supine_SpeedGoat_v3",
    Supine_SpeedGoat_v3_initialize,
    Supine_SpeedGoat_v3_terminate,
    []()->char const*& { return Supine_SpeedGoat_v3_M->errorStatus; },
    []()->unsigned char& { return Supine_SpeedGoat_v3_M->Timing.stopRequestedFlag; },
    { task_1, task_2, task_3, task_4, task_5, task_6, task_7 },
    slrealtime::getSegmentVector()
};

int main(int argc, char *argv[]) {
    slrealtime::BaseAddress::set(base_address);
    return slrealtime::runModel(argc, argv, Supine_SpeedGoat_v3_Info);
}
