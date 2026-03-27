###########################################################################
## Makefile generated for component 'Supine_SpeedGoat_v3'. 
## 
## Makefile     : Supine_SpeedGoat_v3.mk
## Generated on : Thu Mar 26 12:57:38 2026
## Final product: $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/Supine_SpeedGoat_v3
## Product type : executable
## 
###########################################################################

###########################################################################
## MACROS
###########################################################################

# Macro Descriptions:
# PRODUCT_NAME            Name of the system to build
# MAKEFILE                Name of this makefile

PRODUCT_NAME              = Supine_SpeedGoat_v3
MAKEFILE                  = Supine_SpeedGoat_v3.mk
MATLAB_ROOT               = C:/PROGRA~1/MATLAB/R2022a
MATLAB_BIN                = C:/PROGRA~1/MATLAB/R2022a/bin
MATLAB_ARCH_BIN           = $(MATLAB_BIN)/win64
START_DIR                 = G:/Users/ECSE2026/github_repos/EMG_TQ_Modelling/exp_model
SOLVER                    = 
SOLVER_OBJ                = 
CLASSIC_INTERFACE         = 0
TGT_FCN_LIB               = ISO_C++
MODEL_HAS_DYNAMICALLY_LOADED_SFCNS = 0
RELATIVE_PATH_TO_ANCHOR   = ../..
C_STANDARD_OPTS           = 
CPP_STANDARD_OPTS         = 
LIBLIBSLREALTIME_LIBSRC_IP_SLRT_X64_OBJS = 

###########################################################################
## TOOLCHAIN SPECIFICATIONS
###########################################################################

# Toolchain Name:          Simulink Real-Time Toolchain
# Supported Version(s):    
# ToolchainInfo Version:   2022a
# Specification Revision:  1.0
# 
#-------------------------------------------
# Macros assumed to be defined elsewhere
#-------------------------------------------

# SLREALTIME_QNX_SP_ROOT
# SLREALTIME_QNX_VERSION

#-----------
# MACROS
#-----------

QCC_TARGET             = gcc_ntox86_64

TOOLCHAIN_SRCS = 
TOOLCHAIN_INCS = 
TOOLCHAIN_LIBS = -L$(MATLAB_ROOT)/toolbox/slrealtime/target/win64/target/lib -ltraceparser -lpps -lslrealtime_kernel -lslrealtime_platform -lslrealtime_rtps -lsocket -lboost_system -lboost_log -lpci -lopenblas -lpcap

#------------------------
# BUILD TOOL COMMANDS
#------------------------

# C Compiler: QNX C Compiler
CC = qcc

# Linker: QCC Linker
LD = q++

# C++ Compiler: QNX C++ Compiler
CPP = q++

# C++ Linker: QCC C++ Linker
CPP_LD = q++

# Archiver: QNX Archiver
AR = ntox86_64-gcc-ar

# Builder: GMAKE Utility
MAKE = make


#-------------------------
# Directives/Utilities
#-------------------------

CDEBUG              = -g -O0 -finstrument-functions
C_OUTPUT_FLAG       = -o
LDDEBUG             = -g
OUTPUT_FLAG         = -o
CPPDEBUG            = -g -O0 -finstrument-functions
CPP_OUTPUT_FLAG     = -o
CPPLDDEBUG          = -g
OUTPUT_FLAG         = -o
ARDEBUG             =
STATICLIB_OUTPUT_FLAG =
RM                  = @del /F
ECHO                = @echo
MV                  = @move
RUN                 =

#--------------------------------------
# "Faster Runs" Build Configuration
#--------------------------------------

ARFLAGS              = ruvs
CFLAGS               = -c -V$(QCC_TARGET) -g \
                       -O2 -fwrapv
CPPFLAGS             = -c -V$(QCC_TARGET) -g -std=gnu++14 -stdlib=libstdc++ \
                       -O2 -fwrapv
CPP_LDFLAGS          = -V$(QCC_TARGET) -g -std=gnu++14 -stdlib=libstdc++
CPP_SHAREDLIB_LDFLAGS  = -V$(QCC_TARGET) -shared -Wl,--no-undefined -g
LDFLAGS              = -V$(QCC_TARGET) -g -std=gnu++14 -stdlib=libstdc++
MAKE_FLAGS           = -f $(MAKEFILE)
SHAREDLIB_LDFLAGS    = -V$(QCC_TARGET) -shared -Wl,--no-undefined -g



###########################################################################
## OUTPUT INFO
###########################################################################

PRODUCT = $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/Supine_SpeedGoat_v3
PRODUCT_TYPE = "executable"
BUILD_TYPE = "Top-Level Standalone Executable"

###########################################################################
## INCLUDE PATHS
###########################################################################

INCLUDES_BUILDINFO = -I$(START_DIR) -I$(MATLAB_ROOT)/extern/include/multimedia -I$(MATLAB_ROOT)/toolbox/shared/dsp/vision/matlab/include -I$(MATLAB_ROOT)/toolbox/dsp/halidesim/include -I$(MATLAB_ROOT)/toolbox/dsp/include -I$(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw -I$(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/dist/include -I$(MATLAB_ROOT)/toolbox/slrealtime/target/kernel/dist/include -I$(MATLAB_ROOT)/extern/include -I$(MATLAB_ROOT)/simulink/include -I$(MATLAB_ROOT)/rtw/c/src -I$(MATLAB_ROOT)/rtw/c/src/ext_mode/common -IC:/PROGRA~3/SPEEDG~1/SPEEDG~1/R2022a/940~1.3_2/SG_BLO~1/common/libsg -IC:/PROGRA~3/SPEEDG~1/SPEEDG~1/R2022a/940~1.3_2 -IC:/PROGRA~3/SPEEDG~1/SPEEDG~1/R2022a/940~1.3_2/SG_BLO~1/analog/include -I$(MATLAB_ROOT)/toolbox/dsp/extern/src/export/include/src -I$(MATLAB_ROOT)/toolbox/dsp/extern/src/export/include -I$(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/instrumented

INCLUDES = $(INCLUDES_BUILDINFO)

###########################################################################
## DEFINES
###########################################################################

DEFINES_ = -DSIMULINK_REAL_TIME
DEFINES_BUILD_ARGS = -DCLASSIC_INTERFACE=0 -DALLOCATIONFCN=0 -DEXT_MODE=1 -DMAT_FILE=0 -DONESTEPFCN=1 -DTERMFCN=1 -DMULTI_INSTANCE_CODE=0 -DINTEGER_CODE=0 -DMT=1
DEFINES_CUSTOM = 
DEFINES_OPTS = -DTID01EQ=1
DEFINES_STANDARD = -DMODEL=Supine_SpeedGoat_v3 -DNUMST=8 -DNCSTATES=16 -DHAVESTDIO -DRT -DUSE_RTMODEL

DEFINES = $(DEFINES_) $(DEFINES_BUILD_ARGS) $(DEFINES_CUSTOM) $(DEFINES_OPTS) $(DEFINES_STANDARD)

###########################################################################
## SOURCE FILES
###########################################################################

SRCS = $(MATLAB_ROOT)/toolbox/dsp/extern/src/dspeph/eph_zc_fcn_rt.c $(MATLAB_ROOT)/rtw/c/src/rt_matrx.c $(MATLAB_ROOT)/rtw/c/src/rt_printf.c $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/Supine_SpeedGoat_v3.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/Supine_SpeedGoat_v3_cal.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rtGetInf.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rtGetNaN.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rt_nonfinite.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rtmodel.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/slrealtime_datatype_ground.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rte_Supine_SpeedGoat_v3_parameters.cpp $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/main.cpp C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_ad_s.c C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_da_s.c C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_di_s.c C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_do_s.c C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_setup_s.c $(MATLAB_ROOT)/toolbox/coder/profile/src/host_timer_x86.c $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/instrumented/slrealtime_code_profiling_utility_functions.cpp

ALL_SRCS = $(SRCS)

###########################################################################
## OBJECTS
###########################################################################

OBJS = eph_zc_fcn_rt.o rt_matrx.o rt_printf.o Supine_SpeedGoat_v3.o Supine_SpeedGoat_v3_cal.o rtGetInf.o rtGetNaN.o rt_nonfinite.o rtmodel.o slrealtime_datatype_ground.o rte_Supine_SpeedGoat_v3_parameters.o main.o sg_IO191_ad_s.o sg_IO191_da_s.o sg_IO191_di_s.o sg_IO191_do_s.o sg_IO191_setup_s.o host_timer_x86.o slrealtime_code_profiling_utility_functions.o

ALL_OBJS = $(OBJS)

###########################################################################
## PREBUILT OBJECT FILES
###########################################################################

PREBUILT_OBJS = 

###########################################################################
## LIBRARIES
###########################################################################

LIBS = $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/dist/win64/lib/libecatinterface_slrt_x64.a $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/dist/win64/lib/libEcMaster_slrt_x64.a $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/dist/win64/lib/libslrealtime_libsrc_pervar_slrt_x64.a $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/dist/win64/lib/libslrealtime_libsrc_ip_slrt_x64.a

###########################################################################
## SYSTEM LIBRARIES
###########################################################################

SYSTEM_LIBS = 

###########################################################################
## ADDITIONAL TOOLCHAIN FLAGS
###########################################################################

#---------------
# C Compiler
#---------------

CFLAGS_BASIC = $(DEFINES) $(INCLUDES)

CFLAGS += $(CFLAGS_BASIC)

#-----------------
# C++ Compiler
#-----------------

CPPFLAGS_BASIC = $(DEFINES) $(INCLUDES)

CPPFLAGS += $(CPPFLAGS_BASIC)

#---------------
# C++ Linker
#---------------

CPP_LDFLAGS_ = -lsg_qnx710_x86_64 -LC:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/common/libsg
CPP_LDFLAGS_BLOCKMODULES = 

CPP_LDFLAGS += $(CPP_LDFLAGS_) $(CPP_LDFLAGS_BLOCKMODULES)

#------------------------------
# C++ Shared Library Linker
#------------------------------

CPP_SHAREDLIB_LDFLAGS_ = -lsg_qnx710_x86_64 -LC:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/common/libsg
CPP_SHAREDLIB_LDFLAGS_BLOCKMODULES = 

CPP_SHAREDLIB_LDFLAGS += $(CPP_SHAREDLIB_LDFLAGS_) $(CPP_SHAREDLIB_LDFLAGS_BLOCKMODULES)

#-----------
# Linker
#-----------

LDFLAGS_ = -lsg_qnx710_x86_64 -LC:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/common/libsg
LDFLAGS_BLOCKMODULES = 

LDFLAGS += $(LDFLAGS_) $(LDFLAGS_BLOCKMODULES)

#--------------------------
# Shared Library Linker
#--------------------------

SHAREDLIB_LDFLAGS_ = -lsg_qnx710_x86_64 -LC:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/common/libsg
SHAREDLIB_LDFLAGS_BLOCKMODULES = 

SHAREDLIB_LDFLAGS += $(SHAREDLIB_LDFLAGS_) $(SHAREDLIB_LDFLAGS_BLOCKMODULES)

###########################################################################
## INLINED COMMANDS
###########################################################################

###########################################################################
## PHONY TARGETS
###########################################################################

.PHONY : all build buildobj clean info prebuild


all : build
	@echo "### Successfully generated all binary outputs."


build : prebuild $(PRODUCT)


buildobj : prebuild $(OBJS) $(PREBUILT_OBJS) $(LIBS)
	@echo "### Successfully generated all binary outputs."


prebuild : 


###########################################################################
## FINAL TARGET
###########################################################################

#-------------------------------------------
# Create a standalone executable            
#-------------------------------------------

$(PRODUCT) : $(OBJS) $(PREBUILT_OBJS) $(LIBS)
	@echo "### Creating standalone executable "$(PRODUCT)" ..."
	$(CPP_LD) $(CPP_LDFLAGS) -o $(PRODUCT) $(OBJS) -Wl,--start-group $(LIBS) -Wl,--end-group $(SYSTEM_LIBS) $(TOOLCHAIN_LIBS)
	@echo "### Created: $(PRODUCT)"


###########################################################################
## INTERMEDIATE TARGETS
###########################################################################

#---------------------
# SOURCE-TO-OBJECT
#---------------------

%.o : $(RELATIVE_PATH_TO_ANCHOR)/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(RELATIVE_PATH_TO_ANCHOR)/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/src/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/src/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_interrupts/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_interrupts/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/IP/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/IP/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(START_DIR)/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(START_DIR)/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/dsp/extern/src/dspeph/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/dsp/extern/src/dspeph/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/rtw/c/src/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/rtw/c/src/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/simulink/src/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/simulink/src/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : ../%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : ../%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/coder/profile/src/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(MATLAB_ROOT)/toolbox/coder/profile/src/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


%.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/instrumented/%.c
	$(CC) $(CFLAGS) -o $@ $<


%.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/instrumented/%.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


eph_zc_fcn_rt.o : $(MATLAB_ROOT)/toolbox/dsp/extern/src/dspeph/eph_zc_fcn_rt.c
	$(CC) $(CFLAGS) -o $@ $<


rt_matrx.o : $(MATLAB_ROOT)/rtw/c/src/rt_matrx.c
	$(CC) $(CFLAGS) -o $@ $<


rt_printf.o : $(MATLAB_ROOT)/rtw/c/src/rt_printf.c
	$(CC) $(CFLAGS) -o $@ $<


Supine_SpeedGoat_v3.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/Supine_SpeedGoat_v3.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


Supine_SpeedGoat_v3_cal.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/Supine_SpeedGoat_v3_cal.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


rtGetInf.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rtGetInf.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


rtGetNaN.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rtGetNaN.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


rt_nonfinite.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rt_nonfinite.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


rtmodel.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rtmodel.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


slrealtime_datatype_ground.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/slrealtime_datatype_ground.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


rte_Supine_SpeedGoat_v3_parameters.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/rte_Supine_SpeedGoat_v3_parameters.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


main.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/main.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


sg_IO191_ad_s.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_ad_s.c
	$(CC) $(CFLAGS) -o $@ $<


sg_IO191_da_s.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_da_s.c
	$(CC) $(CFLAGS) -o $@ $<


sg_IO191_di_s.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_di_s.c
	$(CC) $(CFLAGS) -o $@ $<


sg_IO191_do_s.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_do_s.c
	$(CC) $(CFLAGS) -o $@ $<


sg_IO191_setup_s.o : C:/ProgramData/Speedgoat/speedgoatlib/R2022a/9.4.0.3_2/sg_blocks/analog/sg_IO191_setup_s.c
	$(CC) $(CFLAGS) -o $@ $<


host_timer_x86.o : $(MATLAB_ROOT)/toolbox/coder/profile/src/host_timer_x86.c
	$(CC) $(CFLAGS) -o $@ $<


slrealtime_code_profiling_utility_functions.o : $(START_DIR)/Supine_SpeedGoat_v3_slrealtime_rtw/instrumented/slrealtime_code_profiling_utility_functions.cpp
	$(CPP) $(CPPFLAGS) -o $@ $<


#------------------------
# BUILDABLE LIBRARIES
#------------------------

$(MATLAB_ROOT)/toolbox/slrealtime/simulink/blocks/dist/win64/lib/libslrealtime_libsrc_ip_slrt_x64.a : $(LIBLIBSLREALTIME_LIBSRC_IP_SLRT_X64_OBJS)
	@echo "### Creating static library $@ ..."
	$(AR) $(ARFLAGS)  $@ $(LIBLIBSLREALTIME_LIBSRC_IP_SLRT_X64_OBJS)


###########################################################################
## DEPENDENCIES
###########################################################################

$(ALL_OBJS) : rtw_proj.tmw $(MAKEFILE)


###########################################################################
## MISCELLANEOUS TARGETS
###########################################################################

info : 
	@echo "### PRODUCT = $(PRODUCT)"
	@echo "### PRODUCT_TYPE = $(PRODUCT_TYPE)"
	@echo "### BUILD_TYPE = $(BUILD_TYPE)"
	@echo "### INCLUDES = $(INCLUDES)"
	@echo "### DEFINES = $(DEFINES)"
	@echo "### ALL_SRCS = $(ALL_SRCS)"
	@echo "### ALL_OBJS = $(ALL_OBJS)"
	@echo "### LIBS = $(LIBS)"
	@echo "### MODELREF_LIBS = $(MODELREF_LIBS)"
	@echo "### SYSTEM_LIBS = $(SYSTEM_LIBS)"
	@echo "### TOOLCHAIN_LIBS = $(TOOLCHAIN_LIBS)"
	@echo "### CFLAGS = $(CFLAGS)"
	@echo "### LDFLAGS = $(LDFLAGS)"
	@echo "### SHAREDLIB_LDFLAGS = $(SHAREDLIB_LDFLAGS)"
	@echo "### CPPFLAGS = $(CPPFLAGS)"
	@echo "### CPP_LDFLAGS = $(CPP_LDFLAGS)"
	@echo "### CPP_SHAREDLIB_LDFLAGS = $(CPP_SHAREDLIB_LDFLAGS)"
	@echo "### ARFLAGS = $(ARFLAGS)"
	@echo "### MAKE_FLAGS = $(MAKE_FLAGS)"


clean : 
	$(ECHO) "### Deleting all derived files..."
	$(RM) $(subst /,\,$(PRODUCT))
	$(RM) $(subst /,\,$(ALL_OBJS))
	$(ECHO) "### Deleted all derived files."


