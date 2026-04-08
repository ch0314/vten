/* svdpi.h — Minimal DPI-C stub for standalone compilation.
 *
 * When building with xsim/xelab, the simulator provides its own svdpi.h.
 * This stub allows gcc standalone compilation for syntax/link checking.
 * The actual simulator header takes precedence at elaboration time.
 *
 * NOTE: Verilator compiles .c files as C++ (via CXX). The extern "C"
 * wrapper is required so these declarations match Verilator's
 * verilated_dpi.cpp symbols (which also use extern "C" linkage).
 */

#ifndef SVDPI_H
#define SVDPI_H

#include <stdint.h>

/* Core types */
typedef void* svOpenArrayHandle;
typedef uint32_t svBitVecVal;

/* DPI open-array functions.
 * Declared (not defined) here. The actual implementation is provided
 * by the simulator at link time:
 *   - Verilator: verilated_dpi.cpp
 *   - xsim: simulator runtime (loaded at elaboration) */
#ifdef __cplusplus
extern "C" {
#endif

extern void* svGetArrayPtr(const svOpenArrayHandle h);
extern int svSize(const svOpenArrayHandle h, int d);
extern int svSizeOfArray(const svOpenArrayHandle h);
extern int svLeft(const svOpenArrayHandle h, int d);
extern int svRight(const svOpenArrayHandle h, int d);
extern void* svGetArrElemPtr1(const svOpenArrayHandle h, int indx1);
extern void* svGetArrElemPtr2(const svOpenArrayHandle h, int indx1, int indx2);
extern void* svGetArrElemPtr3(const svOpenArrayHandle h, int indx1, int indx2, int indx3);

#ifdef __cplusplus
}
#endif

#endif /* SVDPI_H */
