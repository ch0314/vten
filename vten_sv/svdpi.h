/* svdpi.h — Minimal DPI-C stub for standalone compilation.
 *
 * When building with xsim/xelab, the simulator provides its own svdpi.h.
 * This stub allows gcc standalone compilation for syntax/link checking.
 * The actual simulator header takes precedence at elaboration time.
 */

#ifndef SVDPI_H
#define SVDPI_H

#include <stdint.h>

/* Core types */
typedef void* svOpenArrayHandle;
typedef uint32_t svBitVecVal;

/* Get pointer to contiguous data of an open array */
static inline void* svGetArrayPtr(svOpenArrayHandle h) {
    return h;
}

/* Get size of dimension d (1-based) of an open array */
static inline int svSize(svOpenArrayHandle h, int d) {
    (void)h; (void)d;
    return 0;  /* Stub: actual size determined at runtime by simulator */
}

/* Get left/right index of dimension d */
static inline int svLeft(const svOpenArrayHandle h, int d) {
    (void)h; (void)d;
    return 0;
}

static inline int svRight(const svOpenArrayHandle h, int d) {
    (void)h; (void)d;
    return 0;
}

/* Get pointer to element at index in 1-D open array */
static inline void* svGetArrElemPtr1(const svOpenArrayHandle h, int indx1) {
    (void)h; (void)indx1;
    return NULL;  /* Stub: actual implementation by simulator */
}

/* Get pointer to element at (indx1, indx2) in 2-D open array */
static inline void* svGetArrElemPtr2(const svOpenArrayHandle h, int indx1, int indx2) {
    (void)h; (void)indx1; (void)indx2;
    return NULL;
}

/* Get pointer to element at (indx1, indx2, indx3) in 3-D open array */
static inline void* svGetArrElemPtr3(const svOpenArrayHandle h, int indx1, int indx2, int indx3) {
    (void)h; (void)indx1; (void)indx2; (void)indx3;
    return NULL;
}

#endif /* SVDPI_H */
