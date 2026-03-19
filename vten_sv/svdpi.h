/* svdpi.h — Minimal DPI-C stub for standalone compilation.
 *
 * When building with xsim/xelab, the simulator provides its own svdpi.h.
 * This stub allows gcc standalone compilation for syntax/link checking.
 * The actual simulator header takes precedence at elaboration time.
 */

#ifndef SVDPI_H
#define SVDPI_H

#include <stdint.h>

/* svOpenArrayHandle — opaque pointer to SV open array */
typedef void* svOpenArrayHandle;

/* Get pointer to contiguous data of an open array */
static inline void* svGetArrayPtr(svOpenArrayHandle h) {
    return h;
}

/* Get size of dimension d (1-based) of an open array */
static inline int svSize(svOpenArrayHandle h, int d) {
    (void)h; (void)d;
    return 0;  /* Stub: actual size determined at runtime by simulator */
}

#endif /* SVDPI_H */
