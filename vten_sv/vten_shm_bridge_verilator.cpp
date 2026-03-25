/**
 * vten_shm_bridge_verilator.cpp — DPI-C wrapper for Verilator
 *
 * Verilator uses verilated_dpi.h instead of standard svdpi.h.
 * This file wraps the existing C functions from vten_shm_bridge.c
 * so they are visible to Verilator's DPI system.
 *
 * Spec reference: 08_backend_abstraction.md §7.4
 */

#include "verilated_dpi.h"

// Link against existing C implementation
extern "C" {
#include "vten_shm_bridge.h"
}

// No additional wrappers needed — Verilator resolves DPI imports
// directly via the extern "C" declarations in vten_shm_bridge.h.
// This file ensures verilated_dpi.h is included in the build,
// providing Verilator's DPI infrastructure (svBit, svLogic, etc.).
//
// The C functions in vten_shm_bridge.c are compiled separately
// and linked into the Verilator binary. The DPI import declarations
// in vten_dpi_imports.svh map directly to those C functions.
