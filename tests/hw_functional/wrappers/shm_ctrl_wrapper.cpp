/* shm_ctrl_wrapper.cpp — extern "C" wrapper for verilator-compiled
 * vten_shm_controller, exposing simulation control to Python ctypes.
 */

#include "Vvten_shm_controller.h"
#include "Vvten_shm_controller___024root.h"
#include "verilated.h"

/* Include mock header for mock_* functions */
extern "C" {
#include "dpi_mock.h"
}

/* ── FSM state constants (from vten_types.svh) ── */
#define S_INIT          0
#define S_WAIT_HOST     1
#define S_LOAD_BATCH    2
#define S_FEED          3
#define S_EXECUTE       4
#define S_DRAIN         5
#define S_COMPLETE      6
#define S_ERROR         7
#define S_SHUTDOWN      8

/* ── Simulation context ── */
typedef struct {
    VerilatedContext* ctx;
    Vvten_shm_controller* dut;
    uint64_t sim_time;
} SimContext;

extern "C" {

/* ── Lifecycle ── */

void* shm_ctrl_create(void) {
    SimContext* sc = new SimContext;
    sc->ctx = new VerilatedContext;
    sc->ctx->debug(0);
    sc->ctx->randReset(2);
    sc->dut = new Vvten_shm_controller(sc->ctx, "TOP");
    sc->sim_time = 0;

    /* Initialize inputs */
    sc->dut->clk = 0;
    sc->dut->rst_n = 0;
    sc->dut->feed_ready = 0;
    sc->dut->sched_all_committed = 0;
    sc->dut->sched_all_drained = 0;
    sc->dut->sched_error = 0;
    sc->dut->sched_error_cmd_id = 0;
    sc->dut->sched_error_code = 0;

    return sc;
}

void shm_ctrl_destroy(void* handle) {
    SimContext* sc = (SimContext*)handle;
    if (sc == NULL) return;
    sc->dut->final();
    delete sc->dut;
    delete sc->ctx;
    delete sc;
}

/* ── Clock and reset ── */

void shm_ctrl_tick(void* handle) {
    SimContext* sc = (SimContext*)handle;

    /* Rising edge */
    sc->dut->clk = 1;
    sc->sim_time++;
    sc->ctx->timeInc(1);
    sc->dut->eval();

    /* Falling edge */
    sc->dut->clk = 0;
    sc->sim_time++;
    sc->ctx->timeInc(1);
    sc->dut->eval();
}

void shm_ctrl_reset(void* handle, int cycles) {
    SimContext* sc = (SimContext*)handle;
    sc->dut->rst_n = 0;
    for (int i = 0; i < cycles; i++) {
        shm_ctrl_tick(handle);
    }
    sc->dut->rst_n = 1;
    /* One tick after reset release */
    shm_ctrl_tick(handle);
}

/* Run N clock ticks */
void shm_ctrl_run(void* handle, int n) {
    for (int i = 0; i < n; i++) {
        shm_ctrl_tick(handle);
    }
}

/* ── State access ── */

int shm_ctrl_get_state(void* handle) {
    SimContext* sc = (SimContext*)handle;
    return (int)sc->dut->rootp->vten_shm_controller__DOT__state;
}

int shm_ctrl_get_num_commands(void* handle) {
    SimContext* sc = (SimContext*)handle;
    return (int)sc->dut->rootp->vten_shm_controller__DOT__num_commands;
}

int shm_ctrl_get_feed_idx(void* handle) {
    SimContext* sc = (SimContext*)handle;
    return (int)sc->dut->rootp->vten_shm_controller__DOT__feed_idx;
}

/* ── Port access ── */

int shm_ctrl_get_feed_valid(void* handle) {
    SimContext* sc = (SimContext*)handle;
    return (int)sc->dut->feed_valid;
}

int shm_ctrl_get_feed_done(void* handle) {
    SimContext* sc = (SimContext*)handle;
    return (int)sc->dut->feed_done;
}

/* Get a 32-bit word from feed_data (which is 303 bits = 10 words).
 * word_idx: 0-9 (word 0 = bits [31:0]) */
unsigned int shm_ctrl_get_feed_data_word(void* handle, int word_idx) {
    SimContext* sc = (SimContext*)handle;
    if (word_idx < 0 || word_idx >= 10) return 0;
    return sc->dut->feed_data[word_idx];
}

void shm_ctrl_set_feed_ready(void* handle, int val) {
    SimContext* sc = (SimContext*)handle;
    sc->dut->feed_ready = val & 1;
}

void shm_ctrl_set_sched_all_committed(void* handle, int val) {
    SimContext* sc = (SimContext*)handle;
    sc->dut->sched_all_committed = val & 1;
}

void shm_ctrl_set_sched_all_drained(void* handle, int val) {
    SimContext* sc = (SimContext*)handle;
    sc->dut->sched_all_drained = val & 1;
}

void shm_ctrl_set_sched_error(void* handle, int val) {
    SimContext* sc = (SimContext*)handle;
    sc->dut->sched_error = val & 1;
}

void shm_ctrl_set_sched_error_cmd_id(void* handle, int val) {
    SimContext* sc = (SimContext*)handle;
    sc->dut->sched_error_cmd_id = val & 0xFFFF;
}

void shm_ctrl_set_sched_error_code(void* handle, int val) {
    SimContext* sc = (SimContext*)handle;
    sc->dut->sched_error_code = val & 0xFFFF;
}

uint64_t shm_ctrl_get_sim_time(void* handle) {
    SimContext* sc = (SimContext*)handle;
    return sc->sim_time;
}

} /* extern "C" */
