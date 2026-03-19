/* dpi_mock.c — Mock DPI-C bridge implementation for verilator testing.
 *
 * Heap-based SHM mock: Python packs SHM image → mock_load_shm_image() →
 * verilator SV calls DPI-C functions → reads/writes heap memory.
 */

#include "dpi_mock.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* svdpi.h provides svOpenArrayHandle, svLogicVecVal etc.
 * Verilator's svdpi.h is pure C-compatible. */
#include "svdpi.h"

/* svGetArrayPtr / svSize: provided by verilated_dpi but that header is C++.
 * Declare the subset we need as extern (defined in verilated_dpi.cpp). */
extern void* svGetArrayPtr(const svOpenArrayHandle h);
extern int svSize(const svOpenArrayHandle h, int d);

/* ── Internal state ── */
static uint8_t*         shm_mem       = NULL;
static int              shm_mem_size  = 0;
static ControlHeader*   ctrl          = NULL;
static uint8_t*         cmd_base      = NULL;
static uint8_t*         stats_base    = NULL;
static uint8_t*         bufdesc_base  = NULL;
static uint8_t*         data_base     = NULL;

static BufferDescriptor buf_cache[MAX_BUFFERS];
static int              buf_cache_valid = 0;

/* Mock counters */
static int              complete_count = 0;
static int              error_count    = 0;
static char             last_error_msg[256] = {0};
static int              wait_host_result = 0;  /* 0=OK, 1=TIMEOUT */

/* ── Internal helpers ── */

static void _setup_pointers(void) {
    if (shm_mem == NULL || shm_mem_size < CONTROL_SIZE) return;

    ctrl = (ControlHeader*)shm_mem;
    cmd_base     = shm_mem + ctrl->cmd_region_offset;
    stats_base   = shm_mem + ctrl->stats_region_offset;
    bufdesc_base = shm_mem + ctrl->buf_desc_offset;
    data_base    = shm_mem + ctrl->data_region_offset;
    buf_cache_valid = 0;
}

static void _load_buf_cache(void) {
    if (bufdesc_base == NULL || ctrl == NULL) return;
    int n = (int)ctrl->num_buffers;
    if (n > MAX_BUFFERS) n = MAX_BUFFERS;
    for (int i = 0; i < n; i++) {
        uint8_t* desc_ptr = bufdesc_base + i * BUF_DESC_SIZE;
        buf_cache[i].buffer_id   = *(uint16_t*)(desc_ptr + 0x00);
        buf_cache[i].direction   = *(uint8_t*)(desc_ptr + 0x02);
        buf_cache[i].flags       = *(uint8_t*)(desc_ptr + 0x03);
        buf_cache[i].size        = *(uint32_t*)(desc_ptr + 0x04);
        buf_cache[i].data_offset = *(uint64_t*)(desc_ptr + 0x08);
    }
    buf_cache_valid = 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Mock control functions (Python ctypes interface)
 * ═══════════════════════════════════════════════════════════════════════════ */

int mock_load_shm_image(const void* data, int size) {
    if (shm_mem != NULL) {
        free(shm_mem);
        shm_mem = NULL;
    }

    shm_mem = (uint8_t*)malloc(size);
    if (shm_mem == NULL) return -1;

    memcpy(shm_mem, data, size);
    shm_mem_size = size;

    _setup_pointers();

    /* Reset counters */
    complete_count = 0;
    error_count = 0;
    last_error_msg[0] = '\0';
    wait_host_result = 0;

    return 0;
}

void* mock_get_shm_ptr(int* out_size) {
    if (out_size) *out_size = shm_mem_size;
    return shm_mem;
}

void mock_reset(void) {
    if (shm_mem != NULL) {
        free(shm_mem);
        shm_mem = NULL;
    }
    shm_mem_size = 0;
    ctrl = NULL;
    cmd_base = NULL;
    stats_base = NULL;
    bufdesc_base = NULL;
    data_base = NULL;
    buf_cache_valid = 0;
    complete_count = 0;
    error_count = 0;
    last_error_msg[0] = '\0';
    wait_host_result = 0;
}

void mock_set_host_status(int status) {
    if (ctrl != NULL) {
        ctrl->host_status = (uint32_t)status;
    }
}

int mock_get_backend_status(void) {
    if (ctrl == NULL) return -1;
    return (int)ctrl->backend_status;
}

int mock_get_error_code(void) {
    if (ctrl == NULL) return -1;
    return (int)ctrl->error_code;
}

int mock_get_complete_count(void) {
    return complete_count;
}

int mock_get_error_count(void) {
    return error_count;
}

const char* mock_get_last_error_msg(void) {
    return last_error_msg;
}

void mock_set_wait_host_result(int result) {
    wait_host_result = result;
}

int mock_read_stats(int cmd_id, StatsEntry* out) {
    if (stats_base == NULL || out == NULL) return -1;
    memcpy(out, stats_base + cmd_id * STATS_SLOT_SIZE, sizeof(StatsEntry));
    return 0;
}

int mock_read_shm_bytes(int offset, void* out, int size) {
    if (shm_mem == NULL || out == NULL) return -1;
    if (offset + size > shm_mem_size) return -1;
    memcpy(out, shm_mem + offset, (size_t)size);
    return size;
}

int mock_get_session_seq(void) {
    if (ctrl == NULL) return -1;
    return (int)ctrl->session_seq;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DPI-C function implementations (called by verilator)
 * ═══════════════════════════════════════════════════════════════════════════ */

int vten_shm_init(const char* session_id) {
    (void)session_id;
    /* Mock: memory already loaded via mock_load_shm_image */
    if (shm_mem == NULL) return -1;

    _setup_pointers();

    if (ctrl->magic != SHM_MAGIC) return -1;
    if (ctrl->version != PROTOCOL_VERSION) return -1;

    ctrl->backend_status = BACKEND_IDLE;
    ctrl->session_seq++;

    return 0;  /* VTEN_OK */
}

void vten_cleanup(void) {
    /* Don't free memory — Python may inspect it after simulation */
}

int vten_wait_host_signal_safe(int timeout_ms) {
    (void)timeout_ms;
    return wait_host_result;
}

int vten_read_host_status(void) {
    if (ctrl == NULL) return -1;
    return (int)ctrl->host_status;
}

void vten_set_backend_status(int status) {
    if (ctrl == NULL) return;
    ctrl->backend_status = (uint32_t)status;
}

void vten_signal_complete(void) {
    if (ctrl == NULL) return;
    ctrl->backend_status = BACKEND_DONE;
    complete_count++;
}

void vten_signal_error(int code, const char* msg) {
    if (ctrl == NULL) return;
    ctrl->backend_status = BACKEND_ERROR;
    ctrl->error_code = (uint32_t)code;
    if (msg != NULL) {
        snprintf(last_error_msg, sizeof(last_error_msg), "%s", msg);
        snprintf(ctrl->error_message, 64, "%s", msg);
    }
    error_count++;
}

int vten_read_num_commands(void) {
    if (ctrl == NULL) return 0;
    return (int)ctrl->num_commands;
}

int vten_read_num_buffers(void) {
    if (ctrl == NULL) return 0;
    return (int)ctrl->num_buffers;
}

int vten_read_timeout_ms(void) {
    if (ctrl == NULL) return 10000;
    int t = (int)ctrl->timeout_ms;
    return (t == 0) ? 10000 : t;
}

int vten_read_flags(void) {
    if (ctrl == NULL) return 0;
    return (int)ctrl->flags;
}

int vten_read_command(int cmd_id,
    int* opcode, int* interface_id, int* protocol, int* role,
    int* buffer_id, int* probe, int* flags, int* size,
    long long* phys_addr,
    int* reg_offset, int* reg_value, int* reg_mask, int* reg_expected,
    int* golden_buf_id,
    int* num_deps, int* num_commit_deps,
    int* dep_ids, int* commit_dep_ids)
{
    if (cmd_base == NULL) return -1;

    uint8_t* slot = cmd_base + cmd_id * CMD_SLOT_SIZE;

    *opcode         = (int)(*(uint16_t*)(slot + 0x00));
    *interface_id   = (int)(*(uint16_t*)(slot + 0x04));
    *protocol       = (int)(*(uint8_t*)(slot + 0x06));
    *role           = (int)(*(uint8_t*)(slot + 0x07));
    *buffer_id      = (int)(*(uint16_t*)(slot + 0x08));
    *probe          = (int)(*(uint8_t*)(slot + 0x0A));
    *flags          = (int)(*(uint8_t*)(slot + 0x0B));
    *size           = (int)(*(uint32_t*)(slot + 0x0C));
    *phys_addr      = (long long)(*(uint64_t*)(slot + 0x10));
    *reg_offset     = (int)(*(uint32_t*)(slot + 0x18));
    *reg_value      = (int)(*(uint32_t*)(slot + 0x1C));
    *reg_mask       = (int)(*(uint32_t*)(slot + 0x20));
    *reg_expected   = (int)(*(uint32_t*)(slot + 0x24));
    *golden_buf_id  = (int)(*(uint16_t*)(slot + 0x28));
    *num_deps       = (int)(*(uint8_t*)(slot + 0x2A));
    *num_commit_deps = (int)(*(uint8_t*)(slot + 0x2B));

    uint16_t* dep_ptr  = (uint16_t*)(slot + 0x2C);
    uint16_t* cdep_ptr = (uint16_t*)(slot + 0x34);
    for (int i = 0; i < 4; i++) {
        dep_ids[i]        = (int)dep_ptr[i];
        commit_dep_ids[i] = (int)cdep_ptr[i];
    }

    return 0;
}

void vten_read_command_deps(int cmd_id,
    int* num_dep, svLogicVecVal* dep_ids,
    int* num_cdep, svLogicVecVal* cdep_ids)
{
    if (cmd_base == NULL) return;

    uint8_t* slot = cmd_base + cmd_id * CMD_SLOT_SIZE;

    *num_dep  = (int)(*(uint8_t*)(slot + 0x2A));
    *num_cdep = (int)(*(uint8_t*)(slot + 0x2B));

    uint16_t* deps  = (uint16_t*)(slot + 0x2C);
    uint16_t* cdeps = (uint16_t*)(slot + 0x34);

    /* svLogicVecVal: aval holds the value bits, bval=0 for known values */
    for (int i = 0; i < 4; i++) {
        dep_ids[i].aval  = (unsigned int)deps[i];
        dep_ids[i].bval  = 0;
        cdep_ids[i].aval = (unsigned int)cdeps[i];
        cdep_ids[i].bval = 0;
    }
}

void vten_read_data(int buf_id, int offset, int size, const svOpenArrayHandle dst) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();

    if (buf_id < 0 || buf_id >= MAX_BUFFERS) return;
    BufferDescriptor* desc = &buf_cache[buf_id];

    uint8_t* src = data_base + (int)desc->data_offset + offset;
    void* dst_ptr = svGetArrayPtr(dst);
    if (dst_ptr != NULL) {
        memcpy(dst_ptr, src, (size_t)size);
    }
}

void vten_write_data(int buf_id, int offset, int size, const svOpenArrayHandle src) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();

    if (buf_id < 0 || buf_id >= MAX_BUFFERS) return;
    BufferDescriptor* desc = &buf_cache[buf_id];

    uint8_t* dst = data_base + (int)desc->data_offset + offset;
    const void* src_ptr = svGetArrayPtr(src);
    if (src_ptr != NULL) {
        memcpy(dst, src_ptr, (size_t)size);
    }
}

void vten_write_cmd_stats(int cmd_id, int status,
    int issue_cycle, int commit_cycle,
    int first_active, int last_active,
    int active_cycles, int total_beats, int stall_cycles)
{
    if (stats_base == NULL) return;

    StatsEntry* entry = (StatsEntry*)(stats_base + cmd_id * STATS_SLOT_SIZE);
    entry->status             = (uint8_t)status;
    entry->reserved           = 0;
    entry->error_code         = 0;
    entry->issue_cycle        = (uint32_t)issue_cycle;
    entry->commit_cycle       = (uint32_t)commit_cycle;
    entry->first_active_cycle = (uint32_t)first_active;
    entry->last_active_cycle  = (uint32_t)last_active;
    entry->active_cycles      = (uint32_t)active_cycles;
    entry->total_beats        = (uint32_t)total_beats;
    entry->stall_cycles       = (uint32_t)stall_cycles;
}

void vten_write_cmd_status(int cmd_id, int status) {
    if (stats_base == NULL) return;

    StatsEntry* entry = (StatsEntry*)(stats_base + cmd_id * STATS_SLOT_SIZE);
    entry->status = (uint8_t)status;
}

void vten_read_golden(int buf_id, int beat_index, const svOpenArrayHandle dst) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();

    if (buf_id < 0 || buf_id >= MAX_BUFFERS) return;
    BufferDescriptor* desc = &buf_cache[buf_id];

    int bytes_per_beat = svSize(dst, 1);
    if (bytes_per_beat <= 0) bytes_per_beat = 32; /* fallback */
    int byte_offset = beat_index * bytes_per_beat;

    uint8_t* src = data_base + (int)desc->data_offset + byte_offset;
    void* dst_ptr = svGetArrayPtr(dst);
    if (dst_ptr != NULL) {
        memcpy(dst_ptr, src, (size_t)bytes_per_beat);
    }
}

void vten_log_mismatch(int cycle, int beat,
    int expected_hi, int expected_lo,
    int actual_hi, int actual_lo)
{
    fprintf(stderr, "[PROBE MISMATCH] cycle=%d beat=%d "
            "expected=0x%08X_%08X actual=0x%08X_%08X\n",
            cycle, beat, expected_hi, expected_lo, actual_hi, actual_lo);
}
