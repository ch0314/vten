/* vten_shm_bridge.c — DPI-C bridge implementation for vTen SHM communication
 * Reference: docs/architecture.md
 * C99, POSIX API only.
 *
 * Compilation:
 *   gcc -shared -fPIC -o libvten_shm.so vten_shm_bridge.c -lrt -lpthread
 */

/* Enable POSIX extensions for clock_gettime, sem_timedwait, etc. */
#define _POSIX_C_SOURCE 200809L

#include "vten_shm_bridge.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* svdpi.h is provided by the simulator (xsim) */
#include "svdpi.h"

/* ── Internal state (static pointers) ── */
static void*            shm_base      = NULL;
static ControlHeader*   ctrl          = NULL;
static void*            cmd_base      = NULL;
static void*            stats_base    = NULL;
static void*            bufdesc_base  = NULL;
static void*            data_base     = NULL;
static sem_t*           sem_h2b       = NULL;
static sem_t*           sem_b2h       = NULL;
static size_t           shm_size      = 0;
static char             shm_name_buf[256] = {0};  /* saved for remap */

static BufferDescriptor buf_cache[MAX_BUFFERS];
static int              buf_cache_valid = 0;

/* Mismatch log file (JSONL format) */
static FILE*            mismatch_fp     = NULL;

/* ── Internal helpers ── */

static void _load_buf_cache(void) {
    if (bufdesc_base == NULL || ctrl == NULL) return;
    int n = (int)ctrl->num_buffers;
    if (n > MAX_BUFFERS) n = MAX_BUFFERS;
    /* Bulk copy buffer descriptor region into local cache via memcpy,
     * then parse individual fields for structured access. */
    uint8_t raw_cache[MAX_BUFFERS * BUF_DESC_SIZE];
    memcpy(raw_cache, bufdesc_base, n * BUF_DESC_SIZE);
    for (int i = 0; i < n; i++) {
        uint8_t* desc_ptr = raw_cache + i * BUF_DESC_SIZE;
        buf_cache[i].buffer_id   = *(uint16_t*)(desc_ptr + 0x00);
        buf_cache[i].direction   = *(uint8_t*)(desc_ptr + 0x02);
        buf_cache[i].flags       = *(uint8_t*)(desc_ptr + 0x03);
        buf_cache[i].size        = *(uint32_t*)(desc_ptr + 0x04);
        buf_cache[i].data_offset = *(uint64_t*)(desc_ptr + 0x08);
    }
    buf_cache_valid = 1;
    fprintf(stderr, "[vten_shm_bridge] buf_cache reloaded: %d buffers\n", n);
}

/* Drain stale semaphore counts (non-blocking) */
static void _drain_semaphore(sem_t* sem) {
    while (sem_trywait(sem) == 0) {
        /* discard stale count */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

int vten_shm_init(const char* session_id) {
    /* Build SHM name: /vten_{session_id} — also save for remap */
    char shm_name[256];
    snprintf(shm_name, sizeof(shm_name), "/vten_%s", session_id);
    snprintf(shm_name_buf, sizeof(shm_name_buf), "%s", shm_name);

    /* Open existing SHM (created by Python host) */
    int fd = shm_open(shm_name, O_RDWR, 0);
    if (fd < 0) {
        fprintf(stderr, "[vten_shm_bridge] shm_open(%s) failed: %s\n",
                shm_name, strerror(errno));
        return VTEN_ERROR;
    }

    /* Get SHM size from file */
    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "[vten_shm_bridge] fstat failed: %s\n", strerror(errno));
        close(fd);
        return VTEN_ERROR;
    }
    shm_size = (size_t)st.st_size;

    /* mmap */
    shm_base = mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (shm_base == MAP_FAILED) {
        fprintf(stderr, "[vten_shm_bridge] mmap failed: %s\n", strerror(errno));
        shm_base = NULL;
        return VTEN_ERROR;
    }

    /* Set up control header pointer */
    ctrl = (ControlHeader*)shm_base;

    /* Verify magic and version */
    if (ctrl->magic != SHM_MAGIC) {
        fprintf(stderr, "[vten_shm_bridge] magic mismatch: expected 0x%08X, got 0x%08X\n",
                SHM_MAGIC, ctrl->magic);
        return VTEN_ERROR;
    }
    if (ctrl->version != PROTOCOL_VERSION) {
        fprintf(stderr, "[vten_shm_bridge] version mismatch: expected 0x%08X, got 0x%08X\n",
                PROTOCOL_VERSION, ctrl->version);
        return VTEN_ERROR;
    }

    /* Compute region base pointers */
    cmd_base     = (uint8_t*)shm_base + ctrl->cmd_region_offset;
    stats_base   = (uint8_t*)shm_base + ctrl->stats_region_offset;
    bufdesc_base = (uint8_t*)shm_base + ctrl->buf_desc_offset;
    data_base    = (uint8_t*)shm_base + ctrl->data_region_offset;

    /* Invalidate buffer cache — will reload on first access */
    buf_cache_valid = 0;

    /* Open semaphores */
    char sem_h2b_name[256], sem_b2h_name[256];
    snprintf(sem_h2b_name, sizeof(sem_h2b_name), "/vten_%s_h2b", session_id);
    snprintf(sem_b2h_name, sizeof(sem_b2h_name), "/vten_%s_b2h", session_id);

    sem_h2b = sem_open(sem_h2b_name, 0);
    if (sem_h2b == SEM_FAILED) {
        fprintf(stderr, "[vten_shm_bridge] sem_open(%s) failed: %s\n",
                sem_h2b_name, strerror(errno));
        sem_h2b = NULL;
        return VTEN_ERROR;
    }

    sem_b2h = sem_open(sem_b2h_name, 0);
    if (sem_b2h == SEM_FAILED) {
        fprintf(stderr, "[vten_shm_bridge] sem_open(%s) failed: %s\n",
                sem_b2h_name, strerror(errno));
        sem_b2h = NULL;
        return VTEN_ERROR;
    }

    /* Drain stale semaphore counts */
    _drain_semaphore(sem_h2b);
    _drain_semaphore(sem_b2h);

    /* Increment session sequence */
    ctrl->session_seq++;

    /* Set backend_status = IDLE */
    ctrl->backend_status = BACKEND_IDLE;

    /* Signal host: backend is ready */
    sem_post(sem_b2h);

    /* Open mismatch log file if VTEN_MISMATCH_DIR is set */
    const char* mdir = getenv("VTEN_MISMATCH_DIR");
    if (mdir != NULL && mdir[0] != '\0') {
        char mpath[512];
        snprintf(mpath, sizeof(mpath), "%s/mismatches.jsonl", mdir);
        mismatch_fp = fopen(mpath, "w");
        if (mismatch_fp == NULL)
            fprintf(stderr, "[vten_shm_bridge] warning: cannot open %s: %s\n",
                    mpath, strerror(errno));
    }

    fprintf(stderr, "[vten_shm_bridge] init OK: shm=%s size=%zu cmds=%u bufs=%u\n",
            shm_name, shm_size, ctrl->num_commands, ctrl->num_buffers);

    return VTEN_OK;
}

int vten_shm_remap(void) {
    /* Check if host grew the SHM via ftruncate. If so, munmap + mmap at new size.
     * Called from S_LOAD_BATCH before reading new batch metadata. */
    if (shm_base == NULL || shm_name_buf[0] == '\0') return VTEN_ERROR;

    int fd = shm_open(shm_name_buf, O_RDWR, 0);
    if (fd < 0) {
        fprintf(stderr, "[vten_shm_bridge] remap: shm_open failed: %s\n", strerror(errno));
        return VTEN_ERROR;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "[vten_shm_bridge] remap: fstat failed: %s\n", strerror(errno));
        close(fd);
        return VTEN_ERROR;
    }

    size_t new_size = (size_t)st.st_size;

    /* Always invalidate buffer descriptor cache — host may have rewritten
     * buffer descriptors in-place for a new batch even without resizing. */
    buf_cache_valid = 0;

    /* Re-derive region pointers: even without resize the host may have
     * updated the control header (num_commands, num_buffers, offsets). */
    ctrl         = (ControlHeader*)shm_base;
    cmd_base     = (uint8_t*)shm_base + ctrl->cmd_region_offset;
    stats_base   = (uint8_t*)shm_base + ctrl->stats_region_offset;
    bufdesc_base = (uint8_t*)shm_base + ctrl->buf_desc_offset;
    data_base    = (uint8_t*)shm_base + ctrl->data_region_offset;

    fprintf(stderr, "[vten_shm_bridge] remap called: cur=%zu new=%zu cmds=%u bufs=%u\n",
            shm_size, new_size, ctrl->num_commands, ctrl->num_buffers);

    if (new_size == shm_size) {
        /* No resize needed — pointers and cache already refreshed above */
        close(fd);
        return VTEN_OK;
    }

    fprintf(stderr, "[vten_shm_bridge] remap: %zu → %zu bytes\n", shm_size, new_size);

    /* munmap old mapping */
    munmap(shm_base, shm_size);

    /* mmap at new size */
    shm_base = mmap(NULL, new_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (shm_base == MAP_FAILED) {
        fprintf(stderr, "[vten_shm_bridge] remap: mmap failed: %s\n", strerror(errno));
        shm_base = NULL;
        ctrl = NULL;
        return VTEN_ERROR;
    }

    shm_size = new_size;

    /* Re-derive all region pointers from new mapping */
    ctrl         = (ControlHeader*)shm_base;
    cmd_base     = (uint8_t*)shm_base + ctrl->cmd_region_offset;
    stats_base   = (uint8_t*)shm_base + ctrl->stats_region_offset;
    bufdesc_base = (uint8_t*)shm_base + ctrl->buf_desc_offset;
    data_base    = (uint8_t*)shm_base + ctrl->data_region_offset;

    /* Invalidate buffer cache */
    buf_cache_valid = 0;

    return VTEN_OK;
}

void vten_cleanup(void) {
    if (mismatch_fp != NULL) { fclose(mismatch_fp); mismatch_fp = NULL; }
    if (sem_h2b != NULL) { sem_close(sem_h2b); sem_h2b = NULL; }
    if (sem_b2h != NULL) { sem_close(sem_b2h); sem_b2h = NULL; }
    if (shm_base != NULL && shm_base != MAP_FAILED) {
        munmap(shm_base, shm_size);
        shm_base = NULL;
    }
    ctrl         = NULL;
    cmd_base     = NULL;
    stats_base   = NULL;
    bufdesc_base = NULL;
    data_base    = NULL;
    buf_cache_valid = 0;
    fprintf(stderr, "[vten_shm_bridge] cleanup done\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Host/Backend Synchronization
 * ═══════════════════════════════════════════════════════════════════════════ */

int vten_wait_host_signal_safe(int timeout_ms) {
    if (sem_h2b == NULL) return VTEN_ERROR;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec  += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec++;
        ts.tv_nsec -= 1000000000L;
    }

    int ret = sem_timedwait(sem_h2b, &ts);
    if (ret == -1) {
        if (errno == ETIMEDOUT) return VTEN_TIMEOUT;
        return VTEN_ERROR;
    }
    return VTEN_OK;
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
    if (sem_b2h != NULL) sem_post(sem_b2h);
}

void vten_signal_error(int code, const char* msg) {
    if (ctrl == NULL) return;
    ctrl->backend_status = BACKEND_ERROR;
    ctrl->error_code = (uint32_t)code;            /* offset 0x40 */
    if (msg != NULL) {
        snprintf(ctrl->error_message, MAX_ERROR_MSG_LEN, "%s", msg);  /* offset 0x48 */
    }
    if (sem_b2h != NULL) sem_post(sem_b2h);
}

void vten_signal_error_with_cmd(int code, int cmd_id, const char* msg) {
    if (ctrl == NULL) return;
    ctrl->backend_status = BACKEND_ERROR;
    ctrl->error_code = (uint32_t)code;            /* offset 0x40 */
    ctrl->error_cmd_id = (uint32_t)cmd_id;        /* offset 0x44 */
    if (msg != NULL) {
        snprintf(ctrl->error_message, MAX_ERROR_MSG_LEN, "%s", msg);  /* offset 0x48 */
    }
    if (sem_b2h != NULL) sem_post(sem_b2h);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Control Header Reads
 *
 * Control Header layout (00_data_models.md §11.3):
 *   0x00 magic, 0x04 version, 0x08 host_status, 0x0C backend_status,
 *   0x10 num_commands, 0x14 num_buffers, 0x18 cmd_region_offset,
 *   0x20 stats_region_offset, 0x28 buf_desc_offset, 0x30 data_region_offset,
 *   0x38 total_shm_size, 0x40 error_code, 0x44 error_cmd_id,
 *   0x48 error_message[64], 0x88 flags, 0x8C timeout_ms,
 *   0x90 sim_frequency_hz, 0x94 session_seq, 0x98 reserved[104]
 * ═══════════════════════════════════════════════════════════════════════════ */

int vten_read_num_commands(void) {
    if (ctrl == NULL) return 0;
    return (int)ctrl->num_commands;  /* offset 0x10 */
}

int vten_read_num_buffers(void) {
    if (ctrl == NULL) return 0;
    return (int)ctrl->num_buffers;  /* offset 0x14 */
}

int vten_read_timeout_ms(void) {
    if (ctrl == NULL) return 10000;  /* default 10s */
    int t = (int)ctrl->timeout_ms;  /* offset 0x8C */
    return (t == 0) ? 10000 : t;
}

int vten_read_flags(void) {
    if (ctrl == NULL) return 0;
    return (int)ctrl->flags;  /* offset 0x88 */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Command Region
 * ═══════════════════════════════════════════════════════════════════════════ */

int vten_read_command(int cmd_id,
    int* opcode, int* interface_id, int* protocol, int* role,
    int* buffer_id, int* probe, int* flags, int* size,
    long long* phys_addr,
    int* reg_offset, int* reg_value, int* reg_mask, int* reg_expected,
    int* golden_buf_id,
    int* num_deps, int* num_commit_deps,
    int dep_ids[4], int commit_dep_ids[4])
{
    if (cmd_base == NULL) return VTEN_ERROR;

    uint8_t* slot = (uint8_t*)cmd_base + cmd_id * CMD_SLOT_SIZE;

    *opcode         = (int)(*(uint16_t*)(slot + 0x00));
    /* cmd_id at 0x02 is implicit (= cmd_id arg) */
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

    return VTEN_OK;
}

void vten_read_command_deps(int cmd_id,
    int* num_dep, int* dep_ids,
    int* num_cdep, int* cdep_ids)
{
    if (cmd_base == NULL) return;

    uint8_t* slot = (uint8_t*)cmd_base + cmd_id * CMD_SLOT_SIZE;

    *num_dep  = (int)(*(uint8_t*)(slot + 0x2A));
    *num_cdep = (int)(*(uint8_t*)(slot + 0x2B));

    uint16_t* deps  = (uint16_t*)(slot + 0x2C);
    uint16_t* cdeps = (uint16_t*)(slot + 0x34);

    /* Fixed-size array 'output int arr[0:3]' is passed as int* by both
     * Verilator and xsim.  Do NOT use svGetArrayPtr — that's for open arrays. */
    if (dep_ids != NULL) {
        for (int i = 0; i < 4; i++)
            dep_ids[i] = (int)deps[i];
    }
    if (cdep_ids != NULL) {
        for (int i = 0; i < 4; i++)
            cdep_ids[i] = (int)cdeps[i];
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Data Region
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Bulk transfer via byte[] open array + memcpy ──
 * SV side declares: byte arr[0:size-1]
 * svGetArrayPtr(byte[]) returns contiguous uint8_t* on both Verilator and xsim.
 * This is the fast path — one memcpy per beat instead of per-byte DPI calls. */

void vten_read_data_bulk(int buf_id, int offset, int size, void* dst_handle) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();
    BufferDescriptor* desc = &buf_cache[buf_id];
    uint8_t* src = (uint8_t*)data_base + desc->data_offset + offset;
    uint8_t* dst = (uint8_t*)svGetArrayPtr((svOpenArrayHandle)dst_handle);
    if (dst != NULL) {
        memcpy(dst, src, (size_t)size);
    }
}

void vten_write_data_bulk(int buf_id, int offset, int size, const void* src_handle) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();
    BufferDescriptor* desc = &buf_cache[buf_id];
    uint8_t* dst = (uint8_t*)data_base + desc->data_offset + offset;
    const uint8_t* src = (const uint8_t*)svGetArrayPtr((svOpenArrayHandle)src_handle);
    if (src != NULL) {
        memcpy(dst, src, (size_t)size);
    }
}

void vten_read_golden_bulk(int buf_id, int offset, int size, void* dst_handle) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();
    BufferDescriptor* desc = &buf_cache[buf_id];
    if (offset + size > (int)desc->size) {
        fprintf(stderr, "[vten_shm_bridge] vten_read_golden_bulk: out of bounds "
                "buf_id=%d offset=%d size=%d > buf_size=%u\n",
                buf_id, offset, size, desc->size);
        return;
    }
    uint8_t* src = (uint8_t*)data_base + desc->data_offset + offset;
    uint8_t* dst = (uint8_t*)svGetArrayPtr((svOpenArrayHandle)dst_handle);
    if (dst != NULL) {
        memcpy(dst, src, (size_t)size);
    }
}

/* Scalar byte write — used by AXI4 BFM partial WSTRB slow path. */
void vten_write_data_byte(int buf_id, int offset, int value) {
    if (data_base == NULL) return;
    if (!buf_cache_valid) _load_buf_cache();
    BufferDescriptor* desc = &buf_cache[buf_id];
    uint8_t* dst = (uint8_t*)data_base + desc->data_offset + offset;
    *dst = (uint8_t)value;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Stats Region
 * ═══════════════════════════════════════════════════════════════════════════ */

void vten_write_cmd_stats(int cmd_id, int status,
    int issue_cycle, int commit_cycle,
    int first_active, int last_active,
    int active_cycles, int total_beats, int stall_cycles)
{
    if (stats_base == NULL) return;

    StatsEntry* entry = (StatsEntry*)((uint8_t*)stats_base + cmd_id * STATS_SLOT_SIZE);
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

    StatsEntry* entry = (StatsEntry*)((uint8_t*)stats_base + cmd_id * STATS_SLOT_SIZE);
    entry->status = (uint8_t)status;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Probe
 * ═══════════════════════════════════════════════════════════════════════════ */

void vten_log_mismatch(int cmd_id, int cycle, int beat,
    int expected_hi, int expected_lo,
    int actual_hi, int actual_lo)
{
    fprintf(stderr, "[PROBE MISMATCH] cmd=%d cycle=%d beat=%d "
            "expected=0x%08X_%08X actual=0x%08X_%08X\n",
            cmd_id, cycle, beat, expected_hi, expected_lo, actual_hi, actual_lo);

    if (mismatch_fp != NULL) {
        fprintf(mismatch_fp,
            "{\"cmd_id\":%d,\"cycle\":%d,\"beat\":%d,"
            "\"expected_hi\":\"0x%08X\",\"expected_lo\":\"0x%08X\","
            "\"actual_hi\":\"0x%08X\",\"actual_lo\":\"0x%08X\"}\n",
            cmd_id, cycle, beat, expected_hi, expected_lo, actual_hi, actual_lo);
        fflush(mismatch_fp);
    }
}
