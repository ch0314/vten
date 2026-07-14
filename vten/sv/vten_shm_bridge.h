/* vten_shm_bridge.h — DPI-C bridge header for vTen SHM communication
 * Reference: docs/architecture.md
 * C99, POSIX API only.
 */

#ifndef VTEN_SHM_BRIDGE_H
#define VTEN_SHM_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Return codes ── */
#define VTEN_OK       0
#define VTEN_TIMEOUT  1
#define VTEN_ERROR   -1

/* ── SHM Constants (00_data_models.md §11) ── */
#define SHM_MAGIC           0x5654454E  /* "VTEN" (LE) */
#define PROTOCOL_VERSION    0x00000003  /* v0.4 protocol */
#define CONTROL_SIZE        256
#define CMD_SLOT_SIZE       64
#define STATS_SLOT_SIZE     32
#define BUF_DESC_SIZE       24
#define CACHE_LINE          64

/* ── Max limits ── */
#define MAX_BUFFERS         256
#define MAX_ERROR_MSG_LEN   64

/* ── host_status values (§11.4) ── */
#define HOST_IDLE       0
#define HOST_CMD_READY  1
#define HOST_ACK        2
#define HOST_SHUTDOWN   3

/* ── backend_status values (§11.5) ── */
#define BACKEND_IDLE    0
#define BACKEND_RUNNING 1
#define BACKEND_DONE    2
#define BACKEND_ERROR   3

/* ── Backend Error Codes (§11.13) ── */
#define ERR_OK               0
#define ERR_ADDR_UNMATCH     1
#define ERR_POLL_TIMEOUT     2
#define ERR_BFM_QUEUE_ERROR  3
#define ERR_SCHEDULER_ERROR  4
#define ERR_SHM_ACCESS_ERROR 5
#define ERR_UNKNOWN_OPCODE   6
#define ERR_BFM_MAP_ERROR    7
#define ERR_PROBE_MISMATCH   8
#define ERR_TIMEOUT_CODE     9

/* ── Control Header layout (§11.3) ── */
typedef struct {
    uint32_t magic;                /* 0x00 */
    uint32_t version;              /* 0x04 */
    uint32_t host_status;          /* 0x08 */
    uint32_t backend_status;       /* 0x0C */
    uint32_t num_commands;         /* 0x10 */
    uint32_t num_buffers;          /* 0x14 */
    uint64_t cmd_region_offset;    /* 0x18 */
    uint64_t stats_region_offset;  /* 0x20 */
    uint64_t buf_desc_offset;      /* 0x28 */
    uint64_t data_region_offset;   /* 0x30 */
    uint64_t total_shm_size;       /* 0x38 */
    uint32_t error_code;           /* 0x40 */
    uint32_t error_cmd_id;         /* 0x44 */
    char     error_message[64];    /* 0x48 */
    uint32_t flags;                /* 0x88 */
    uint32_t timeout_ms;           /* 0x8C */
    uint32_t sim_frequency_hz;     /* 0x90 */
    uint32_t session_seq;          /* 0x94 */
    uint8_t  reserved[104];        /* 0x98 */
} ControlHeader;

/* ── Buffer Descriptor layout (§11.8) ── */
typedef struct {
    uint16_t buffer_id;     /* 0x00 */
    uint8_t  direction;     /* 0x02 */
    uint8_t  flags;         /* 0x03 */
    uint32_t size;          /* 0x04 */
    uint64_t data_offset;   /* 0x08 */
    uint64_t reserved;      /* 0x10 */
} BufferDescriptor;

/* ── Stats Entry layout (§11.9) ── */
typedef struct {
    uint8_t  status;              /* 0x00 */
    uint8_t  reserved;            /* 0x01 */
    uint16_t error_code;          /* 0x02 */
    uint32_t issue_cycle;         /* 0x04 */
    uint32_t commit_cycle;        /* 0x08 */
    uint32_t first_active_cycle;  /* 0x0C */
    uint32_t last_active_cycle;   /* 0x10 */
    uint32_t active_cycles;       /* 0x14 */
    uint32_t total_beats;         /* 0x18 */
    uint32_t stall_cycles;        /* 0x1C */
} StatsEntry;

/* ── DPI-C function prototypes ── */

/* Lifecycle */
int  vten_shm_init(const char* session_id);
int  vten_shm_remap(void);  /* Re-mmap if SHM was resized by host (ftruncate) */
void vten_cleanup(void);

/* Host/Backend synchronization */
int  vten_wait_host_signal_safe(int timeout_ms);
int  vten_read_host_status(void);
void vten_set_backend_status(int status);
void vten_signal_complete(void);
void vten_signal_error(int code, const char* msg);
void vten_signal_error_with_cmd(int code, int cmd_id, const char* msg);

/* Control header reads */
int  vten_read_num_commands(void);
int  vten_read_num_buffers(void);
int  vten_read_timeout_ms(void);
int  vten_read_flags(void);

/* Command region */
int  vten_read_command(int cmd_id,
    int* opcode, int* interface_id, int* protocol, int* role,
    int* buffer_id, int* probe, int* flags, int* size,
    long long* phys_addr,
    int* reg_offset, int* reg_value, int* reg_mask, int* reg_expected,
    int* golden_buf_id,
    int* num_deps, int* num_commit_deps,
    int dep_ids[4], int commit_dep_ids[4]);

void vten_read_command_deps(int cmd_id,
    int* num_dep, int* dep_ids,
    int* num_cdep, int* cdep_ids);

/* Data region — bulk transfer (byte[] + memcpy, cross-simulator portable) */
void vten_read_data_bulk(int buf_id, int offset, int size, void* dst);
void vten_write_data_bulk(int buf_id, int offset, int size, const void* src);
void vten_read_golden_bulk(int buf_id, int offset, int size, void* dst);

/* Data region — scalar byte write (AXI4 partial WSTRB path) */
void vten_write_data_byte(int buf_id, int offset, int value);

/* Stats region */
void vten_write_cmd_stats(int cmd_id, int status,
    int issue_cycle, int commit_cycle,
    int first_active, int last_active,
    int active_cycles, int total_beats, int stall_cycles);
void vten_write_cmd_status(int cmd_id, int status);

/* Probe */
void vten_log_mismatch(int cmd_id, int cycle, int beat,
    int expected_hi, int expected_lo,
    int actual_hi, int actual_lo);

#ifdef __cplusplus
}
#endif

#endif /* VTEN_SHM_BRIDGE_H */
