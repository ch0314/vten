/* dpi_mock.h — Mock DPI-C bridge for verilator-based testing.
 *
 * Replaces POSIX SHM with heap-allocated memory.
 * Python loads SHM image via mock_load_shm_image(), verilator reads via DPI-C.
 */

#ifndef DPI_MOCK_H
#define DPI_MOCK_H

#include <stdint.h>

/* ── SHM Constants (must match vten_shm_bridge.h) ── */
#define SHM_MAGIC           0x5654454E
#define PROTOCOL_VERSION    0x00000003
#define CONTROL_SIZE        256
#define CMD_SLOT_SIZE       64
#define STATS_SLOT_SIZE     32
#define BUF_DESC_SIZE       24
#define CACHE_LINE          64
#define MAX_BUFFERS         256

/* Host/Backend status */
#define HOST_IDLE       0
#define HOST_CMD_READY  1
#define HOST_ACK        2
#define HOST_SHUTDOWN   3

#define BACKEND_IDLE    0
#define BACKEND_RUNNING 1
#define BACKEND_DONE    2
#define BACKEND_ERROR   3

/* Error codes */
#define ERR_OK               0
#define ERR_POLL_TIMEOUT     2

/* ── Control Header (256 bytes) ── */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t host_status;
    uint32_t backend_status;
    uint32_t num_commands;
    uint32_t num_buffers;
    uint64_t cmd_region_offset;
    uint64_t stats_region_offset;
    uint64_t buf_desc_offset;
    uint64_t data_region_offset;
    uint64_t total_shm_size;
    uint32_t error_code;
    uint32_t error_cmd_id;
    char     error_message[64];
    uint32_t flags;
    uint32_t timeout_ms;
    uint32_t sim_frequency_hz;
    uint32_t session_seq;
    uint8_t  reserved[104];
} ControlHeader;

/* ── Buffer Descriptor (24 bytes) ── */
typedef struct {
    uint16_t buffer_id;
    uint8_t  direction;
    uint8_t  flags;
    uint32_t size;
    uint64_t data_offset;
    uint64_t reserved;
} BufferDescriptor;

/* ── Stats Entry (32 bytes) ── */
typedef struct {
    uint8_t  status;
    uint8_t  reserved;
    uint16_t error_code;
    uint32_t issue_cycle;
    uint32_t commit_cycle;
    uint32_t first_active_cycle;
    uint32_t last_active_cycle;
    uint32_t active_cycles;
    uint32_t total_beats;
    uint32_t stall_cycles;
} StatsEntry;

#ifdef __cplusplus
extern "C" {
#endif

/* ── Mock control functions (called from Python via ctypes) ── */

/* Load a complete SHM image into mock memory.
 * Returns 0 on success, -1 on error. */
int mock_load_shm_image(const void* data, int size);

/* Get pointer to current mock SHM memory. Returns size via out param. */
void* mock_get_shm_ptr(int* out_size);

/* Reset mock state (free memory, clear all pointers). */
void mock_reset(void);

/* Set host_status field directly (for test control). */
void mock_set_host_status(int status);

/* Get backend_status field. */
int mock_get_backend_status(void);

/* Get error_code field. */
int mock_get_error_code(void);

/* Check if signal_complete was called. Returns count. */
int mock_get_complete_count(void);

/* Check if signal_error was called. Returns count. */
int mock_get_error_count(void);

/* Get last error message from signal_error. */
const char* mock_get_last_error_msg(void);

/* Get wait_host return value (default: 0=OK). Set to control FSM. */
void mock_set_wait_host_result(int result);

#ifdef __cplusplus
}
#endif

#endif /* DPI_MOCK_H */
