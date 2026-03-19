/* shm_ctrl_driver.cpp — Subprocess-based test driver for vten_shm_controller.
 *
 * Protocol: reads JSON commands from stdin, writes JSON responses to stdout.
 * Python pytest drives the simulation via subprocess.Popen.
 *
 * Commands:
 *   {"cmd":"load","file":"<path>"}         Load SHM image from file
 *   {"cmd":"create"}                        Create simulator instance
 *   {"cmd":"reset","cycles":N}              Reset for N cycles
 *   {"cmd":"tick","n":N}                    Run N clock ticks
 *   {"cmd":"get_state"}                     Get FSM state
 *   {"cmd":"get_feed"}                      Get feed_valid, feed_done, feed_data
 *   {"cmd":"set","signal":"<name>","value":V}  Set input signal
 *   {"cmd":"mock_set","field":"<name>","value":V}  Set mock field
 *   {"cmd":"mock_get","field":"<name>"}     Get mock field
 *   {"cmd":"get_internals"}                 Get internal state (num_commands, feed_idx)
 *   {"cmd":"destroy"}                       Destroy simulator
 *   {"cmd":"quit"}                          Exit process
 *
 * Responses: one JSON line per command.
 */

#include "Vvten_shm_controller.h"
#include "Vvten_shm_controller___024root.h"
#include "verilated.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "dpi_mock.h"
}

/* ── FSM state names ── */
static const char* STATE_NAMES[] = {
    "S_INIT", "S_WAIT_HOST", "S_LOAD_BATCH", "S_FEED",
    "S_EXECUTE", "S_DRAIN", "S_COMPLETE", "S_ERROR", "S_SHUTDOWN"
};

/* ── Simulation context ── */
static VerilatedContext* ctx = nullptr;
static Vvten_shm_controller* dut = nullptr;
static uint64_t sim_time = 0;

static void do_tick() {
    dut->clk = 1;
    sim_time++;
    ctx->timeInc(1);
    dut->eval();
    dut->clk = 0;
    sim_time++;
    ctx->timeInc(1);
    dut->eval();
}

/* ── Minimal JSON parser (no external deps) ── */

static char line_buf[65536];

/* Extract string value for key from JSON line. Returns static buffer. */
static char val_buf[4096];
static const char* json_str(const char* json, const char* key) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return nullptr;
    p += strlen(needle);
    /* Skip optional whitespace and colon */
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return nullptr;
    p++; /* skip opening quote */
    int i = 0;
    while (*p && *p != '"' && i < 4095) val_buf[i++] = *p++;
    val_buf[i] = '\0';
    return val_buf;
}

/* Extract int value for key from JSON line. */
static int json_int(const char* json, const char* key, int def) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return def;
    p += strlen(needle);
    while (*p == ' ' || *p == ':') p++;
    return atoi(p);
}

int main(int argc, char** argv) {
    /* Unbuffered stdout for immediate response */
    setbuf(stdout, nullptr);

    while (fgets(line_buf, sizeof(line_buf), stdin)) {
        const char* cmd = json_str(line_buf, "cmd");
        if (!cmd) {
            printf("{\"error\":\"no cmd\"}\n");
            continue;
        }

        if (strcmp(cmd, "load") == 0) {
            const char* file = json_str(line_buf, "file");
            if (!file) {
                printf("{\"error\":\"no file\"}\n");
                continue;
            }
            FILE* f = fopen(file, "rb");
            if (!f) {
                printf("{\"error\":\"fopen failed\"}\n");
                continue;
            }
            fseek(f, 0, SEEK_END);
            int sz = (int)ftell(f);
            fseek(f, 0, SEEK_SET);
            void* buf = malloc(sz);
            fread(buf, 1, sz, f);
            fclose(f);
            int ret = mock_load_shm_image(buf, sz);
            free(buf);
            printf("{\"ok\":true,\"size\":%d,\"ret\":%d}\n", sz, ret);
        }
        else if (strcmp(cmd, "create") == 0) {
            ctx = new VerilatedContext;
            ctx->debug(0);
            ctx->randReset(2);
            dut = new Vvten_shm_controller(ctx, "TOP");
            sim_time = 0;
            dut->clk = 0;
            dut->rst_n = 0;
            dut->feed_ready = 0;
            dut->sched_all_committed = 0;
            dut->sched_all_drained = 0;
            dut->sched_error = 0;
            dut->sched_error_cmd_id = 0;
            dut->sched_error_code = 0;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "reset") == 0) {
            int cycles = json_int(line_buf, "cycles", 5);
            dut->rst_n = 0;
            for (int i = 0; i < cycles; i++) do_tick();
            dut->rst_n = 1;
            do_tick();
            int st = (int)dut->rootp->vten_shm_controller__DOT__state;
            printf("{\"ok\":true,\"state\":%d,\"state_name\":\"%s\"}\n",
                   st, st < 9 ? STATE_NAMES[st] : "UNKNOWN");
        }
        else if (strcmp(cmd, "tick") == 0) {
            int n = json_int(line_buf, "n", 1);
            for (int i = 0; i < n; i++) do_tick();
            int st = (int)dut->rootp->vten_shm_controller__DOT__state;
            printf("{\"ok\":true,\"state\":%d,\"state_name\":\"%s\","
                   "\"feed_valid\":%d,\"feed_done\":%d,\"sim_time\":%lu}\n",
                   st, st < 9 ? STATE_NAMES[st] : "UNKNOWN",
                   (int)dut->feed_valid, (int)dut->feed_done, sim_time);
        }
        else if (strcmp(cmd, "get_state") == 0) {
            int st = (int)dut->rootp->vten_shm_controller__DOT__state;
            printf("{\"state\":%d,\"state_name\":\"%s\"}\n",
                   st, st < 9 ? STATE_NAMES[st] : "UNKNOWN");
        }
        else if (strcmp(cmd, "get_feed") == 0) {
            int fv = (int)dut->feed_valid;
            int fd = (int)dut->feed_done;
            /* Extract opcode from feed_data (bits [3:0] of word 0) */
            int opcode = dut->feed_data[0] & 0xF;
            /* cmd_id (bits [19:4]) */
            int cmd_id = (dut->feed_data[0] >> 4) & 0xFFFF;
            printf("{\"feed_valid\":%d,\"feed_done\":%d,"
                   "\"opcode\":%d,\"cmd_id\":%d}\n",
                   fv, fd, opcode, cmd_id);
        }
        else if (strcmp(cmd, "set") == 0) {
            const char* sig = json_str(line_buf, "signal");
            int val = json_int(line_buf, "value", 0);
            if (!sig) { printf("{\"error\":\"no signal\"}\n"); continue; }
            if (strcmp(sig, "feed_ready") == 0) dut->feed_ready = val & 1;
            else if (strcmp(sig, "sched_all_committed") == 0) dut->sched_all_committed = val & 1;
            else if (strcmp(sig, "sched_all_drained") == 0) dut->sched_all_drained = val & 1;
            else if (strcmp(sig, "sched_error") == 0) dut->sched_error = val & 1;
            else if (strcmp(sig, "sched_error_cmd_id") == 0) dut->sched_error_cmd_id = val & 0xFFFF;
            else if (strcmp(sig, "sched_error_code") == 0) dut->sched_error_code = val & 0xFFFF;
            else { printf("{\"error\":\"unknown signal\"}\n"); continue; }
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "mock_set") == 0) {
            const char* field = json_str(line_buf, "field");
            int val = json_int(line_buf, "value", 0);
            if (!field) { printf("{\"error\":\"no field\"}\n"); continue; }
            if (strcmp(field, "host_status") == 0) mock_set_host_status(val);
            else if (strcmp(field, "wait_host_result") == 0) mock_set_wait_host_result(val);
            else { printf("{\"error\":\"unknown field\"}\n"); continue; }
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "mock_get") == 0) {
            const char* field = json_str(line_buf, "field");
            if (!field) { printf("{\"error\":\"no field\"}\n"); continue; }
            if (strcmp(field, "backend_status") == 0)
                printf("{\"value\":%d}\n", mock_get_backend_status());
            else if (strcmp(field, "error_code") == 0)
                printf("{\"value\":%d}\n", mock_get_error_code());
            else if (strcmp(field, "complete_count") == 0)
                printf("{\"value\":%d}\n", mock_get_complete_count());
            else if (strcmp(field, "error_count") == 0)
                printf("{\"value\":%d}\n", mock_get_error_count());
            else if (strcmp(field, "session_seq") == 0)
                printf("{\"value\":%d}\n", mock_get_session_seq());
            else { printf("{\"error\":\"unknown field\"}\n"); continue; }
        }
        else if (strcmp(cmd, "get_internals") == 0) {
            printf("{\"num_commands\":%d,\"feed_idx\":%d,\"sim_time\":%lu}\n",
                   (int)dut->rootp->vten_shm_controller__DOT__num_commands,
                   (int)dut->rootp->vten_shm_controller__DOT__feed_idx,
                   sim_time);
        }
        else if (strcmp(cmd, "read_shm") == 0) {
            int offset = json_int(line_buf, "offset", 0);
            int size = json_int(line_buf, "size", 4);
            if (size > 1024) size = 1024;
            uint8_t buf[1024];
            int rc = mock_read_shm_bytes(offset, buf, size);
            if (rc < 0) {
                printf("{\"error\":\"read_shm failed\"}\n");
            } else {
                printf("{\"ok\":true,\"bytes\":[");
                for (int i = 0; i < rc; i++) {
                    if (i > 0) printf(",");
                    printf("%d", (int)buf[i]);
                }
                printf("]}\n");
            }
        }
        else if (strcmp(cmd, "destroy") == 0) {
            if (dut) { dut->final(); delete dut; dut = nullptr; }
            if (ctx) { delete ctx; ctx = nullptr; }
            mock_reset();
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "quit") == 0) {
            if (dut) { dut->final(); delete dut; }
            if (ctx) delete ctx;
            mock_reset();
            printf("{\"ok\":true}\n");
            break;
        }
        else {
            printf("{\"error\":\"unknown cmd\"}\n");
        }
    }

    return 0;
}
