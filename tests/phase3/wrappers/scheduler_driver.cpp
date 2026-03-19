/* scheduler_driver.cpp — Subprocess test driver for vten_command_scheduler.
 *
 * Drives the scheduler through feed commands and simulates BFM done signals.
 * JSON line protocol: feed_cmd, feed_batch_done, set_bfm_done, tick, get_state, etc.
 */

#include "Vtb_scheduler.h"
#include "Vtb_scheduler___024root.h"
#include "verilated.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" {
#include "dpi_mock.h"
}

/* ── bfm_cmd_t packing (same as other drivers) ── */
static void pack_bfm_cmd(uint32_t* data, int opcode, int cmd_id, int iface_id,
                          int protocol, int role, int buf_id, int probe, int sync,
                          uint32_t size, uint64_t phys_addr,
                          uint32_t reg_off, uint32_t reg_val,
                          uint32_t reg_mask, uint32_t reg_exp, int golden_buf) {
    for (int i = 0; i < 10; i++) data[i] = 0;
    int pos = 0;
    #define SET_FIELD(val, width) do { \
        uint64_t v = (uint64_t)(val); \
        for (int b = 0; b < (width); b++) { \
            int bit = pos + b; \
            if ((v >> b) & 1) data[bit / 32] |= (1U << (bit % 32)); \
        } \
        pos += (width); \
    } while(0)
    SET_FIELD(golden_buf, 16);
    SET_FIELD(reg_exp, 32);
    SET_FIELD(reg_mask, 32);
    SET_FIELD(reg_val, 32);
    SET_FIELD(reg_off, 32);
    SET_FIELD(phys_addr, 64);
    SET_FIELD(size, 32);
    SET_FIELD(sync, 1);
    SET_FIELD(probe, 1);
    SET_FIELD(buf_id, 16);
    SET_FIELD(role, 1);
    SET_FIELD(protocol, 8);
    SET_FIELD(iface_id, 16);
    SET_FIELD(cmd_id, 16);
    SET_FIELD(opcode, 4);
    #undef SET_FIELD
}

/* ── Simulation context ── */
static VerilatedContext* ctx = nullptr;
static Vtb_scheduler* dut = nullptr;
static uint64_t sim_time = 0;
static int cycle = 0;

static void do_tick() {
    dut->clk = 1;
    sim_time++;
    ctx->timeInc(1);
    dut->cycle_count = cycle;
    dut->eval();
    dut->clk = 0;
    sim_time++;
    ctx->timeInc(1);
    dut->eval();
    cycle++;
}

/* ── JSON helpers ── */
static char line_buf[65536];
static char val_buf[4096];

static const char* json_str(const char* json, const char* key) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return nullptr;
    p += strlen(needle);
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return nullptr;
    p++;
    int i = 0;
    while (*p && *p != '"' && i < 4095) val_buf[i++] = *p++;
    val_buf[i] = '\0';
    return val_buf;
}

static int json_int(const char* json, const char* key, int def) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return def;
    p += strlen(needle);
    while (*p == ' ' || *p == ':') p++;
    return atoi(p);
}

/* Helper: extract opcode from bfm_cmd_data output */
static int extract_opcode(const uint32_t* data) {
    /* opcode is the top 4 bits of the 304-bit packed struct */
    /* In our packing: opcode is at bit position 299..302 (MSB end) */
    /* data[9] bits [11:8] hold the opcode (bits 296..303 = data[9][15:0]) */
    return (data[9] >> 8) & 0xF;
}

static int extract_cmd_id(const uint32_t* data) {
    /* cmd_id is at bits 283..298 (16 bits) */
    /* data[8] bits [27:12] or data[9][0..?] depending on packing */
    /* Let's compute: bit 283 = word 8 bit 27, bit 298 = word 9 bit 10 */
    /* Actually: pos after opcode(4) is 299. cmd_id is 16 bits at pos 283..298 */
    uint32_t lo = (data[8] >> 27) & 0x1F;  /* bits 283..287 from word 8 */
    uint32_t hi = data[9] & 0xFF;           /* bits 288..295 from word 9 */
    return (int)((hi << 5) | lo) & 0xFFFF;
}

int main() {
    setbuf(stdout, nullptr);

    while (fgets(line_buf, sizeof(line_buf), stdin)) {
        const char* cmd = json_str(line_buf, "cmd");
        if (!cmd) { printf("{\"error\":\"no cmd\"}\n"); continue; }

        if (strcmp(cmd, "load") == 0) {
            const char* file = json_str(line_buf, "file");
            if (!file) { printf("{\"error\":\"no file\"}\n"); continue; }
            FILE* f = fopen(file, "rb");
            if (!f) { printf("{\"error\":\"open failed\"}\n"); continue; }
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            char* buf = (char*)malloc(sz);
            fread(buf, 1, sz, f);
            fclose(f);
            int rc = mock_load_shm_image(buf, (int)sz);
            free(buf);
            printf("{\"ok\":%s}\n", rc == 0 ? "true" : "false");
        }
        else if (strcmp(cmd, "create") == 0) {
            ctx = new VerilatedContext;
            ctx->debug(0);
            ctx->randReset(2);
            dut = new Vtb_scheduler(ctx, "TOP");
            sim_time = 0;
            cycle = 0;
            dut->clk = 0;
            dut->rst_n = 0;
            dut->feed_valid = 0;
            dut->feed_done = 0;
            memset(dut->feed_data_flat, 0, sizeof(dut->feed_data_flat));
            /* All BFMs idle by default */
            dut->bfm0_done_valid = 0; dut->bfm0_idle = 1;
            dut->bfm1_done_valid = 0; dut->bfm1_idle = 1;
            dut->bfm2_done_valid = 0; dut->bfm2_idle = 1;
            dut->bfm3_done_valid = 0; dut->bfm3_idle = 1;
            /* Default mapping: iface i → BFM i (for i < 4), rest = -1 */
            dut->itb_0 = 0; dut->itb_1 = 1; dut->itb_2 = 2; dut->itb_3 = 3;
            dut->itb_4 = -1; dut->itb_5 = -1; dut->itb_6 = -1; dut->itb_7 = -1;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "reset") == 0) {
            int cycles = json_int(line_buf, "cycles", 5);
            dut->rst_n = 0;
            for (int i = 0; i < cycles; i++) do_tick();
            dut->rst_n = 1;
            do_tick();
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "set_mapping") == 0) {
            /* Set iface_to_bfm mapping */
            dut->itb_0 = json_int(line_buf, "m0", 0);
            dut->itb_1 = json_int(line_buf, "m1", 1);
            dut->itb_2 = json_int(line_buf, "m2", 2);
            dut->itb_3 = json_int(line_buf, "m3", 3);
            dut->itb_4 = json_int(line_buf, "m4", -1);
            dut->itb_5 = json_int(line_buf, "m5", -1);
            dut->itb_6 = json_int(line_buf, "m6", -1);
            dut->itb_7 = json_int(line_buf, "m7", -1);
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "feed_cmd") == 0) {
            /* Feed one command to the scheduler via feed_valid/feed_data handshake */
            int opcode = json_int(line_buf, "opcode", 2);
            int cmd_id = json_int(line_buf, "cmd_id", 0);
            int iface_id = json_int(line_buf, "iface_id", 0);
            int protocol = json_int(line_buf, "protocol", 1);
            int role = json_int(line_buf, "role", 0);
            int sync = json_int(line_buf, "sync", 0);
            int size = json_int(line_buf, "size", 256);

            uint32_t cmd_data[10];
            pack_bfm_cmd(cmd_data, opcode, cmd_id, iface_id, protocol,
                         role, 0, 0, sync, size, 0, 0, 0, 0, 0, 0);

            for (int i = 0; i < 10; i++) dut->feed_data_flat[i] = cmd_data[i];
            dut->feed_valid = 1;

            /* Wait for feed_ready handshake */
            int accepted = 0;
            for (int t = 0; t < 20; t++) {
                do_tick();
                if (dut->feed_ready) {
                    accepted = 1;
                    break;
                }
            }
            dut->feed_valid = 0;
            printf("{\"ok\":true,\"accepted\":%d}\n", accepted);
        }
        else if (strcmp(cmd, "feed_done") == 0) {
            /* Signal end of batch feeding */
            dut->feed_done = 1;
            do_tick();
            dut->feed_done = 0;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "tick") == 0) {
            int n = json_int(line_buf, "n", 1);
            for (int i = 0; i < n; i++) do_tick();
            printf("{\"ok\":true,\"all_committed\":%d,\"all_drained\":%d,"
                   "\"error_flag\":%d,\"error_cmd_id\":%d,\"error_code\":%d,"
                   "\"cycle\":%d}\n",
                   (int)dut->all_committed, (int)dut->all_drained,
                   (int)dut->error_flag, (int)dut->error_cmd_id,
                   (int)dut->error_code, cycle);
        }
        else if (strcmp(cmd, "set_bfm_done") == 0) {
            /* Simulate BFM completion for a specific BFM index */
            int bfm_idx = json_int(line_buf, "bfm", 0);
            int done_cmd_id = json_int(line_buf, "done_cmd_id", 0);
            int done_error = json_int(line_buf, "done_error", 0);
            int err_code = json_int(line_buf, "error_code", 0);

            switch (bfm_idx) {
                case 0:
                    dut->bfm0_done_valid = 1;
                    dut->bfm0_done_cmd_id = done_cmd_id;
                    dut->bfm0_done_error = done_error;
                    dut->bfm0_done_error_code = err_code;
                    break;
                case 1:
                    dut->bfm1_done_valid = 1;
                    dut->bfm1_done_cmd_id = done_cmd_id;
                    dut->bfm1_done_error = done_error;
                    dut->bfm1_done_error_code = err_code;
                    break;
                case 2:
                    dut->bfm2_done_valid = 1;
                    dut->bfm2_done_cmd_id = done_cmd_id;
                    dut->bfm2_done_error = done_error;
                    dut->bfm2_done_error_code = err_code;
                    break;
                case 3:
                    dut->bfm3_done_valid = 1;
                    dut->bfm3_done_cmd_id = done_cmd_id;
                    dut->bfm3_done_error = done_error;
                    dut->bfm3_done_error_code = err_code;
                    break;
            }
            do_tick();
            /* Deassert done after one cycle */
            dut->bfm0_done_valid = 0;
            dut->bfm1_done_valid = 0;
            dut->bfm2_done_valid = 0;
            dut->bfm3_done_valid = 0;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "set_bfm_idle") == 0) {
            int bfm_idx = json_int(line_buf, "bfm", 0);
            int idle = json_int(line_buf, "idle", 1);
            switch (bfm_idx) {
                case 0: dut->bfm0_idle = idle; break;
                case 1: dut->bfm1_idle = idle; break;
                case 2: dut->bfm2_idle = idle; break;
                case 3: dut->bfm3_idle = idle; break;
            }
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "get_dispatched") == 0) {
            /* Check which BFMs received cmd_valid this cycle */
            printf("{\"bfm0\":%d,\"bfm1\":%d,\"bfm2\":%d,\"bfm3\":%d}\n",
                   (int)dut->bfm0_cmd_valid, (int)dut->bfm1_cmd_valid,
                   (int)dut->bfm2_cmd_valid, (int)dut->bfm3_cmd_valid);
        }
        else if (strcmp(cmd, "get_state") == 0) {
            auto* root = dut->rootp;
            int batch = (int)root->tb_scheduler__DOT__dut__DOT__batch_active;
            int nloaded = (int)root->tb_scheduler__DOT__dut__DOT__num_loaded;
            int ncmds = (int)root->tb_scheduler__DOT__dut__DOT__num_commands;
            printf("{\"batch_active\":%d,\"num_loaded\":%d,\"num_commands\":%d,"
                   "\"all_committed\":%d,\"all_drained\":%d,"
                   "\"error_flag\":%d,\"error_cmd_id\":%d,\"error_code\":%d}\n",
                   batch, nloaded, ncmds,
                   (int)dut->all_committed, (int)dut->all_drained,
                   (int)dut->error_flag, (int)dut->error_cmd_id,
                   (int)dut->error_code);
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
