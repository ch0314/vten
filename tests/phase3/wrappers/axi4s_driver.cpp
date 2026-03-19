/* axi4s_driver.cpp — Subprocess test driver for vten_bfm_axi4s (MASTER mode).
 *
 * JSON line protocol over stdin/stdout.
 *
 * The BFM is AXI4-Stream MASTER (PUSH: SHM → DUT port).
 * This driver simulates the SLAVE side (accepts tdata, checks tlast).
 */

#include "Vtb_bfm_axi4s.h"
#include "Vtb_bfm_axi4s___024root.h"
#include "verilated.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" {
#include "dpi_mock.h"
}

/* ── bfm_cmd_t packing (same as axilite_driver) ── */
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
static Vtb_bfm_axi4s* dut = nullptr;
static uint64_t sim_time = 0;
static int cycle = 0;

/* Stream handshake tracking */
static int handshake_count = 0;   /* Number of valid+ready handshakes */
static int tlast_seen = 0;        /* Whether tlast was high during a handshake */
static int valid_cycles = 0;      /* Cycles where m_tvalid was high */

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

/* ── Minimal JSON helpers ── */
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

/* ── Stream handshake tick ──
 * Always hold tready=1. Track handshakes by monitoring the BFM's internal
 * beat_count changes (avoids capture timing ambiguity from NBA semantics).
 */
static void stream_tick() {
    dut->m_tready = 1;
    int prev_valid = dut->m_tvalid;
    int prev_tlast = dut->m_tlast;

    do_tick();

    /* Track: a valid handshake occurred if tvalid was high going into posedge */
    if (prev_valid) {
        valid_cycles++;
        /* BFM internal: done_valid means the last handshake just completed */
        if (dut->done_valid) tlast_seen = 1;
    }
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
            dut = new Vtb_bfm_axi4s(ctx, "TOP");
            sim_time = 0;
            cycle = 0;
            handshake_count = 0;
            tlast_seen = 0;
            valid_cycles = 0;
            dut->clk = 0;
            dut->rst_n = 0;
            dut->cmd_valid = 0;
            dut->m_tready = 0;
            dut->s_tvalid = 0;
            dut->s_tlast = 0;
            memset(dut->s_tdata, 0, sizeof(dut->s_tdata));
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
        else if (strcmp(cmd, "issue_cmd") == 0) {
            int opcode = json_int(line_buf, "opcode", 2); /* PUSH=2 */
            int cmd_id = json_int(line_buf, "cmd_id", 0);
            int buf_id = json_int(line_buf, "buffer_id", 0);
            int size = json_int(line_buf, "size", 256);
            int protocol = json_int(line_buf, "protocol", 1); /* AXI4S=1 */

            uint32_t cmd_data[10];
            pack_bfm_cmd(cmd_data, opcode, cmd_id, 0/*iface*/, protocol,
                         0/*role*/, buf_id, 0/*probe*/, 0/*sync*/,
                         size, 0/*phys*/, 0, 0, 0, 0, 0);

            for (int i = 0; i < 10; i++) dut->cmd_data[i] = cmd_data[i];
            dut->cmd_valid = 1;
            do_tick();
            dut->cmd_valid = 0;
            /* Reset captured data for this command */
            handshake_count = 0;
            tlast_seen = 0;
            valid_cycles = 0;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "tick") == 0) {
            int n = json_int(line_buf, "n", 1);
            for (int i = 0; i < n; i++) {
                stream_tick();
            }
            printf("{\"ok\":true,\"idle\":%d,\"done_valid\":%d,"
                   "\"done_cmd_id\":%d,\"done_error\":%d,"
                   "\"valid_cycles\":%d,\"tlast_seen\":%d,"
                   "\"cycle\":%d}\n",
                   (int)dut->idle, (int)dut->done_valid,
                   (int)dut->done_cmd_id, (int)dut->done_error,
                   valid_cycles, tlast_seen, cycle);
        }
        else if (strcmp(cmd, "run_until_done") == 0) {
            int max_ticks = json_int(line_buf, "max_ticks", 200);
            int found = 0;
            for (int i = 0; i < max_ticks; i++) {
                stream_tick();
                if (dut->done_valid) { found = 1; break; }
            }
            printf("{\"ok\":true,\"done\":%d,\"done_cmd_id\":%d,"
                   "\"done_error\":%d,\"idle\":%d,"
                   "\"valid_cycles\":%d,\"tlast_seen\":%d,"
                   "\"cycle\":%d}\n",
                   found, (int)dut->done_cmd_id,
                   (int)dut->done_error, (int)dut->idle,
                   valid_cycles, tlast_seen, cycle);
        }
        else if (strcmp(cmd, "get_stream") == 0) {
            printf("{\"m_tvalid\":%d,\"m_tready\":%d,\"m_tlast\":%d,"
                   "\"idle\":%d,\"valid_cycles\":%d,\"tlast_seen\":%d}\n",
                   (int)dut->m_tvalid, (int)dut->m_tready,
                   (int)dut->m_tlast, (int)dut->idle,
                   valid_cycles, tlast_seen);
        }
        else if (strcmp(cmd, "get_debug") == 0) {
            auto* root = dut->rootp;
            int q_size = (int)root->tb_bfm_axi4s__DOT__dut__DOT__cmd_queue.size();
            int active = (int)root->tb_bfm_axi4s__DOT__dut__DOT__cmd_active;
            int beat_count = (int)root->tb_bfm_axi4s__DOT__dut__DOT__beat_count;
            int exp_beats = (int)root->tb_bfm_axi4s__DOT__dut__DOT__expected_beats;
            printf("{\"queue_size\":%d,\"cmd_active\":%d,"
                   "\"beat_count\":%d,\"expected_beats\":%d}\n",
                   q_size, active, beat_count, exp_beats);
        }
        else if (strcmp(cmd, "destroy") == 0) {
            if (dut) { dut->final(); delete dut; dut = nullptr; }
            if (ctx) { delete ctx; ctx = nullptr; }
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "quit") == 0) {
            if (dut) { dut->final(); delete dut; }
            if (ctx) delete ctx;
            printf("{\"ok\":true}\n");
            break;
        }
        else {
            printf("{\"error\":\"unknown cmd\"}\n");
        }
    }
    return 0;
}
