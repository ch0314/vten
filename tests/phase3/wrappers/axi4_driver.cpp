/* axi4_driver.cpp — Subprocess test driver for vten_bfm_axi4.
 *
 * The BFM is AXI4 SLAVE. This driver acts as AXI MASTER to test
 * read (PUSH) and write (PULL) paths.
 *
 * JSON line protocol: issue_cmd, read_burst, write_burst, tick, run_until_done, etc.
 */

#include "Vtb_bfm_axi4.h"
#include "Vtb_bfm_axi4___024root.h"
#include "verilated.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" {
#include "dpi_mock.h"
}

/* ── bfm_cmd_t packing ── */
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
static Vtb_bfm_axi4* dut = nullptr;
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

static long long json_ll(const char* json, const char* key, long long def) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char* p = strstr(json, needle);
    if (!p) return def;
    p += strlen(needle);
    while (*p == ' ' || *p == ':') p++;
    return atoll(p);
}

/* ── Read burst tracking ── */
#define MAX_RBEATS 256
static uint32_t r_beats[MAX_RBEATS][8]; /* 256-bit words */
static int r_beat_count = 0;
static int r_rlast_seen = 0;

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
            dut = new Vtb_bfm_axi4(ctx, "TOP");
            sim_time = 0;
            cycle = 0;
            dut->clk = 0;
            dut->rst_n = 0;
            dut->cmd_valid = 0;
            /* AXI master defaults: no transactions */
            dut->s_arvalid = 0;
            dut->s_awvalid = 0;
            dut->s_wvalid = 0;
            dut->s_rready = 1;
            dut->s_bready = 1;
            memset(dut->cmd_data, 0, sizeof(dut->cmd_data));
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
            /* Issue a BFM command (PUSH or PULL) to populate active_table */
            int opcode = json_int(line_buf, "opcode", 2);
            int cmd_id = json_int(line_buf, "cmd_id", 0);
            int buf_id = json_int(line_buf, "buffer_id", 0);
            int size = json_int(line_buf, "size", 256);
            int protocol = json_int(line_buf, "protocol", 2); /* AXI4=2 */
            long long phys_addr = json_ll(line_buf, "phys_addr", 0);

            uint32_t cmd_data[10];
            pack_bfm_cmd(cmd_data, opcode, cmd_id, 0, protocol,
                         0, buf_id, 0, 0,
                         (uint32_t)size, (uint64_t)phys_addr,
                         0, 0, 0, 0, 0);

            for (int i = 0; i < 10; i++) dut->cmd_data[i] = cmd_data[i];
            dut->cmd_valid = 1;
            do_tick();
            dut->cmd_valid = 0;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "read_burst") == 0) {
            /* Issue an AR transaction and collect R beats.
             * addr, len (0-based: 0=1 beat), size (log2 bytes) */
            long long addr = json_ll(line_buf, "addr", 0);
            int len = json_int(line_buf, "len", 0);
            int arsize = json_int(line_buf, "size", 5); /* 5 = 32 bytes = 256 bits */
            int burst = json_int(line_buf, "burst", 1); /* INCR */
            int max_ticks = json_int(line_buf, "max_ticks", 200);

            /* Issue AR */
            dut->s_araddr = (uint64_t)addr;
            dut->s_arlen = (uint8_t)len;
            dut->s_arsize = (uint8_t)arsize;
            dut->s_arburst = (uint8_t)burst;
            dut->s_arvalid = 1;
            dut->s_rready = 1;

            /* Wait for arready handshake */
            int ar_done = 0;
            r_beat_count = 0;
            r_rlast_seen = 0;

            for (int t = 0; t < max_ticks; t++) {
                /* Save pre-tick state for handshake detection */
                int prev_rvalid = dut->s_rvalid;
                int prev_rlast = dut->s_rlast;
                int prev_arready = dut->s_arready;

                do_tick();

                /* AR handshake: arvalid was set, arready was sampled pre-tick */
                if (!ar_done && dut->s_arvalid && prev_arready) {
                    ar_done = 1;
                    dut->s_arvalid = 0;
                }

                /* R channel: handshake occurred if rvalid was high going
                 * into posedge and rready was high.
                 * Use PRE-tick rlast (prev_rlast) since in Verilator the
                 * BFM's NBA for the next beat's rlast is already visible
                 * in the post-tick state. prev_rlast reflects the rlast
                 * that was presented with the beat being accepted. */
                if (prev_rvalid && dut->s_rready) {
                    r_beat_count++;
                    if (prev_rlast) {
                        r_rlast_seen = 1;
                        break;
                    }
                }
            }

            printf("{\"ok\":true,\"ar_done\":%d,\"beats\":%d,"
                   "\"rlast\":%d,\"rresp\":%d}\n",
                   ar_done, r_beat_count, r_rlast_seen,
                   (int)dut->s_rresp);
        }
        else if (strcmp(cmd, "write_burst") == 0) {
            /* Issue AW+W transactions.
             * addr, len, size, data pattern */
            long long addr = json_ll(line_buf, "addr", 0);
            int len = json_int(line_buf, "len", 0);
            int awsize = json_int(line_buf, "size", 5);
            int burst = json_int(line_buf, "burst", 1);
            int pattern = json_int(line_buf, "pattern", 0xAB); /* fill byte */
            int max_ticks = json_int(line_buf, "max_ticks", 200);

            /* Issue AW */
            dut->s_awaddr = (uint64_t)addr;
            dut->s_awlen = (uint8_t)len;
            dut->s_awsize = (uint8_t)awsize;
            dut->s_awburst = (uint8_t)burst;
            dut->s_awvalid = 1;
            dut->s_bready = 1;

            int aw_done = 0;
            int w_beat = 0;
            int total_beats = len + 1;
            int b_done = 0;

            for (int t = 0; t < max_ticks; t++) {
                int prev_awready = dut->s_awready;
                int prev_wready = dut->s_wready;
                int prev_bvalid = dut->s_bvalid;

                /* AW handshake detection: set awvalid, then check prev_awready */
                /* W channel: drive data before tick */
                if (aw_done && w_beat < total_beats && !dut->s_wvalid) {
                    dut->s_wvalid = 1;
                    for (int i = 0; i < 8; i++)
                        dut->s_wdata[i] = (uint32_t)(pattern | (w_beat << 8));
                    dut->s_wstrb = 0xFFFFFFFF;
                    dut->s_wlast = (w_beat == total_beats - 1) ? 1 : 0;
                }

                do_tick();

                /* AW handshake */
                if (!aw_done && dut->s_awvalid && prev_awready) {
                    aw_done = 1;
                    dut->s_awvalid = 0;
                }

                /* W handshake */
                if (dut->s_wvalid && prev_wready) {
                    w_beat++;
                    if (w_beat >= total_beats) {
                        dut->s_wvalid = 0;
                    } else {
                        /* Update data for next beat */
                        for (int i = 0; i < 8; i++)
                            dut->s_wdata[i] = (uint32_t)(pattern | (w_beat << 8));
                        dut->s_wlast = (w_beat == total_beats - 1) ? 1 : 0;
                    }
                }

                /* B channel */
                if (prev_bvalid && dut->s_bready) {
                    b_done = 1;
                }

                if (b_done && w_beat >= total_beats) break;
            }

            printf("{\"ok\":true,\"aw_done\":%d,\"w_beats\":%d,"
                   "\"b_done\":%d,\"bresp\":%d}\n",
                   aw_done, w_beat, b_done, (int)dut->s_bresp);
        }
        else if (strcmp(cmd, "tick") == 0) {
            int n = json_int(line_buf, "n", 1);
            for (int i = 0; i < n; i++) do_tick();
            printf("{\"ok\":true,\"idle\":%d,\"done_valid\":%d,"
                   "\"done_cmd_id\":%d,\"done_error\":%d,"
                   "\"cycle\":%d}\n",
                   (int)dut->idle, (int)dut->done_valid,
                   (int)dut->done_cmd_id, (int)dut->done_error,
                   cycle);
        }
        else if (strcmp(cmd, "run_until_done") == 0) {
            int max_ticks = json_int(line_buf, "max_ticks", 200);
            int found = 0;
            for (int i = 0; i < max_ticks; i++) {
                do_tick();
                if (dut->done_valid) { found = 1; break; }
            }
            printf("{\"ok\":true,\"done\":%d,\"done_cmd_id\":%d,"
                   "\"done_error\":%d,\"done_error_code\":%d,"
                   "\"idle\":%d,\"cycle\":%d}\n",
                   found, (int)dut->done_cmd_id,
                   (int)dut->done_error, (int)dut->done_error_code,
                   (int)dut->idle, cycle);
        }
        else if (strcmp(cmd, "get_rbeat") == 0) {
            /* Get first word of each captured R beat */
            printf("{\"count\":%d,\"beats\":[", r_beat_count);
            for (int i = 0; i < r_beat_count && i < 64; i++) {
                if (i > 0) printf(",");
                printf("%u", r_beats[i][0]);
            }
            printf("]}\n");
        }
        else if (strcmp(cmd, "get_debug") == 0) {
            auto* root = dut->rootp;
            int at_size = (int)root->tb_bfm_axi4__DOT__dut__DOT__active_table.size();
            int rp_size = (int)root->tb_bfm_axi4__DOT__dut__DOT__read_pending.size();
            int wp_size = (int)root->tb_bfm_axi4__DOT__dut__DOT__write_pending.size();
            int dq_size = (int)root->tb_bfm_axi4__DOT__dut__DOT__done_queue.size();
            int bq_size = (int)root->tb_bfm_axi4__DOT__dut__DOT__b_queue.size();
            int r_act = (int)root->tb_bfm_axi4__DOT__dut__DOT__r_active;
            int w_act = (int)root->tb_bfm_axi4__DOT__dut__DOT__w_active;
            printf("{\"active_table\":%d,\"read_pending\":%d,"
                   "\"write_pending\":%d,\"done_queue\":%d,"
                   "\"b_queue\":%d,\"r_active\":%d,\"w_active\":%d}\n",
                   at_size, rp_size, wp_size, dq_size, bq_size, r_act, w_act);
        }
        else if (strcmp(cmd, "read_stats") == 0) {
            int cmd_id = json_int(line_buf, "cmd_id", 0);
            StatsEntry stats;
            int rc = mock_read_stats(cmd_id, &stats);
            if (rc != 0) {
                printf("{\"error\":\"stats not available\"}\n");
            } else {
                printf("{\"ok\":true,\"status\":%d,\"error_code\":%d,"
                       "\"issue_cycle\":%u,\"commit_cycle\":%u,"
                       "\"first_active\":%u,\"last_active\":%u,"
                       "\"active_cycles\":%u,\"total_beats\":%u,"
                       "\"stall_cycles\":%u}\n",
                       (int)stats.status, (int)stats.error_code,
                       stats.issue_cycle, stats.commit_cycle,
                       stats.first_active_cycle, stats.last_active_cycle,
                       stats.active_cycles, stats.total_beats,
                       stats.stall_cycles);
            }
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
