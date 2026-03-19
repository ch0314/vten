/* axilite_driver.cpp — Subprocess test driver for vten_bfm_axilite.
 *
 * JSON line protocol over stdin/stdout (same as shm_ctrl_driver).
 *
 * The BFM acts as AXI4-Lite MASTER. This driver simulates the SLAVE side
 * (memory-mapped registers). Python tests verify the BFM correctly generates
 * AXI4-Lite transactions for WRITE_REG, READ_REG, POLL_REG commands.
 */

#include "Vtb_bfm_axilite.h"
#include "Vtb_bfm_axilite___024root.h"
#include "verilated.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" {
#include "dpi_mock.h"
}

/* ── bfm_cmd_t bit layout (packed struct, 303 bits total) ──
 * Verilator represents as VlWide<10> (10 x 32-bit words).
 * Word 0 bits [3:0]   = opcode
 * Word 0 bits [19:4]  = cmd_id
 * Word 0 bits [35:20] = interface_id (crosses word boundary)
 * etc.
 * We'll pack cmd_data from a simpler struct.
 */
static void pack_bfm_cmd(uint32_t* data, int opcode, int cmd_id, int iface_id,
                          int protocol, int role, int buf_id, int probe, int sync,
                          uint32_t size, uint64_t phys_addr,
                          uint32_t reg_off, uint32_t reg_val,
                          uint32_t reg_mask, uint32_t reg_exp, int golden_buf) {
    /* Clear all 10 words */
    for (int i = 0; i < 10; i++) data[i] = 0;

    /* Pack bit-by-bit into the packed struct layout.
     * bfm_cmd_t fields (MSB to LSB in SystemVerilog packed struct):
     *   opcode[3:0], cmd_id[15:0], interface_id[15:0], protocol[7:0],
     *   role[0:0], buffer_id[15:0], probe[0:0], sync[0:0], size[31:0],
     *   phys_addr[63:0], reg_offset[31:0], reg_value[31:0],
     *   reg_mask[31:0], reg_expected[31:0], golden_buf_id[15:0]
     *
     * Total = 4+16+16+8+1+16+1+1+32+64+32+32+32+32+16 = 303 bits
     *
     * In verilator VlWide<10>, bit 0 of word[0] = LSB = golden_buf_id[0]
     * We pack from LSB to MSB.
     */
    int pos = 0;

    /* Helper: set a field of 'width' bits at bit position 'pos' */
    #define SET_FIELD(val, width) do { \
        uint64_t v = (uint64_t)(val); \
        for (int b = 0; b < (width); b++) { \
            int bit = pos + b; \
            if ((v >> b) & 1) data[bit / 32] |= (1U << (bit % 32)); \
        } \
        pos += (width); \
    } while(0)

    SET_FIELD(golden_buf, 16);   /* [15:0]   golden_buf_id */
    SET_FIELD(reg_exp, 32);      /* [47:16]  reg_expected */
    SET_FIELD(reg_mask, 32);     /* [79:48]  reg_mask */
    SET_FIELD(reg_val, 32);      /* [111:80] reg_value */
    SET_FIELD(reg_off, 32);      /* [143:112] reg_offset */
    SET_FIELD(phys_addr, 64);    /* [207:144] phys_addr */
    SET_FIELD(size, 32);         /* [239:208] size */
    SET_FIELD(sync, 1);          /* [240]    sync */
    SET_FIELD(probe, 1);         /* [241]    probe */
    SET_FIELD(buf_id, 16);       /* [257:242] buffer_id */
    SET_FIELD(role, 1);          /* [258]    role */
    SET_FIELD(protocol, 8);      /* [266:259] protocol */
    SET_FIELD(iface_id, 16);     /* [282:267] interface_id */
    SET_FIELD(cmd_id, 16);       /* [298:283] cmd_id */
    SET_FIELD(opcode, 4);        /* [302:299] opcode */

    #undef SET_FIELD
}

/* ── Simulation context ── */
static VerilatedContext* ctx = nullptr;
static Vtb_bfm_axilite* dut = nullptr;
static uint64_t sim_time = 0;
static int cycle = 0;

/* Simple register memory for the AXI4-Lite slave model */
#define REG_MEM_SIZE 4096
static uint32_t reg_mem[REG_MEM_SIZE / 4];

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

/* ── AXI4-Lite Slave Model ──
 * Responds to BFM's master transactions by acting as a simple register bank.
 * Called BEFORE do_tick() each cycle.
 *
 * The BFM's always_ff uses the pattern:
 *   m_awvalid <= 1;
 *   if (m_awready) m_awvalid <= 0;  // last NBA wins
 *
 * So we must NOT hold ready=1 before valid is asserted. Instead:
 * - When valid appears and ready is low, assert ready=1 and capture data
 *   immediately (handshake will complete at the upcoming posedge).
 * - On the next call, deassert ready.
 */
static int aw_done = 0, w_done = 0;
static uint32_t captured_awaddr = 0;
static uint32_t captured_wdata = 0;

static void slave_respond() {
    /* ── Write Address Channel ── */
    if (dut->m_awvalid && !dut->m_awready) {
        /* Assert ready → handshake completes at upcoming posedge */
        dut->m_awready = 1;
        captured_awaddr = dut->m_awaddr;
        aw_done = 1;
    } else {
        dut->m_awready = 0;
    }

    /* ── Write Data Channel ── */
    if (dut->m_wvalid && !dut->m_wready) {
        dut->m_wready = 1;
        captured_wdata = dut->m_wdata;
        w_done = 1;
    } else {
        dut->m_wready = 0;
    }

    /* ── Write Response Channel ──
     * Check B handshake first (clear), then set bvalid if needed.
     * This ensures bvalid stays high for at least one tick. */
    if (dut->m_bvalid && dut->m_bready) {
        dut->m_bvalid = 0;
        aw_done = 0;
        w_done = 0;
    } else if (aw_done && w_done && !dut->m_bvalid) {
        /* Commit write and assert response */
        if (captured_awaddr < REG_MEM_SIZE) {
            reg_mem[captured_awaddr / 4] = captured_wdata;
        }
        dut->m_bvalid = 1;
        dut->m_bresp = 0;  // OKAY
    }

    /* ── Read Address Channel ── */
    static uint32_t read_addr = 0;
    static int ar_pending = 0;

    if (dut->m_arvalid && !dut->m_arready) {
        dut->m_arready = 1;
        read_addr = dut->m_araddr;
        ar_pending = 1;
    } else {
        dut->m_arready = 0;
    }

    /* ── Read Data Channel ──
     * Check R handshake first, then assert rvalid if pending. */
    if (dut->m_rvalid && dut->m_rready) {
        dut->m_rvalid = 0;
        ar_pending = 0;
    } else if (ar_pending && !dut->m_rvalid) {
        dut->m_rvalid = 1;
        dut->m_rresp = 0;  // OKAY
        if (read_addr < REG_MEM_SIZE) {
            dut->m_rdata = reg_mem[read_addr / 4];
        } else {
            dut->m_rdata = 0xDEADBEEF;
        }
    }
}

int main() {
    setbuf(stdout, nullptr);

    while (fgets(line_buf, sizeof(line_buf), stdin)) {
        const char* cmd = json_str(line_buf, "cmd");
        if (!cmd) { printf("{\"error\":\"no cmd\"}\n"); continue; }

        if (strcmp(cmd, "create") == 0) {
            ctx = new VerilatedContext;
            ctx->debug(0);
            ctx->randReset(2);
            dut = new Vtb_bfm_axilite(ctx, "TOP");
            sim_time = 0;
            cycle = 0;
            memset(reg_mem, 0, sizeof(reg_mem));
            dut->clk = 0;
            dut->rst_n = 0;
            dut->cmd_valid = 0;
            dut->m_awready = 0;
            dut->m_wready = 0;
            dut->m_arready = 0;
            dut->m_bvalid = 0;
            dut->m_rvalid = 0;
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
            /* Issue a BFM command (WRITE_REG, READ_REG, POLL_REG) */
            int opcode = json_int(line_buf, "opcode", 5);
            int cmd_id = json_int(line_buf, "cmd_id", 0);
            int reg_off = json_int(line_buf, "reg_offset", 0);
            int reg_val = json_int(line_buf, "reg_value", 0);
            int reg_mask = json_int(line_buf, "reg_mask", 0);
            int reg_exp = json_int(line_buf, "reg_expected", 0);

            uint32_t cmd_data[10];
            pack_bfm_cmd(cmd_data, opcode, cmd_id, 0/*iface*/, 3/*AXI4L*/,
                         0/*role*/, 0/*bufid*/, 0/*probe*/, 0/*sync*/,
                         4/*size*/, 0/*phys*/, reg_off, reg_val,
                         reg_mask, reg_exp, 0/*golden*/);

            /* Assert cmd_valid for one cycle */
            for (int i = 0; i < 10; i++) dut->cmd_data[i] = cmd_data[i];
            dut->cmd_valid = 1;
            do_tick();
            dut->cmd_valid = 0;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "tick") == 0) {
            int n = json_int(line_buf, "n", 1);
            for (int i = 0; i < n; i++) {
                slave_respond();
                do_tick();
            }
            printf("{\"ok\":true,\"idle\":%d,\"done_valid\":%d,"
                   "\"done_cmd_id\":%d,\"done_error\":%d,"
                   "\"done_error_code\":%d,\"cycle\":%d}\n",
                   (int)dut->idle, (int)dut->done_valid,
                   (int)dut->done_cmd_id, (int)dut->done_error,
                   (int)dut->done_error_code, cycle);
        }
        else if (strcmp(cmd, "run_until_done") == 0) {
            int max_ticks = json_int(line_buf, "max_ticks", 100);
            int found = 0;
            for (int i = 0; i < max_ticks; i++) {
                slave_respond();
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
        else if (strcmp(cmd, "get_axi") == 0) {
            printf("{\"awaddr\":%u,\"awvalid\":%d,\"wdata\":%u,\"wvalid\":%d,"
                   "\"araddr\":%u,\"arvalid\":%d,\"idle\":%d}\n",
                   dut->m_awaddr, (int)dut->m_awvalid,
                   dut->m_wdata, (int)dut->m_wvalid,
                   dut->m_araddr, (int)dut->m_arvalid,
                   (int)dut->idle);
        }
        else if (strcmp(cmd, "get_debug") == 0) {
            auto* root = dut->rootp;
            int q_size = (int)root->tb_bfm_axilite__DOT__dut__DOT__cmd_queue.size();
            int active = (int)root->tb_bfm_axilite__DOT__dut__DOT__cmd_active;
            /* Extract opcode from current_cmd: bits [302:299] → word 9, bits [11:8] */
            uint32_t w9 = root->tb_bfm_axilite__DOT__dut__DOT__current_cmd[9];
            int opcode = (w9 >> 11) & 0xF;
            /* cmd_id: bits [298:283] → crosses word 8/9 */
            uint32_t w8 = root->tb_bfm_axilite__DOT__dut__DOT__current_cmd[8];
            int cmd_id = ((w9 & 0x7FF) << 5) | ((w8 >> 27) & 0x1F);
            printf("{\"queue_size\":%d,\"cmd_active\":%d,"
                   "\"cur_opcode\":%d,\"cur_cmd_id\":%d,"
                   "\"w9\":\"0x%08X\",\"w8\":\"0x%08X\"}\n",
                   q_size, active, opcode, cmd_id, w9, w8);
        }
        else if (strcmp(cmd, "set_reg") == 0) {
            int addr = json_int(line_buf, "addr", 0);
            int value = json_int(line_buf, "value", 0);
            if (addr < REG_MEM_SIZE) reg_mem[addr / 4] = (uint32_t)value;
            printf("{\"ok\":true}\n");
        }
        else if (strcmp(cmd, "get_reg") == 0) {
            int addr = json_int(line_buf, "addr", 0);
            uint32_t val = (addr < REG_MEM_SIZE) ? reg_mem[addr / 4] : 0;
            printf("{\"value\":%u}\n", val);
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
