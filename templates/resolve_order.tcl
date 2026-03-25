# resolve_order.tcl — vten build Stage 4
# Vivado get_compile_order로 커널별 컴파일 순서 해결
# args: <xpr_path> <tb_top_sv_path> <output_prj_path>

set xpr_path    [lindex $argv 0]
set tb_top_path [lindex $argv 1]
set output_prj  [lindex $argv 2]

# 1) 프로젝트 열기
open_project $xpr_path

# 2) 커널별 generated SV 파일 추가 + top 설정
set gen_dir [file dirname $tb_top_path]
foreach f [glob -nocomplain $gen_dir/*.sv] {
    add_files -fileset sim_1 $f
}
set_property top tb_top [get_filesets sim_1]

# Force-enable all files (Vivado auto-disables RTL not referenced by default top)
foreach f [get_files -of [get_filesets sim_1]] {
    catch { set_property IS_ENABLED 1 $f }
}
update_compile_order -fileset sim_1

# 3) Vivado 의존성 분석 → 정렬된 순서 추출
set ordered [get_files -compile_order sources -used_in simulation -of [get_filesets sim_1]]

# Build a set of files already included by compile_order
set included [dict create]
foreach f $ordered {
    dict set included [get_property NAME $f] 1
}

# 4) .prj 파일 생성 — compile_order 결과 먼저, 누락된 파일 추가
set fp [open $output_prj w]
foreach f $ordered {
    set ftype [get_property FILE_TYPE $f]
    set lib   [get_property LIBRARY $f]
    set path  [get_property NAME $f]
    if {$ftype eq "SystemVerilog"} {
        puts $fp "sv $lib $path"
    } elseif {$ftype eq "Verilog"} {
        puts $fp "verilog $lib $path"
    }
}

# Append any RTL files missing from compile_order (auto-disabled by Vivado)
foreach f [get_files -of [get_filesets sim_1]] {
    set path [get_property NAME $f]
    set ftype [get_property FILE_TYPE $f]
    if {![dict exists $included $path]} {
        if {$ftype eq "SystemVerilog"} {
            set lib [get_property LIBRARY $f]
            puts $fp "sv $lib $path"
        } elseif {$ftype eq "Verilog"} {
            set lib [get_property LIBRARY $f]
            puts $fp "verilog $lib $path"
        }
    }
}
close $fp

# 5) 프로젝트 원복 (다음 커널을 위해 tb_top.sv 제거)
remove_files -fileset sim_1 $tb_top_path
catch {save_project}
close_project
