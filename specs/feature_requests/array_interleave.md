# Feature Request: Array Interface Interleave Mode

## Problem

Array 인터페이스의 데이터 분배 방식이 `block_split`만 지원됨. MAC 출력처럼 beat-interleaved 레이아웃의 텐서를 array 포트에 분배할 때, 사용자가 텐서 레이아웃을 stream-contiguous로 직접 변환해야 함.

### 현재 동작 (block_split only)

```
Tensor: [B0_S0, B0_S1, B0_S2, B1_S0, B1_S1, B1_S2, ...]  (beat-interleaved)
                              ↓ block_split ÷ 3
Port 0: [B0_S0, B0_S1, B0_S2, B1_S0]   ← 틀림! S0 beats만 받아야 함
Port 1: [B1_S1, B1_S2, B2_S0, B2_S1]   ← 완전 엉킴
Port 2: [B2_S2, B3_S0, B3_S1, B3_S2]
```

### Workaround (현재)

사용자가 텐서를 stream-contiguous로 직접 재배치:
```
Tensor: [S0_B0, S0_B1, ..., S1_B0, S1_B1, ..., S2_B0, ...]
```

이 방식은 동작하지만:
- generate_inputs()가 HW의 자연스러운 beat-interleaved 레이아웃 대신 인위적 변환 필요
- golden model과 HW 데이터 흐름 사이의 불일치 발생
- 디버그 시 confusion 유발 (이 이슈 발견에 며칠 소요)

## Proposed Solution

`ArraySpec`에 `interleave` 옵션 추가:

### kernel_spec.yaml

```yaml
psum_in:
  protocol: axi4_stream
  role: slave
  data_width: 256
  array:
    dimensions: [6]
    interleave:
      unit: 32        # bytes per beat (= bus_width / 8)
  packing:
    element_width: 8
    elements_per_beat: 32
    bus_width: 256
```

### 기대 동작

```
Tensor: [B0_S0, B0_S1, ..., B0_S5, B1_S0, B1_S1, ..., B1_S5, ...]
                              ↓ interleave (unit=32, 6 ports)
Port 0: [B0_S0, B1_S0, B2_S0, ...]   ← beat 순서 유지, stream 0만
Port 1: [B0_S1, B1_S1, B2_S1, ...]
...
Port 5: [B0_S5, B1_S5, B2_S5, ...]
```

## Implementation Guide

### 1. Model (`vten/spec/models.py`)

`ArraySpec`에 `interleave` 필드 추가:

```python
@dataclass
class ArraySpec:
    dimensions: list[int]
    flat_name_pattern: str | None = None
    interleave: InterleaveSpec | None = None   # NEW
```

기존 `InterleaveSpec` (`unit: int`) 재사용 가능.

### 2. Parser (`vten/spec/parser.py`)

Array 파싱에 interleave 옵션 처리 추가.

### 3. Engine (`vten/runtime/engine.py`)

`_block_split_data` 호출부 (line 941-946) 수정:

```python
# Before:
if iface_spec.array and not exposed._port_buffers:
    flat_names = iface_spec.array.flat_names(exposed.top_interface)
    exposed._port_buffers = _block_split_data(
        exposed._serialized, flat_names, exposed._serialized_size
    )
    exposed._port_mode = "block"

# After:
if iface_spec.array and not exposed._port_buffers:
    flat_names = iface_spec.array.flat_names(exposed.top_interface)
    if iface_spec.array.interleave and exposed._serialized is not None:
        # Reuse MultiPortSerializer's interleave logic
        from vten.runtime.serializer import MultiPortSerializer
        pseudo_spec = SplitSpec(
            mode="channel_interleave",
            ports=[PortDef(name=n, base_addr=0) for n in flat_names],
            interleave=iface_spec.array.interleave,
        )
        splitter = MultiPortSerializer()
        exposed._port_buffers = splitter.split_tensor(
            exposed._serialized, pseudo_spec
        )
        exposed._port_mode = "channel_interleave"
        exposed._interleave_unit = iface_spec.array.interleave.unit
    else:
        exposed._port_buffers = _block_split_data(
            exposed._serialized, flat_names, exposed._serialized_size
        )
        exposed._port_mode = "block"
```

### 4. Output reassembly

`_read_tensor_bytes` 또는 deserialization 경로에서 `_interleave_unit`이 설정된 경우 `MultiPortSerializer.reassemble()`로 역변환 필요.

### 5. Deserialization (output 방향)

DEV_TO_HOST array tensor의 경우, 각 port의 데이터를 interleave 역순으로 재조립:

```python
if exposed._port_mode == "channel_interleave" and exposed._interleave_unit:
    raw_bytes = MultiPortSerializer.reassemble(port_data, exposed._interleave_unit)
```

## Use Case

NPU_3D의 mac_atu → psum_buffer 연결:
- mac_atu 출력: 6개 AXIS stream (OUT_GROUP × K_SIZE = 2 × 3)
- 각 beat에서 6 stream이 동시에 valid
- 자연스러운 텐서 표현: `(total_beats, 6 * bytes_per_beat)` — beat 단위로 인터리브

## Testing

기존 `tests/test_runtime_serializer.py`의 `test_channel_interleave_*` 패턴을 array 버전으로 확장:

```python
def test_array_interleave_6_ports():
    """Array with interleave should round-robin split like channel_interleave."""
    # 6 ports, unit=32, 1024 beats per port
    data = bytes(range(256)) * (6 * 1024 * 32 // 256)
    # ... verify each port gets correct interleaved slice
```
