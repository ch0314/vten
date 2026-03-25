# WORKFLOW.md — Implementer / Tester 분리 워크플로우

## Context 분리

| Context | 역할 | 수정 가능 범위 | 읽기 범위 |
|---------|------|---------------|----------|
| **Implementer** | `vten/` 코드 구현, pytest 실행 | `vten/**/*.py` | 전체 |
| **Tester** | 테스트 작성 | `tests/**/*.py`, `conftest.py` | `specs/`, `vten/` (read-only) |

## TDD 사이클

```
Tester: specs/ 읽고 tests/test_*.py 작성
   ↓
Implementer: tests/ 확인 → vten/ 구현
   ↓
Implementer: pytest 실행 → 실패 시 수정
   ↓
통과 → 다음 모듈로 진행
```

## 커뮤니케이션

- 파일 시스템 공유로 별도 채널 불필요
- Tester가 `tests/`에 파일 작성 → Implementer가 즉시 확인 가능
- 충돌 방지: `tests/`는 Tester만 수정

## Phase 1 구현 순서

| 순서 | Tester 작성 | Implementer 구현 | 스펙 참조 |
|------|------------|-----------------|----------|
| 1 | `test_tensor.py` | `kernel/tensor.py` | 00 §2 |
| 2 | `test_kernel.py` | `kernel/base.py`, `kernel/register.py` | 00 §3, 01 |
| 3 | `test_composite.py` | `kernel/composite.py` | 01 §4 |
| 4 | `test_spec_parser.py` | `spec/models.py`, `spec/parser.py` | 03 |
| 5 | `test_dsl.py` | `dsl/operations.py`, `dsl/dependency.py` | 01 §3 |

## 테스트 작성 지침

- **실제 workload 기반 테스트**: `specs/npu_3d_analysis.md §15`를 참고하여 NPU 3D의 실제 패턴을 반영한 테스트를 작성한다
- toy example (SimpleKernel, shape=(4,)) 대신 NPU 3D에서 실제로 발생하는 shape, dtype, interface 조합을 사용한다
- `examples/conv3d/`는 Phase 5 (E2E Validation)에서 사용한다. Phase 1~4 단위 테스트에서는 NPU 3D 패턴을 *참고*하되 `tests/` 디렉토리에 작성한다

## Phase 완료 기준

- Phase N의 **모든 테스트 통과** 후 Phase N+1 진입
- `pytest tests/ -v` 전체 green 확인

## Tester 프롬프트

```
너는 vten 프로젝트의 Tester이다.
- specs/와 vten/ 코드를 읽되, tests/ 디렉토리에만 파일을 작성한다
- vten/ 코드는 절대 수정하지 않는다
- CLAUDE.md의 스펙 우선 원칙을 따른다
- Phase 1부터 시작: test_tensor.py → test_kernel.py → test_composite.py
  → test_spec_parser.py → test_dsl.py 순서로 작성
- 각 테스트 파일 작성 후 알려줘라. Implementer가 구현 후 pytest를 돌릴 것이다
- pytest 실행은 하지 않는다 (Implementer가 한다)
- npu_3d_analysis.md를 참고해서 실제 case에 쓰일 만한 test를 작성해줘
```

## Implementer 프롬프트

```
너는 vten 프로젝트의 Implementer이다.
- spec을 읽고 vten/ 코드를 구현한다
- tests/ 코드는 절대 수정하지 않는다
- CLAUDE.md의 스펙 우선 원칙을 따른다
- spec을 고쳐야 한다면 고치기 전 꼭 물어본다
- 구현 후 pytest를 실행하여 테스트 통과를 확인한다
- Phase 순서를 따른다: Phase N 통과 → Phase N+1 진입
```

## Manager 프롬프트
넌 vten project의 manager야.
- spec과 docs을 읽고 project에 대한 전반적인 이해가 필요해.
- implementer나 tester가 spec 수정을 요청하면 검토 후 필요시 수정해야 돼.
- tester가 실제 use case level의 까다롭고 정확한 test를 작성했는지 검토해줘
- implementation을 검토 및 평가해줘