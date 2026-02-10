# Python 3.14 Multi-Persona Debate Agent System

이 프로젝트는 Python 3.14.2의 **Free-threading (No-GIL)** 기능을 활용하여 진정한 병렬 처리가 가능한 AI 토론 시스템입니다.

### 1. 의존성 및 환경 (Dependencies)
- **Python 3.14.2 호환성 확보**:
  - `faiss-cpu` 버전을 `1.13.0` 이상으로 업그레이드.
  - `langchain`, `langgraph` 최신 버전 적용으로 `pydantic` 및 Python 3.14 호환성 문제 해결.
- **신규 라이브러리 추가**:
  - `duckduckgo-search`: 웹 검색 기능.
  - `loguru`: 향상된 로깅 시스템.
  - `aiofiles`: 비동기 파일 입출력 지원.

### 2. 에이전트 시스템 (Agents)
- **DebateAgent 구현** (`app/agents/persona_agents.py`):
  - `BaseAgent`를 상속받아 토론 특화 에이전트 구현.
  - **4가지 페르소나**:
    - `Optimist` (낙관론자): 기회와 긍정적 측면 강조.
    - `Pessimist` (비판론자): 리스크와 윤리적 문제 지적.
    - `Realist` (현실주의자): 데이터와 실현 가능성 검토.
    - `Moderator` (중재자): 토론 내용을 종합하여 최종 결론 도출.
- **BaseAgent 리팩토링** (`app/agents/base_agent.py`):
  - 레거시 `AgentExecutor` 제거 및 최신 `langgraph` 기반 `create_react_agent` 도입.
  - 도구 호출(Tool Calling) 호환성 및 안정성 개선.

### 3. 멀티스레딩 오케스트레이터 (Core Logic)
- **DebateOrchestrator** (`app/core/orchestrator.py`):
  - `ThreadPoolExecutor`를 사용하여 에이전트들을 **병렬 실행**.
  - Python 3.14 Free-threading 환경에서 CPU 연산 병렬화 극대화.
  - `check_gil_status()`: 현재 인터프리터의 GIL 활성화 여부 확인 기능 포함.
- **SharedDebateHistory**:
  - 스레드 안전(Thread-safe)한 토론 기록 저장소 구현.

### 4. 도구 (Tools)
- **WebSearchTool** (`app/tools/search_tool.py`):
  - `DuckDuckGo` API를 활용한 실시간 웹 검색 도구.
  - 에이전트들이 최신 정보를 바탕으로 근거 있는 토론을 진행하도록 지원.

## 🛠 실행 방법

1. **API 키 설정 (@.env)**
   ```bash
   OPENAI_API_KEY=sk-your-key-here
   ```

2. **토론 시뮬레이션 실행**
   ```bash
   # 가상환경 활성화 (선택)
   source venv/bin/activate

   # 시뮬레이션 시작
   python3 debate_simulation.py "주제 입력"
   # 예: python3 debate_simulation.py "The pros and cons of Universal Basic Income"
   ```

## ⚠️ 참고 사항
- **Pydantic V1 경고**: Python 3.14와 LangChain의 일부 호환성 문제로 실행 시 경고가 발생할 수 있으나, 기능에는 지장이 없습니다.
- **DuckDuckGo Rate Limit**: 무료 검색 API 특성상 과도한 요청 시 일시적으로 차단될 수 있습니다.
