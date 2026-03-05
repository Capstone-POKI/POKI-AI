# PitchCoach AI 서버 배포 가이드 (EC2 + Docker Compose)

## 1. 배포 아키텍처

```
EC2 t3.small (2 vCPU / 2GB RAM)
┌──────────────────────────────────────┐
│  ~/docker-compose.yml (통합)          │
│                                      │
│  ┌──────────────┐  ┌──────────────┐  │
│  │ NestJS API   │→→│ FastAPI AI   │  │
│  │ :3000        │  │ :8000        │  │
│  └──────┬───────┘  └──────────────┘  │
│         │                            │
│  ┌──────┴───────┐                    │
│  │ PostgreSQL   │                    │
│  │ :5432        │                    │
│  └──────────────┘                    │
│                                      │
│  Nginx (443 → 3000)                  │
└──────────────────────────────────────┘
```

- NestJS → FastAPI: Docker 내부 통신 (`http://fastapi-ai:8000`)
- 외부 접근: HTTPS(443) → Nginx → NestJS(3000) 만 공개
- FastAPI: 외부 미공개 (내부 전용)

---

## 2. EC2 디렉토리 구조

```
/home/ec2-user/
├── docker-compose.yml      ← 통합 (nestjs + fastapi-ai + postgres)
├── Pitchcoach-BACK/        ← NestJS 레포
├── Pitchcoach-AI/          ← FastAPI AI 레포
│   ├── .env                ← 환경변수 (Git 미포함)
│   └── credentials/        ← GCP 서비스 계정 키 (Git 미포함)
│       └── service-account.json
└── (nginx, certbot 설정)
```

---

## 3. 초기 세팅 (1회)

### 3-1. AI 레포 클론

```bash
cd ~
git clone <AI-repo-url> Pitchcoach-AI
```

### 3-2. 환경변수 생성

```bash
cat > ~/Pitchcoach-AI/.env << 'EOF'
PROJECT_ID=pitchcoachai
LOCATION=us
OCR_PROCESSOR_ID=e41bb5d1cae96184
LAYOUT_PROCESSOR_ID=82698693210d7aa8
FORM_PROCESSOR_ID=662d7f1f1e179648
GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account.json
GEMINI_API_KEY=<실제키>
OPENAI_API_KEY=<실제키>
EOF
```

### 3-3. GCP 서비스 계정 키 전송

로컬에서 실행:

```bash
scp -i pitchcoach.pem \
  credentials/pitchcoachai-a423ca0e1477.json \
  ec2-user@13.124.115.167:~/Pitchcoach-AI/credentials/service-account.json
```

### 3-4. docker-compose.yml 수정

`~/docker-compose.yml`에 fastapi-ai 서비스 추가:

```yaml
version: '3'
services:
  nestjs:
    build: ./Pitchcoach-BACK
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/poki?schema=public
      AI_SERVER_URL: http://fastapi-ai:8000
    depends_on:
      - postgres
      - fastapi-ai
    restart: always

  fastapi-ai:
    build: ./Pitchcoach-AI
    ports:
      - "8000:8000"
    env_file: ./Pitchcoach-AI/.env
    volumes:
      - ./Pitchcoach-AI/credentials:/app/credentials:ro
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: poki
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: always

volumes:
  pgdata:
```

---

## 4. 빌드 및 실행

```bash
cd ~
docker compose down
docker compose up -d --build
```

### 로그 확인

```bash
# 전체
docker compose logs -f

# AI 서버만
docker compose logs -f fastapi-ai
```

### 상태 확인

```bash
# 컨테이너 상태
docker compose ps

# Health check
curl http://localhost:8000/health

# NestJS → FastAPI 내부 통신 테스트
docker compose exec nestjs curl http://fastapi-ai:8000/health
```

---

## 5. CI/CD 자동 배포

### 트리거

`main` 브랜치 push 시 GitHub Actions 자동 실행

### 동작

```
Push main
 → GitHub Actions
 → SSH EC2
 → cd ~/Pitchcoach-AI && git pull origin main
 → cd ~ && docker compose down && docker compose up -d --build
```

### GitHub Secrets 필요

| Secret   | 값                  |
|----------|---------------------|
| EC2_HOST | 13.124.115.167      |
| EC2_KEY  | pitchcoach.pem 내용 |

### 주의

- 백엔드 deploy.yml도 같은 `docker compose up -d --build` 실행
- 양쪽 CI/CD가 동시에 실행되면 충돌 가능 → 순차 배포 권장

---

## 6. API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/api/pitches/{pitchId}/notices/analyze` | 공고문 업로드 및 분석 |
| GET | `/api/notices/{noticeId}` | 공고문 분석 결과 조회 |
| PATCH | `/api/notices/{noticeId}` | 공고문 기본 정보 수정 |
| POST | `/api/pitches/{pitchId}/ir-decks/analyze` | IR Deck 업로드 및 분석 |
| GET | `/api/ir-decks/{deckId}` | IR Deck 종합 분석 결과 |
| GET | `/api/ir-decks/{deckId}/slides` | 슬라이드별 상세 조회 |

Swagger: `http://localhost:8000/docs`

---

## 7. 예상 리소스 사용량

| 서비스 | 예상 메모리 |
|--------|------------|
| NestJS | ~200MB |
| FastAPI AI | ~300MB |
| PostgreSQL | ~200MB |
| Nginx | ~20MB |
| 합계 | ~800MB |

2GB RAM 기준 여유 충분.

---

## 8. 트러블슈팅

### 컨테이너가 시작되지 않을 때

```bash
docker compose logs fastapi-ai
```

### GCP 인증 오류

```
google.auth.exceptions.DefaultCredentialsError
```

→ credentials 볼륨 마운트 확인:

```bash
docker compose exec fastapi-ai ls -la /app/credentials/
```

### 메모리 부족

```bash
free -h
docker stats --no-stream
```

→ swap 추가:

```bash
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

### 디스크 부족

```bash
docker system prune -f
docker image prune -a -f
```

---

## 9. 보안 체크리스트

- [ ] `.env`는 Git에 포함되지 않음
- [ ] `credentials/`는 Git에 포함되지 않음
- [ ] 8000 포트는 외부 미공개 (Security Group에서 차단)
- [ ] EC2 SSH 키는 GitHub Secrets에만 존재
- [ ] API 키는 `.env`에만 존재
