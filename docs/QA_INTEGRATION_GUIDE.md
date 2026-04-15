# AI-Backend Q&A 통합 가이드

## 개요

Pitchcoach-AI와 Pitchcoach-BACK의 Q&A 기능 통합 설명입니다.

## 아키텍처

```
┌─────────────────────┐
│  Frontend (React)   │
└──────────┬──────────┘
           │
     ┌─────▼──────┐
     │  Backend   │
     │  (NestJS)  │
     └─────┬──────┘
           │
     ┌─────▼──────────┐
     │  AI Service    │
     │  (FastAPI)     │
     └────────────────┘
```

## API 플로우

### 1. 질문 생성 및 조회

```
Frontend → Backend (POST /api/pitches/:pitchId/questions/generate)
        ↓
Backend → AI Service (POST /api/pitches/:pitchId/qa/questions/generate)
        ↓
AI Service → Database (저장 또는 메모리)
        ↓
Backend → Frontend (질문 리스트 반환)
```

### 2. 답변 제출 및 평가

```
Frontend → Backend (POST /api/pitches/:pitchId/answers)
        ↓
Backend → AI Service (POST /api/pitches/:pitchId/qa/sessions/:sessionId/answers)
        ↓
AI Service (Gemini) → 답변 평가
        ↓
Backend → Database (저장)
        ↓
Frontend (답변 평가 결과 표시)
```

## Backend 통합 구현

### 1. 환경 설정

`.env` 파일에 AI 서비스 URL 추가:

```bash
# AI 서비스 연결 설정
AI_SERVICE_URL=http://localhost:8000
AI_SERVICE_TIMEOUT=30000  # 밀리초
```

### 2. QA 서비스 모듈 작성

**src/modules/qa/qa.service.ts:**

```typescript
import { Injectable } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { ConfigService } from '@nestjs/config';
import { firstValueFrom } from 'rxjs';

@Injectable()
export class QaService {
  private aiServiceUrl: string;

  constructor(
    private httpService: HttpService,
    private configService: ConfigService,
  ) {
    this.aiServiceUrl = this.configService.get('AI_SERVICE_URL', 'http://localhost:8000');
  }

  /**
   * AI 서비스에서 Q&A 질문을 생성합니다.
   */
  async generateQuestions(
    pitchId: string,
    noticeContent: string,
    irDeckSummary: string,
    presentationContent?: string,
  ) {
    const formData = new FormData();
    formData.append('notice_content', noticeContent);
    formData.append('irdecksummary', irDeckSummary);
    if (presentationContent) {
      formData.append('presentation_content', presentationContent);
    }

    const response = await firstValueFrom(
      this.httpService.post(
        `${this.aiServiceUrl}/api/pitches/${pitchId}/qa/questions/generate`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: this.configService.get('AI_SERVICE_TIMEOUT', 30000),
        },
      ),
    );

    return response.data;
  }

  /**
   * 특정 세션의 질문 리스트를 조회합니다.
   */
  async getQuestions(pitchId: string, sessionId?: string) {
    const url = sessionId
      ? `${this.aiServiceUrl}/api/pitches/${pitchId}/qa/sessions/${sessionId}`
      : `${this.aiServiceUrl}/api/pitches/${pitchId}/questions`;

    const response = await firstValueFrom(
      this.httpService.get(url, {
        timeout: this.configService.get('AI_SERVICE_TIMEOUT', 30000),
      }),
    );

    return response.data;
  }

  /**
   * 답변을 제출하고 평가합니다.
   */
  async submitAnswer(
    pitchId: string,
    sessionId: string,
    questionId: string,
    textTranscript?: string,
    audioFile?: Express.Multer.File,
  ) {
    const formData = new FormData();
    formData.append('question_id', questionId);
    if (textTranscript) {
      formData.append('text_transcript', textTranscript);
    }
    if (audioFile) {
      formData.append('audio', new Blob([audioFile.buffer]), audioFile.originalname);
    }

    const response = await firstValueFrom(
      this.httpService.post(
        `${this.aiServiceUrl}/api/pitches/${pitchId}/qa/sessions/${sessionId}/answers`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: this.configService.get('AI_SERVICE_TIMEOUT', 30000),
        },
      ),
    );

    return response.data;
  }

  /**
   * 평가 결과를 조회합니다.
   */
  async getAnswerEvaluation(
    pitchId: string,
    sessionId: string,
    answerId: string,
  ) {
    const response = await firstValueFrom(
      this.httpService.get(
        `${this.aiServiceUrl}/api/pitches/${pitchId}/qa/sessions/${sessionId}/answers/${answerId}`,
        {
          timeout: this.configService.get('AI_SERVICE_TIMEOUT', 30000),
        },
      ),
    );

    return response.data;
  }
}
```

### 3. QA 컨트롤러 작성

**src/modules/qa/qa.controller.ts:**

```typescript
import {
  Controller,
  Post,
  Get,
  Param,
  Body,
  UseGuards,
  UseInterceptors,
  UploadedFile,
} from '@nestjs/common';
import { JwtAuthGuard } from 'src/auth/guards/jwt-auth.guard';
import { FileInterceptor } from '@nestjs/platform-express';
import { QaService } from './qa.service';
import { CreateQuestionDto, SubmitAnswerDto } from './dto';

@Controller('/api/pitches/:pitchId')
@UseGuards(JwtAuthGuard)
export class QaController {
  constructor(private qaService: QaService) {}

  /**
   * 질문을 생성합니다.
   * POST /api/pitches/:pitchId/questions/generate
   */
  @Post('questions/generate')
  async generateQuestions(
    @Param('pitchId') pitchId: string,
    @Body() createQuestionDto: CreateQuestionDto,
  ) {
    return this.qaService.generateQuestions(
      pitchId,
      createQuestionDto.noticeContent,
      createQuestionDto.irDeckSummary,
      createQuestionDto.presentationContent,
    );
  }

  /**
   * 질문 리스트를 조회합니다.
   * GET /api/pitches/:pitchId/questions
   */
  @Get('questions')
  async getQuestions(@Param('pitchId') pitchId: string) {
    return this.qaService.getQuestions(pitchId);
  }

  /**
   * 답변을 제출합니다.
   * POST /api/pitches/:pitchId/qa/sessions/:sessionId/answers
   */
  @Post('qa/sessions/:sessionId/answers')
  @UseInterceptors(FileInterceptor('audio'))
  async submitAnswer(
    @Param('pitchId') pitchId: string,
    @Param('sessionId') sessionId: string,
    @Body() submitAnswerDto: SubmitAnswerDto,
    @UploadedFile() audioFile?: Express.Multer.File,
  ) {
    return this.qaService.submitAnswer(
      pitchId,
      sessionId,
      submitAnswerDto.questionId,
      submitAnswerDto.textTranscript,
      audioFile,
    );
  }

  /**
   * 평가 결과를 조회합니다.
   * GET /api/pitches/:pitchId/qa/sessions/:sessionId/answers/:answerId
   */
  @Get('qa/sessions/:sessionId/answers/:answerId')
  async getAnswerEvaluation(
    @Param('pitchId') pitchId: string,
    @Param('sessionId') sessionId: string,
    @Param('answerId') answerId: string,
  ) {
    return this.qaService.getAnswerEvaluation(pitchId, sessionId, answerId);
  }
}
```

### 4. DTO 정의

**src/modules/qa/dto/index.ts:**

```typescript
export class CreateQuestionDto {
  noticeContent: string;
  irDeckSummary: string;
  presentationContent?: string;
}

export class SubmitAnswerDto {
  questionId: string;
  textTranscript?: string;
  // audio는 multipart/form-data로 전송됨
}
```

### 5. 모듈 등록

**src/modules/qa/qa.module.ts:**

```typescript
import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { QaService } from './qa.service';
import { QaController } from './qa.controller';

@Module({
  imports: [HttpModule],
  controllers: [QaController],
  providers: [QaService],
  exports: [QaService],
})
export class QaModule {}
```

## 데이터 저장 (선택사항)

현재 AI 서비스는 메모리 기반이므로, Backend에서 데이터를 영속화할 수 있습니다:

```typescript
// Backend의 Question 엔티티
@Entity('questions')
export class Question {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  pitchId: string;

  @Column()
  sessionId: string;

  @Column()
  type: string; // NOTICE, PITCHBOOK, PRESENTER, EVALUATOR

  @Column('text')
  content: string;

  @Column('text', { nullable: true })
  guidance: string;

  @Column({ type: 'int' })
  order: number;

  @CreateDateColumn()
  createdAt: Date;
}

// Backend의 Answer 엔티티
@Entity('answers')
export class Answer {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  pitchId: string;

  @Column()
  questionId: string;

  @Column('text')
  textTranscript: string;

  @Column('json', { nullable: true })
  feedback: {
    score: number;
    relevance: number;
    clarity: number;
    structure: number;
    strengths: string[];
    improvements: string[];
    reasoning: string;
  };

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  evaluatedAt: Date;
}
```

## 테스트

### cURL 테스트

```bash
# 1. 질문 생성 (AI 서비스)
curl -X POST "http://localhost:8000/api/pitches/pitch-123/qa/questions/generate" \
  -F "notice_content=공고문..." \
  -F "irdecksummary=IR Deck..."

# 2. 질문 조회 (Backend)
curl -X GET "http://localhost:3000/api/pitches/pitch-123/questions" \
  -H "Authorization: Bearer $ACCESS_TOKEN"

# 3. 답변 제출 (Backend)
curl -X POST "http://localhost:3000/api/pitches/pitch-123/qa/sessions/session-id/answers" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "questionId=question-id" \
  -F "textTranscript=답변 내용..."
```

## 다음 단계

1. **Database 영속화**: QA results를 DB에 저장
2. **실시간 알림**: 평가 완료 시 WebSocket으로 알림
3. **배치 처리**: 여러 답변을 동시에 처리
4. **Analytics**: 평가 결과 분석 대시보드
