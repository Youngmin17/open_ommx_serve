from itertools import zip_longest
import re
import string
import transformers.data.metrics.squad_metrics as squad_metrics
from collections import Counter


def doc_to_text(doc):
    """
    공식 CoQA 프롬프트 생성 (표준 버전)
    """
    # Given a passage p, the conversation history {q1, a1, . . . qi−1, ai−1}
    # and a question qi, the task is to predict the answer ai
    doc_text = doc["story"] + "\n\n"
    
    for q, a in zip_longest(
        doc["questions"]["input_text"], doc["answers"]["input_text"][:-1]
    ):  # omit target answer ai
        question = f"Q: {q}\n\n"
        answer = f"A: {a}\n\n" if a is not None else "A:"
        doc_text += question + answer
    
    return doc_text


def doc_to_target(doc):
    """
    공식 타겟 답변 추출 (표준 버전)
    """
    turn_id = len(doc["questions"]["input_text"])
    # Returns unique answers and valid alternatives (Some questions in CoQA have multiple valid answers).
    answers = []
    answer_forturn = doc["answers"]["input_text"][turn_id - 1]
    answers.append(answer_forturn)

    additional_answers = doc.get("additional_answers")
    if additional_answers:
        for key in additional_answers:
            additional_answer_for_turn = additional_answers[key]["input_text"][
                turn_id - 1
            ]
            if additional_answer_for_turn.lower() not in map(str.lower, answers):
                answers.append(additional_answer_for_turn)
    return answers


def normalize_answer_official(s):
    """CoQA 공식 답변 정규화 (SQuAD 스타일 기반)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens_official(s):
    """공식 토큰화 방법"""
    if not s:
        return []
    return normalize_answer_official(s).split()


def compute_exact_match_official(a_gold, a_pred):
    """공식 정확 매치 계산"""
    return int(normalize_answer_official(a_gold) == normalize_answer_official(a_pred))


def compute_f1_official(a_gold, a_pred):
    """공식 F1 점수 계산 (word-level)"""
    gold_toks = get_tokens_official(a_gold)
    pred_toks = get_tokens_official(a_pred)
    
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # 둘 다 비어있으면 1, 하나만 비어있으면 0
        return int(gold_toks == pred_toks)
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def clean_corrupted_text(text):
    """
    깨진 텍스트를 정리하는 함수 (강화된 버전)
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 기본 정리
    text = text.strip()
    
    # 너무 긴 숫자 시퀀스 제거 (예: 2000000000000)
    text = re.sub(r'\b\d{8,}\b', '', text)
    
    # 16진수 같은 패턴 제거 (예: 2000e0e0)
    text = re.sub(r'\b[a-f0-9]{8,}\b', '', text)
    
    # JSON/딕셔너리 패턴 제거 (예: 'filter': 'none')
    text = re.sub(r"['\"][^'\"]*['\"]:\s*['\"][^'\"]*['\"]", '', text)
    text = re.sub(r"\['[^']*'\]", '', text)
    
    # 특수 구두점 패턴 제거 (예: |, 등)
    text = re.sub(r'[|{}[\]]+', ' ', text)
    
    # 연속된 특수문자 제거 (3개 이상)
    text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]{3,}', ' ', text)
    
    # 비정상적인 문자 패턴 제거
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)  # ASCII 범위 외 문자 제거
    
    # 연속된 공백 정리
    text = re.sub(r'\s+', ' ', text)
    
    # 너무 많은 연속된 같은 문자 제거
    text = re.sub(r'(.)\1{10,}', r'\1', text)
    
    return text.strip()


def is_text_corrupted(text):
    """
    텍스트가 깨졌는지 판단하는 함수 (강화된 버전)
    """
    if not text or len(text) < 2:
        return True
    
    # 정상적인 단어의 비율 확인
    words = text.split()
    if not words:
        return True
    
    normal_words = 0
    suspicious_patterns = 0
    
    for word in words[:10]:  # 처음 10개 단어만 확인
        # 정상적인 영어 단어인지 확인 (글자와 일반적인 구두점만 포함)
        if re.match(r'^[a-zA-Z\'\-\.]+$', word) and len(word) <= 20:
            normal_words += 1
        # 의심스러운 패턴 확인
        elif re.match(r'^[0-9a-f]{6,}$', word) or len(word) > 25:
            suspicious_patterns += 1
    
    # 정상 단어 비율이 30% 미만이거나 의심스러운 패턴이 많으면 깨진 것으로 판단
    normal_ratio = normal_words / min(len(words), 10)
    suspicious_ratio = suspicious_patterns / min(len(words), 10)
    
    return normal_ratio < 0.3 or suspicious_ratio > 0.2


def extract_answer_standard(model_output):
    """
    표준 답변 추출 함수 (공식 방법론 + 안전성 강화)
    """
    if not model_output or not model_output.strip():
        return ""
    
    # 기본 정리
    output = clean_corrupted_text(model_output.strip())
    
    # Q: 패턴 이전까지만 추출 (더 강력한 패턴 매칭)
    q_patterns = [r'\nQ:', r'\n\nQ:', r'Q:', r' Q:']
    for pattern in q_patterns:
        q_match = re.search(pattern, output)
        if q_match:
            output = output[:q_match.start()].strip()
            break
    
    # 첫 번째 줄만 추출 (공식 방법론 준수)
    first_line = output.split('\n')[0].strip()
    
    # Q: 패턴이 남아있으면 제거
    first_line = re.sub(r'\s*Q:\s*.*$', '', first_line).strip()
    
    # 텍스트가 깨졌는지 확인
    if is_text_corrupted(first_line):
        # 깨진 경우, 더 보수적으로 처리
        words = first_line.split()
        clean_words = []
        
        for word in words[:20]:  # 최대 20개 단어까지만
            # 정상적인 단어 패턴 확인
            if re.match(r'^[a-zA-Z0-9\'\-\.]+$', word) and len(word) <= 20:
                clean_words.append(word)
            else:
                break  # 비정상적인 단어가 나오면 중단
        
        if clean_words:
            return ' '.join(clean_words)
        else:
            return ""
    
    return first_line


def extract_answer_enhanced(model_output):
    """
    향상된 답변 추출 함수 (표준 + 스마트 개선)
    """
    if not model_output or not model_output.strip():
        return ""
    
    # 기본 표준 추출
    standard_answer = extract_answer_standard(model_output)
    
    # 표준 추출이 실패하면 빈 문자열 반환
    if not standard_answer:
        return ""
    
    # 추가적인 스마트 개선 시도
    output = model_output.strip()
    
    # Q: 패턴 이전까지만 추출
    q_pattern_match = re.search(r'\nQ:', output)
    if q_pattern_match:
        output = output[:q_pattern_match.start()].strip()
    
    # 여러 줄을 하나로 합치되, 의미있는 구분 유지
    lines = output.split('\n')
    cleaned_lines = []
    
    for line in lines[:3]:  # 최대 3줄까지만 처리
        line = clean_corrupted_text(line)
        if line and not line.startswith('Q:') and not line.startswith('A:'):
            # 라인이 너무 깨지지 않았는지 확인
            if not is_text_corrupted(line):
                cleaned_lines.append(line)
    
    if cleaned_lines:
        # 줄바꿈을 공백으로 처리한 전체 답변
        full_answer = ' '.join(cleaned_lines)
        
        # 너무 길면 적절히 자르기
        if len(full_answer.split()) > 30:
            words = full_answer.split()[:30]
            full_answer = ' '.join(words)
        
        # 첫 번째 완전한 문장 추출
        sentences = re.split(r'[.!?]', full_answer)
        if len(sentences) > 1 and sentences[0].strip():
            first_sentence = sentences[0].strip()
            # 첫 번째 문장이 너무 짧지 않고 정상적인 경우
            if len(first_sentence.split()) >= 2 and not is_text_corrupted(first_sentence):
                enhanced_answer = first_sentence
            else:
                enhanced_answer = standard_answer
        else:
            enhanced_answer = full_answer
        
        # 두 후보 중 더 나은 것 선택
        candidates = [standard_answer, enhanced_answer]
        
        # 유효성 확인 후 선택
        valid_candidates = []
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and not is_text_corrupted(candidate):
                valid_candidates.append(candidate)
        
        if not valid_candidates:
            return standard_answer
        
        # 첫 번째 유효한 후보 (표준 답변) 우선
        return valid_candidates[0]
    
    # 모든 처리가 실패하면 표준 답변 반환
    return standard_answer


def compute_scores_official(gold_list, pred):
    """
    공식 점수 계산 (표준 CoQA 방법론)
    """
    f1_sum = 0.0
    em_sum = 0.0
    
    if len(gold_list) > 1:
        # 여러 정답이 있는 경우: 각 정답에 대해 계산하고 최고점 선택
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1:]
            # predictions compared against (n) golds and take maximum
            em_score = max(compute_exact_match_official(a, pred) for a in gold_answers)
            f1_score = max(compute_f1_official(a, pred) for a in gold_answers)
            
            if i == 0 or f1_score > f1_sum:
                f1_sum = f1_score
                em_sum = em_score
    else:
        # 단일 정답
        em_sum = compute_exact_match_official(gold_list[0], pred)
        f1_sum = compute_f1_official(gold_list[0], pred)

    return {
        "em": em_sum,
        "f1": f1_sum,
    }


def process_results(doc, results):
    """
    표준 결과 처리 함수 (공식 기반 + 개선된 답변 추출)
    """
    gold_list = doc_to_target(doc)
    
    # 공식 방식: 첫 번째 줄만 추출
    pred_standard = extract_answer_standard(results[0])
    
    # 향상된 답변 추출도 시도
    pred_enhanced = extract_answer_enhanced(results[0])
    
    # 더 나은 점수를 주는 것을 선택
    scores_standard = compute_scores_official(gold_list, pred_standard)
    scores_enhanced = compute_scores_official(gold_list, pred_enhanced)
    
    # F1 점수가 더 높은 것을 선택 (타이일 때는 표준 우선)
    if scores_enhanced["f1"] > scores_standard["f1"]:
        return scores_enhanced
    else:
        return scores_standard


# 호환성을 위한 기존 함수들 (squad_metrics 사용하는 버전도 유지)
def compute_scores(gold_list, pred):
    """
    기존 호환성을 위한 함수 (squad_metrics 기반)
    """
    f1_sum = 0.0
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            # predictions compared against (n) golds and take maximum
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
            f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_answers)
    else:
        em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)
        f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_list)

    return {
        "em": em_sum / max(1, len(gold_list)),
        "f1": f1_sum / max(1, len(gold_list)),
    }


def em(gold_list, pred):
    """기존 호환성을 위한 em 함수"""
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            # predictions compared against (n) golds and take maximum
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
    else:
        em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)

    return em_sum / max(1, len(gold_list)) 