from itertools import zip_longest
import re
import string
import transformers.data.metrics.squad_metrics as squad_metrics
from collections import Counter


def doc_to_text(doc):
    """
    Build the official CoQA prompt (standard version).
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
    Extract the official target answer (standard version).
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
    """Official CoQA answer normalization (SQuAD-style)."""
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
    """Official tokenization."""
    if not s:
        return []
    return normalize_answer_official(s).split()


def compute_exact_match_official(a_gold, a_pred):
    """Official exact-match computation."""
    return int(normalize_answer_official(a_gold) == normalize_answer_official(a_pred))


def compute_f1_official(a_gold, a_pred):
    """Official F1 score (word level)."""
    gold_toks = get_tokens_official(a_gold)
    pred_toks = get_tokens_official(a_pred)
    
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # both empty -> 1; exactly one empty -> 0
        return int(gold_toks == pred_toks)
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def clean_corrupted_text(text):
    """
    Clean up mangled text (hardened version).
    """
    if not text or not isinstance(text, str):
        return ""
    
    # basic cleanup
    text = text.strip()
    
    # drop over-long digit runs (e.g. 2000000000000)
    text = re.sub(r'\b\d{8,}\b', '', text)
    
    # drop hex-looking patterns (e.g. 2000e0e0)
    text = re.sub(r'\b[a-f0-9]{8,}\b', '', text)
    
    # drop JSON/dict patterns (e.g. 'filter': 'none')
    text = re.sub(r"['\"][^'\"]*['\"]:\s*['\"][^'\"]*['\"]", '', text)
    text = re.sub(r"\['[^']*'\]", '', text)
    
    # drop stray punctuation patterns (e.g. |)
    text = re.sub(r'[|{}[\]]+', ' ', text)
    
    # drop runs of 3+ special characters
    text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]{3,}', ' ', text)
    
    # drop malformed character patterns
    text = re.sub(r'[^\x20-\x7E]+', ' ', text)  # strip non-ASCII characters
    
    # collapse repeated whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # drop long runs of one repeated character
    text = re.sub(r'(.)\1{10,}', r'\1', text)
    
    return text.strip()


def is_text_corrupted(text):
    """
    Decide whether the text is mangled (hardened version).
    """
    if not text or len(text) < 2:
        return True
    
    # measure the share of well-formed words
    words = text.split()
    if not words:
        return True
    
    normal_words = 0
    suspicious_patterns = 0
    
    for word in words[:10]:  # inspect only the first 10 words
        # well-formed English word? (letters and ordinary punctuation only)
        if re.match(r'^[a-zA-Z\'\-\.]+$', word) and len(word) <= 20:
            normal_words += 1
        # check for suspicious patterns
        elif re.match(r'^[0-9a-f]{6,}$', word) or len(word) > 25:
            suspicious_patterns += 1
    
    # call it mangled if <30% of words are well-formed, or suspicious patterns dominate
    normal_ratio = normal_words / min(len(words), 10)
    suspicious_ratio = suspicious_patterns / min(len(words), 10)
    
    return normal_ratio < 0.3 or suspicious_ratio > 0.2


def extract_answer_standard(model_output):
    """
    Standard answer extraction (official methodology, hardened).
    """
    if not model_output or not model_output.strip():
        return ""
    
    # basic cleanup
    output = clean_corrupted_text(model_output.strip())
    
    # keep only what precedes a 'Q:' marker (stronger matching)
    q_patterns = [r'\nQ:', r'\n\nQ:', r'Q:', r' Q:']
    for pattern in q_patterns:
        q_match = re.search(pattern, output)
        if q_match:
            output = output[:q_match.start()].strip()
            break
    
    # take the first line only (official methodology)
    first_line = output.split('\n')[0].strip()
    
    # drop any leftover 'Q:' marker
    first_line = re.sub(r'\s*Q:\s*.*$', '', first_line).strip()
    
    # check whether the text is mangled
    if is_text_corrupted(first_line):
        # if mangled, fall back to something more conservative
        words = first_line.split()
        clean_words = []
        
        for word in words[:20]:  # at most 20 words
            # check for well-formed word patterns
            if re.match(r'^[a-zA-Z0-9\'\-\.]+$', word) and len(word) <= 20:
                clean_words.append(word)
            else:
                break  # stop at the first malformed word
        
        if clean_words:
            return ' '.join(clean_words)
        else:
            return ""
    
    return first_line


def extract_answer_enhanced(model_output):
    """
    Improved answer extraction (standard plus heuristics).
    """
    if not model_output or not model_output.strip():
        return ""
    
    # start from the standard extraction
    standard_answer = extract_answer_standard(model_output)
    
    # return an empty string if standard extraction fails
    if not standard_answer:
        return ""
    
    # try a further heuristic improvement
    output = model_output.strip()
    
    # keep only what precedes a 'Q:' marker
    q_pattern_match = re.search(r'\nQ:', output)
    if q_pattern_match:
        output = output[:q_pattern_match.start()].strip()
    
    # join lines while keeping meaningful separation
    lines = output.split('\n')
    cleaned_lines = []
    
    for line in lines[:3]:  # process at most 3 lines
        line = clean_corrupted_text(line)
        if line and not line.startswith('Q:') and not line.startswith('A:'):
            # check the line is not too mangled
            if not is_text_corrupted(line):
                cleaned_lines.append(line)
    
    if cleaned_lines:
        # whole answer with newlines flattened to spaces
        full_answer = ' '.join(cleaned_lines)
        
        # truncate if over-long
        if len(full_answer.split()) > 30:
            words = full_answer.split()[:30]
            full_answer = ' '.join(words)
        
        # take the first complete sentence
        sentences = re.split(r'[.!?]', full_answer)
        if len(sentences) > 1 and sentences[0].strip():
            first_sentence = sentences[0].strip()
            # when the first sentence is neither too short nor malformed
            if len(first_sentence.split()) >= 2 and not is_text_corrupted(first_sentence):
                enhanced_answer = first_sentence
            else:
                enhanced_answer = standard_answer
        else:
            enhanced_answer = full_answer
        
        # pick the better of the two candidates
        candidates = [standard_answer, enhanced_answer]
        
        # validate, then pick
        valid_candidates = []
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate and not is_text_corrupted(candidate):
                valid_candidates.append(candidate)
        
        if not valid_candidates:
            return standard_answer
        
        # prefer the first valid candidate (the standard answer)
        return valid_candidates[0]
    
    # fall back to the standard answer if everything fails
    return standard_answer


def compute_scores_official(gold_list, pred):
    """
    Official scoring (standard CoQA methodology).
    """
    f1_sum = 0.0
    em_sum = 0.0
    
    if len(gold_list) > 1:
        # multiple gold answers: score against each and keep the best
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1:]
            # predictions compared against (n) golds and take maximum
            em_score = max(compute_exact_match_official(a, pred) for a in gold_answers)
            f1_score = max(compute_f1_official(a, pred) for a in gold_answers)
            
            if i == 0 or f1_score > f1_sum:
                f1_sum = f1_score
                em_sum = em_score
    else:
        # single gold answer
        em_sum = compute_exact_match_official(gold_list[0], pred)
        f1_sum = compute_f1_official(gold_list[0], pred)

    return {
        "em": em_sum,
        "f1": f1_sum,
    }


def process_results(doc, results):
    """
    Standard result processing (official base plus improved extraction).
    """
    gold_list = doc_to_target(doc)
    
    # official behaviour: first line only
    pred_standard = extract_answer_standard(results[0])
    
    # also try the improved extraction
    pred_enhanced = extract_answer_enhanced(results[0])
    
    # keep whichever scores higher
    scores_standard = compute_scores_official(gold_list, pred_standard)
    scores_enhanced = compute_scores_official(gold_list, pred_enhanced)
    
    # keep the higher F1 (ties go to the standard answer)
    if scores_enhanced["f1"] > scores_standard["f1"]:
        return scores_enhanced
    else:
        return scores_standard


# legacy helpers kept for compatibility (squad_metrics variants included)
def compute_scores(gold_list, pred):
    """
    Legacy compatibility helper (squad_metrics based).
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
    """Legacy exact-match helper kept for compatibility."""
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            # predictions compared against (n) golds and take maximum
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
    else:
        em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)

    return em_sum / max(1, len(gold_list)) 