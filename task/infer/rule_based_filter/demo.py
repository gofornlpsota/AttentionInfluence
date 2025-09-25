import spacy
import textstat
import nltk
import re
import json
import math
from collections import Counter
from tqdm import tqdm

from util.data import load_data_from_file, save_data_to_json

# --- GLOBAL RESOURCES & CONSTANTS ---

# Load models and resources once to avoid reloading on every call
try:
    NLP = spacy.load("en_core_web_sm")
    ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    ENGLISH_WORDS = set(nltk.corpus.words.words())
except Exception as e:
    print("Error loading NLP models. Please ensure you have run the setup instructions.")
    print(e)
    exit()

# Keyword sets for reasoning module
PROCEDURAL_WORDS = {"first", "second", "third", "next", "then", "finally", "firstly", "secondly", "step", "after", "conclusion"}
CAUSAL_WORDS = {"because", "therefore", "hence", "thus", "consequently", "result", "if", "unless", "however", "although", "since", "due"}
# NOTE: This is a very small, illustrative list. A real system would use a much larger blocklist.
TOXIC_WORDS = {"idiot", "stupid", "hate", "kill", "damn", "hell"}

# --- MODULE A: QUALITY & READABILITY ---

def calculate_readability(text):
    """Calculates Flesch Reading Ease score, normalized to [0, 1]."""
    score = textstat.flesch_reading_ease(text)
    # Normalize score. Flesch score range is ~0-100. We'll clip and scale.
    return max(0, min(1, score / 100.0))

def calculate_ttr(tokens):
    """Calculates Type-Token Ratio. Returns 0 for no tokens."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def calculate_grammar_score(tokens):
    """
    A heuristic for grammar/spelling. Calculates the ratio of words
    found in a standard English dictionary.
    """
    if not tokens:
        return 0.0
    # Lowercase for dictionary lookup
    lower_tokens = [t.lower() for t in tokens if t.isalpha()]
    if not lower_tokens:
        return 0.0
    
    correctly_spelled = sum(1 for t in lower_tokens if t in ENGLISH_WORDS)
    return correctly_spelled / len(lower_tokens)

# --- MODULE B: UTILITY & INFORMATION ---

def calculate_length_score(num_tokens, ideal_length=300, sigma=150):
    """
    Scores document length based on a Gaussian function centered at ideal_length.
    """
    return math.exp(-((num_tokens - ideal_length) ** 2) / (2 * sigma ** 2))

def calculate_info_density(tokens):
    """Calculates the ratio of non-stopword tokens."""
    if not tokens:
        return 0.0
    non_stopwords = [t for t in tokens if t.lower() not in ENGLISH_STOPWORDS]
    return len(non_stopwords) / len(tokens)

def calculate_ner_density(doc, num_tokens):
    """Calculates the density of Named Entities using spaCy."""
    if not num_tokens:
        return 0.0
    # Higher density might mean more factual content. Scale by 5 for a better score range.
    return min(1.0, (len(doc.ents) / num_tokens) * 5)

# --- MODULE C: REASONING DENSITY ---

def calculate_procedural_score(tokens):
    """Calculates density of procedural words."""
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t.lower() in PROCEDURAL_WORDS)
    # Scale score for better visibility, as these words are rare.
    return min(1.0, (count / len(tokens)) * 10)

def calculate_causal_score(tokens):
    """Calculates density of causal and logical connectors."""
    if not tokens:
        return 0.0
    count = sum(1 for t in tokens if t.lower() in CAUSAL_WORDS)
    return min(1.0, (count / len(tokens)) * 10)

def calculate_analytical_score(text, num_sentences):
    """Calculates density of questions."""
    if not num_sentences:
        return 0.0
    question_count = text.count('?')
    return min(1.0, question_count / num_sentences)

# --- MODULE D: SPECIAL CONTENT ---
def calculate_math_bonus(text):
    """
    Detects mathematical content using regex for equations, LaTeX, and keywords.
    """
    # Pattern for LaTeX style math: $...$, $$...$$, \[...\], \(...\)
    latex_pattern = r'(\$[^$]+\$|\$\$[^$]+\$\$|\\\[.*?\\\]|\\\(.*?\\\))'
    # Pattern for simple equations like x = y + z, a^2 + b^2 = c^2
    equation_pattern = r'\b[a-zA-Z]\s*=\s*[a-zA-Z0-9\s\+\-\*\/\^\.]+'
    # Keywords
    math_keywords = r'\b(equation|theorem|proof|calculate|variable|integral|derivative|matrix|vector|algebra|geometry)\b'
    
    latex_matches = len(re.findall(latex_pattern, text))
    equation_matches = len(re.findall(equation_pattern, text))
    keyword_matches = len(re.findall(math_keywords, text, re.IGNORECASE))
    
    # Give significant weight to LaTeX
    total_matches = (latex_matches * 5) + (equation_matches * 2) + keyword_matches
    
    score = (total_matches / len(text)) * 200 if len(text) > 0 else 0
    return min(1.0, score)

def calculate_code_bonus(text):
    """Detects code snippets using regex and keywords."""
    # Regex for common code patterns: function defs, loops, imports, HTML tags
    code_patterns = [
        r'\b(def|class|public|static|void|import|require)\b',
        r'[\{\};]',
        r'</?\w+>', 
        r'=\s*[\'"`].*?[\'"`]'
    ]
    matches = sum(len(re.findall(p, text)) for p in code_patterns)
    # Normalize by text length, scale for bonus
    score = (matches / len(text)) * 100 if len(text) > 0 else 0
    return min(1.0, score)

def calculate_instruction_bonus(text):
    """Detects instructional formats like "How to" and lists."""
    if re.search(r'^(how to|guide to|instructions for)', text, re.IGNORECASE):
        return 1.0
    # Check for numbered lists
    if len(re.findall(r'^\d+\.\s', text, re.MULTILINE)) > 2:
        return 0.8
    return 0.0

def calculate_toxicity_score(tokens):
    """
    Calculates a score based on the presence of toxic words.
    Returns 1.0 if no toxic words are found.
    """
    if not tokens: return 1.0
    toxic_count = sum(1 for token in tokens if token.lower() in TOXIC_WORDS)
    # Score decreases with the density of toxic words.
    return 1.0 - min(1.0, (toxic_count / len(tokens)) * 5)

def calculate_pii_score(text):
    """
    Detects PII like emails and phone numbers. Returns 0.0 if any PII is found.
    """
    # Regex for emails and common phone number formats
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    
    if re.search(email_pattern, text) or re.search(phone_pattern, text):
        return 0.0
    return 1.0

# --- MAIN SCORER FUNCTION ---

def QURI_scorer_detailed_english(document: str) -> dict:
    """
    Calculates a detailed breakdown of data quality scores for an English document.
    """
    # --- 1. Preprocessing ---
    if not isinstance(document, str) or not document.strip():
        # Return a zeroed-out structure for empty/invalid documents
        return { "final_score": 0.0, "error": "Input is not a valid string." }

    '''
    ValueError: [E088] Text of length 1027683 exceeds maximum of 1000000. The parser and NER models require roughly 1GB of temporary memory per 100,000 characters in the input. This means long texts may cause memory allocation errors. If you're not using the parser or NER, it's probably safe to increase the `nlp.max_length` limit. The limit is in number of characters, so you can check whether your inputs are too long by checking `len(text)`.
    '''
    trunced_document = document[:1000000]
    doc = NLP(trunced_document)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences = list(doc.sents)
    
    num_tokens = len(tokens)
    num_sentences = len(sentences)

    if num_tokens == 0:
        return { "final_score": 0.0, "error": "Document contains no usable tokens." }

    # --- 2. Score Calculation for each sub-metric ---
    scores = {
        "quality": {
            "readability": calculate_readability(document),
            "vocabulary_richness": calculate_ttr(tokens),
            "grammar": calculate_grammar_score(tokens)
        },
        "utility": {
            "length": calculate_length_score(num_tokens),
            "information_density": calculate_info_density(tokens),
            "named_entity_density": calculate_ner_density(doc, num_tokens)
        },
        "reasoning": {
            "procedural_words": calculate_procedural_score(tokens),
            "causal_and_logical_words": calculate_causal_score(tokens),
            "analytical_content": calculate_analytical_score(document, num_sentences)
        },
        "special_content": {"code_bonus": calculate_code_bonus(document), "instruction_bonus": calculate_instruction_bonus(document), "math_bonus": calculate_math_bonus(document)},
        "safety": {"toxicity": calculate_toxicity_score(tokens), "pii": calculate_pii_score(document)}
    }

    # --- 3. Aggregation ---
    quality_weights = {"readability": 0.4, "vocabulary_richness": 0.4, "grammar": 0.2}
    utility_weights = {"length": 0.3, "information_density": 0.5, "named_entity_density": 0.2}
    reasoning_weights = {"procedural_words": 0.5, "causal_and_logical_words": 0.4, "analytical_content": 0.1}
    special_weights = {"math_bonus": 0.333, "code_bonus": 0.333, "instruction_bonus": 0.333}
    safety_weights = {"toxicity": 0.7, "pii": 0.3}

    final_module_weights = {"quality": 0.25, "utility": 0.25, "reasoning": 0.40, "special_content": 0.10}

    # Calculate module scores
    scores["quality"]["module_score"] = sum(scores["quality"][k] * quality_weights[k] for k in quality_weights)
    scores["utility"]["module_score"] = sum(scores["utility"][k] * utility_weights[k] for k in utility_weights)
    scores["reasoning"]["module_score"] = sum(scores["reasoning"][k] * reasoning_weights[k] for k in reasoning_weights)
    scores["special_content"]["module_score"] = sum(scores["special_content"][k] * special_weights[k] for k in special_weights)
    scores["safety"]["module_score"] = sum(scores["safety"][k] * safety_weights[k] for k in safety_weights)
    # Calculate final score
    final_score = sum(scores[mod]["module_score"] * final_module_weights[mod] for mod in final_module_weights)
    scores["final_score"] = min(1.0, final_score)

    return scores

def process_examples(examples, text_field = 'content_split'):
    for example in tqdm(examples):
        example['rbf_model_output'] = QURI_scorer_detailed_english(example[text_field])
    return examples

def process_file(input_path, output_path, text_field = 'content_split', limit_example_cnt = None):
    examples = load_data_from_file(input_path)
    if limit_example_cnt is not None:
        examples = examples[:limit_example_cnt]
    process_examples(examples, text_field = text_field)
    save_data_to_json(examples, output_path, pretty=True)

# --- Step 3: Test Cases ---
if __name__ == "__main__":
    # Define test cases as a list of tuples (name, text)
    # This format is cleaner and avoids syntax errors with large dictionaries.
    test_cases = [
        ("1. High-Quality Instructional Text", 
         """
        How to bake a classic apple pie. First, you need to prepare the dough. You will need flour, butter, and cold water. 
        Secondly, prepare the apple filling. Peel and slice six medium-sized apples. Because the apples can brown, 
        toss them with lemon juice. Consequently, they will stay fresh. After preparing the filling, roll out your dough. 
        Finally, assemble the pie and bake at 375°F for 45 minutes. Therefore, if you follow these steps, you will have a delicious pie.
        """),

        ("2. Low-Quality Spam Text", 
         """
        buy now!!!!!! cheap price best quality click here for free money. this is not a scam. free free free. win big today. 
        repetitive text repetitive text repetitive text. i like it it is good. wow.
        """),

        ("3. Technical Code Snippet", 
         """
        ```python
        import pandas as pd
        
        # A class for data processing
        class DataProcessor:
            def __init__(self, filepath):
                self.df = pd.read_csv(filepath)

            def process(self):
                # This function cleans the data.
                self.df.dropna(inplace=True)
                return self.df
        
        def main():
            processor = DataProcessor("data.csv")
            cleaned_df = processor.process()
            print(cleaned_df.head())
        ```
        """),

        ("4. Factual Encyclopedia Entry", 
         """
        The Apollo 11 mission was the first crewed mission to land on the Moon. The mission, carried out by NASA, 
        launched on July 16, 1969. It carried Commander Neil Armstrong, Command Module Pilot Michael Collins, and Lunar Module Pilot Buzz Aldrin. 
        On July 20, 1969, Armstrong and Aldrin became the first humans to land on another celestial body. The landing took place in the Sea of Tranquility.
        This event was a major victory for the United States in the Space Race.
        """),

        ("5. Short, Simple Sentence", 
         """
        The cat sat on the mat.
        """)
    ]

    # Iterate through the list to run the tests
    for name, text in test_cases:
        print(f"--- {name} ---")
        scores = QURI_scorer_detailed_english(text)
        # Use json.dumps for pretty printing the dictionary
        print(json.dumps(scores, indent=2))
        print("\n" + "="*50 + "\n")


    # preview test
    limit_example_cnt = 1000
    process_file(input_path = '7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.gz.parquet',
    output_path ='7B_dense_part-00000-30319b05-9eb9-4dad-bcb5-5bec535d4536-c000.added_ruled_based_filter.json',
    text_field = 'content_split', limit_example_cnt = limit_example_cnt)