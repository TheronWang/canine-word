import os
import random
import json
import logging
import string
from collections import defaultdict
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


def clean_description(description: str) -> list[str]:
    """Cleans and tokenizes a description string into a list of words."""
    # Convert to lowercase
    description = description.lower()
    word_list = description.split()
    
    # Remove punctuation from each word
    translator = str.maketrans('', '', string.punctuation)
    word_list = [word.translate(translator) for word in word_list]
    # Remove empty strings after punctuation removal
    word_list = [word for word in word_list if word]
    
    # Unify the tense of words (lemmatization)
    lemmatizer = WordNetLemmatizer()
    word_list = [lemmatizer.lemmatize(word, pos='v') for word in word_list]  # Lemmatize verbs
    word_list = [lemmatizer.lemmatize(word, pos='n') for word in word_list]  # Lemmatize nouns
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    word_list = [word for word in word_list if word not in stop_words]
    
    # Remove meaningless words
    words_not_useful = {"dog", "dogs", "cat", "cats", "camera", "video", "something", "thing", "object", "objects",
                         "stuff", "background", "animal", "seems", "seem", "appear", "appears", "seen", "nearby", "lot",
                         "lots", "various", "varied", "multiple", "many", "possibly", "maybe", "perhaps", "could", "would", "should",
                         "also", "get", "got", "either", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                         "first", "second", "third", "next", "then", "way", "ways", "part", "parts", "area", "areas",
                         "still", "long", "short", "big", "small", "large", "tiny", "huge", "enough", "lot", "lots",
                         "kind", "kinds", "type", "types", "indicates", "indicate", "shows", "show", "display", "displays",
                         "picture", "image", "scene", "scenery", "view", "views", "landscape", "portrait", "someone", "yellow",
                         "red", "blue", "green", "black", "white", "gray", "colour", "color", "colors", "colours",
                         "likely", "possibly", "maybe", "perhaps", "guess", "seems", "seem", "appear", "appears", "behind",
                         "infront", "front", "back", "top", "bottom", "left", "right", "side", "sides", "language", "text", "words", "word",
                         "letters", "letter", "number", "numbers", "digit", "digits", "cause", "causes", "due", "effect", "effects",
                         "indicate", "indicates", "signify", "signifies", "symbol", "symbols", "mark", "marks", "label", "labels",
                         "tag", "tags", "caption", "captions", "may", "might", "could", "would", "should", "also", "too", "either",
                         "neither", "every", "each", "various", "varied", "multiple", "many", "whole", "entire", "complete", "full",
                         "general", "specific", "certain", "certainly", "definitely", "surely", "obviously", "clearly", "onto", "upon",
                         "brown", "gray", "grey", "orange", "purple", "pink", "cyan", "magenta", "beige", "maroon", "navy", "teal", "lavender", "turquoise",
                         "across", "about", "see", "new", "old", "young", "youngster", "child", "children", "kid", "kids", "teen", "teens",
                         "adult", "adults", "event", "events", "happen", "happens", "happened", "occur", "occurs", "occurred",
                         "always", "never", "often", "seldom", "rarely", "usually", "sometimes", "frequently", "infrequently",
                         "before", "after", "during", "while", "among", "between", "suddenly", "suddenly", "quickly", "slowly",
                         "sudden", "abrupt", "abruptly", "near", "far", "close", "closely", "distance", "distances", "another",
                         "noise", "noises", "sound", "sounds", "music", "musical", "melody", "melodies", "tune", "tunes", "happen", "happens",
                         "happened", "additionally", "furthermore", "moreover", "besides", "plus", "extra", "added", "addition", "additionally",
                         "over", "under", "above", "below", "near", "far", "next", "previous", "past", "situation", "situations",
                         "within", "without", "across", "along", "around", "throughout", "towards", "toward", "beside", "besides", "beyond"}
    word_list = [word for word in word_list if word not in words_not_useful]
    
    # Remove duplicates while preserving order
    seen = set()
    cleaned_words = []
    for word in word_list:
        if word not in seen:
            seen.add(word)
            cleaned_words.append(word)
    
    return cleaned_words


def process_labels(label_data: dict, output_folder: str):
    """Processes label data and saves cleaned labels to the output path."""
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "sentence_labels.json")
    processed_data = {}
    for audio_name in label_data:
        description = label_data[audio_name]
        cleaned_words = clean_description(description)
        processed_data[audio_name] = cleaned_words
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    logging.info(f"Processed labels saved to {output_path}")


def get_frequent_keywords(sentence_labels: str, top_k: int = 50) -> None:
    with open(sentence_labels, 'r', encoding='utf-8') as f:
        label_data = json.load(f)
    word_freq = defaultdict(int)
    for audio_name in label_data:
        words = label_data[audio_name]
        for word in words:
            word_freq[word] += 1
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    print(f"Top {top_k} frequent keywords:")
    for word, freq in sorted_words[:top_k]:
        print(f"{word}: {freq}")


def process_sentence_descriptions(annotation_path: str, output_folder: str):
    with open(annotation_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    label_data = label_data['result']
    print(f"Loaded {len(label_data)} label entries.")
    process_labels(label_data, output_folder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers(dest="cmd")

    p1 = subp.add_parser("process", help="Process video annotations into keyword labels")
    p1.add_argument("--annotations", required=True, help="Path to video annotation JSON")
    p1.add_argument("--output",      required=True, help="Output folder")

    p2 = subp.add_parser("freq", help="Print top-K frequent keywords")
    p2.add_argument("--labels", required=True, help="Path to sentence_labels.json")
    p2.add_argument("--top_k",  type=int, default=100)

    args = parser.parse_args()
    if args.cmd == "process":
        process_sentence_descriptions(args.annotations, args.output)
    elif args.cmd == "freq":
        get_frequent_keywords(sentence_labels=args.labels, top_k=args.top_k)
    else:
        parser.print_help()