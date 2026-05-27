#################### IMPORTS ####################

import streamlit as st

# Utilities
import os
import string
import time
import difflib
import datetime
from textwrap import dedent

# Image processing and data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# Animation
import matplotlib.animation as animation

# Sentence selection
import nltk
from nltk.corpus import gutenberg, brown
from nltk.tokenize import PunktTokenizer
import random

#################### CACHED DATA ####################

@st.cache_data(show_spinner = "Loading images... (estimated time: 1 second)")
def load_images():
    characters = list(string.ascii_lowercase) + [str(i) for i in range(10)]
    char_sign_dict = {}
    for char in characters:
        sign_array = np.array(Image.open(f"asl_signs/{char}.png").convert("L"))
        binary_array = np.where(sign_array > 1, 0, 255)
        char_sign_dict[char] = binary_array

    MAX_SIZE = (512, 512)
    def pad_to_max(array, max_size = MAX_SIZE, color = 255):
        """Pad array to max_size (width, height) with white background."""
        h, w = array.shape
        canvas = np.full((max_size[1], max_size[0]), color, dtype=np.uint8)
        y_off = (max_size[1] - h) // 2
        x_off = (max_size[0] - w) // 2
        canvas[y_off:y_off+h, x_off:x_off+w] = array
        return canvas

    def make_countdown_frame(number, size = MAX_SIZE):
        """Create a white image with a big number in the center."""
        image = Image.new("L", size, color = 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 400)
        draw.text((140, 20), str(number), fill = 0, font = font)
        return np.array(image)
    
    char_sign_dict_resized = {char: pad_to_max(char_sign_dict[char]) for char in char_sign_dict.keys()}
    countdown_arrays = [make_countdown_frame(num) for num in range(3, 0, -1)]
    white_array = np.full((MAX_SIZE[1], MAX_SIZE[0]), 255, dtype = np.uint8)
    black_array = np.full((MAX_SIZE[1], MAX_SIZE[0]), 0, dtype = np.uint8)
    return char_sign_dict_resized, countdown_arrays, white_array, black_array

@st.cache_data(show_spinner = "Loading all sentences... (estimated time: 10 seconds)")
def load_sentences():
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("gutenberg")
    nltk.download("brown")

    clean_sentences = {}

    for fileid in gutenberg.fileids():
        sentences = gutenberg.sents(fileid)
        clean_sentences[fileid] = []
        for sentence in sentences:
            cleaned_sentence = clean_and_filter_sentence(sentence)
            if cleaned_sentence is not None:
                clean_sentences[fileid].append(cleaned_sentence)

    for category in brown.categories():
        sentences = brown.sents(categories = category)
        clean_sentences[category] = []
        for sentence in sentences:
            cleaned_sentence = clean_and_filter_sentence(sentence)
            if cleaned_sentence is not None:
                clean_sentences[category].append(cleaned_sentence)

    return clean_sentences

#################### HELPER FUNCTIONS ####################

def clean_and_filter_sentence(sentence):
    cleaned_sentence = " ".join(ch for ch in sentence if ch.isalpha() or ch == " " or ch == "'")
    cleaned_sentence = cleaned_sentence.replace(" ' ", "").lower()
    length = len(cleaned_sentence)
    if length >= 30 and length <= 90:
        return cleaned_sentence
    else:
        return None

def calculate_diff_score(s1, s2):
    "S1 is the input, S2 is the actual sentence"
    s1 = s1.lower()
    s2 = s2.lower()

    diff = list(difflib.ndiff(s1, s2))
    insertions = deletions = replacements = 0
    
    i = 0
    while i < len(diff):
        d = diff[i]
        if d.startswith("- "):
            if i+1 < len(diff) and diff[i+1].startswith("+ "):
                replacements += 1
                i += 2
                continue
            else:
                deletions += 1
        elif d.startswith("+ "):
            insertions += 1
        i += 1
    
    total_changes = insertions + deletions + replacements
    score = round((1 - total_changes / len(s2)) * 100, 2)
    return {
        "insertions": insertions,
        "deletions": deletions,
        "replacements": replacements,
        "total_changes": total_changes,
        "score": score,
        "input_sentence": s1,
        "actual_sentence": s2 
    }

def visualize_diff(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return None
    diff = difflib.ndiff(s1, s2)
    html = ""
    for d in diff:
        code = d[0]
        char = d[2:]
        if code == " ":   # unchanged
            html += f"<span style='color:gray'>{char}</span>"
        elif code == "-": # deletion
            html += f"<span style='color:red; text-decoration:line-through'>{char}</span>"
        elif code == "+": # insertion
            html += f"<span style='color:green; font-weight:bold'>{char}</span>"
    return f"<div style='background-color:white; padding:10px; border-radius:5px'>{html}</div>"

def split_sentence(sentence):
    if len(sentence) < 50: return sentence
    words = sentence.split(" ")
    word_lengths = [len(word) for word in words]
    word_lengths_cumsum = np.cumsum(word_lengths) + np.arange(0, len(words))
    index = np.argmax(word_lengths_cumsum > 50)
    words_before_index = words[:index]
    words_after_index = words[index:]
    return " ".join(words_before_index) + "\n" + " ".join(words_after_index)

def create_html_animation(sentence, category_readable, question_number, interval = 500):
    frames = [char_sign_dict_resized.get(char, white_array) for char in sentence]

    fig, ax = plt.subplots(figsize = (6, 7.5))
    im = ax.imshow(frames[0], cmap = "binary_r")
    ax.axis("off")

    # Update function
    sentence_split = split_sentence(sentence)
    def update(frame_idx):
        question_title_part_1 = rf"$\mathbf{{QUESTION\ {question_number}}}$"
        question_title_part_2 = f"(sourced from {category_readable})"
        if len(question_title_part_1) + len(question_title_part_2) > 60:
            question_title = rf"""{question_title_part_1}
{question_title_part_2}"""
        else:
            question_title = rf"{question_title_part_1} {question_title_part_2}"

        if sentence_split[frame_idx] == " ":
            sentence_title = rf"""{sentence_split[:frame_idx]} {sentence_split[frame_idx+1:]}"""
        elif sentence_split[frame_idx] == "\n":
            sentence_title = rf"""{sentence_split[:frame_idx]}
{sentence_split[frame_idx+1:]}"""
        else:
            sentence_title = rf"""{sentence_split[:frame_idx]}$\mathbf{{{sentence_split[frame_idx]}}}${sentence_split[frame_idx+1:]}"""
        title = rf"""{question_title}
{sentence_title}"""
        ax.set_title(title)
        im.set_array(frames[frame_idx])
        return [im]

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames = len(frames), interval = interval, blit = True, repeat = True
    )
    plt.close(fig)  # prevent duplicate static plot
    
    return ani

st.set_page_config(layout = "wide")
st.title("ASL Fingerspelling Tester")
char_sign_dict_resized, countdown_arrays, white_array, black_array = load_images()
clean_sentences = load_sentences()

#################### INPUT ####################

# Session State for Inputs
if "num_questions" not in st.session_state:
    st.session_state["num_questions"] = 0
if "num_frames_per_second" not in st.session_state:
    st.session_state["num_frames_per_second"] = None
if "user_selected_categories_readable" not in st.session_state:
    st.session_state["user_selected_categories_readable"] = []
if "user_selected_categories_slug" not in st.session_state:
    st.session_state["user_selected_categories_slug"] = []

with st.sidebar: 
    st.header("Instructions for Use")
    st.write(dedent("""
    1. Input the number of questions, speed, and the categories/works to include
    2. Click 'Submit' and wait for the the questions to generate
    3. Whenever you are ready, click the 'Start' button
    4. You will have three seconds to prepare before the signs start showing
    5. Click the 'Enter' button (on your keyboard) to submit your answer
    6. When you are done answering, click 'Check' to see results. You can also see:
    * the animation of each letter in the sentence
    * the difference between what you typed and the correct sentence"""))

col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
with col1:
    num_questions = st.slider("Number of Questions", min_value = 1, max_value = 20, value = 5, key = "slider")
with col2:
    num_frames_per_second = st.slider(
        "Animation Speed", min_value = 0.1, max_value = 5.0, value = 2.0, step = 0.1,
        help = "This refers to the number of frames per second. The default is 2. Lower values means the frames move slower, and higher values means the frame moves faster."
    )

readable_to_slug_gutenberg = {
    "Emma by Jane Austen": "austen-emma.txt",
    "Persuasion by Jane Austen": "austen-persuasion.txt",
    "Sense and Sensibility by Jane Austen": "austen-sense.txt",
    "King James Bible (KJV)": "bible-kjv.txt",  # collection
    "Poems by William Blake": "blake-poems.txt",  # anthology
    "Stories by William Cullen Bryant": "bryant-stories.txt",  # collection
    "Buster Brown by Thornton Burgess": "burgess-busterbrown.txt",
    "Alice's Adventures in Wonderland by Lewis Carroll": "carroll-alice.txt",
    "The Ball and the Cross by G.K. Chesterton": "chesterton-ball.txt",
    "The Wisdom of Father Brown by G.K. Chesterton": "chesterton-brown.txt",
    "The Man Who Was Thursday by G.K. Chesterton": "chesterton-thursday.txt",
    "Parents' Assistant by Maria Edgeworth": "edgeworth-parents.txt",
    "Moby Dick by Herman Melville": "melville-moby_dick.txt",
    "Paradise Lost by John Milton": "milton-paradise.txt",
    "Julius Caesar by William Shakespeare": "shakespeare-caesar.txt",
    "Hamlet by William Shakespeare": "shakespeare-hamlet.txt",
    "Macbeth by William Shakespeare": "shakespeare-macbeth.txt",
    "Leaves of Grass by Walt Whitman": "whitman-leaves.txt"
}
readable_to_slug_brown = {
    category.replace("_", " ").title(): category for category in brown.categories()
}
readable_to_slug_all = {**readable_to_slug_gutenberg, **readable_to_slug_brown}
slug_to_readable_all = {slug: readable for readable, slug in readable_to_slug_all.items()}
readable_categories = list(readable_to_slug_all.keys())

user_selected_categories_readable = st.multiselect("Categories/Works to Include", readable_categories, default = readable_categories)
user_selected_categories_slug = [readable_to_slug_all[readable] for readable in user_selected_categories_readable]

#################### PROCESSING BASED ON INPUT ####################

def select_sentences(clean_sentences, user_selected_categories):
    if "selected_sentences" not in st.session_state or st.session_state["selected_sentences"] == []:
        category_sentence_pool = [(category, sentence) for category, sentences in clean_sentences.items() for sentence in sentences
                                  if category in user_selected_categories]
        selected_categories_sentences = [random.choice(category_sentence_pool) for _ in range(num_questions)]
        selected_categories = [category_sentence[0] for category_sentence in selected_categories_sentences]
        selected_sentences = [category_sentence[1] for category_sentence in selected_categories_sentences]
        st.session_state["selected_categories"] = selected_categories
        st.session_state["selected_sentences"] = selected_sentences
    return True

def save_animation(index, countdown_arrays, interval = 500, repetitions = 2):
    sentence = st.session_state["selected_sentences"][index]

    countdown_durations = [1000] * 3
    sentence_arrays = ([char_sign_dict_resized.get(char, white_array) for char in sentence] + [black_array]) * repetitions
    sentence_durations = [interval] * (len(sentence_arrays) + 1) * repetitions

    all_arrays = list(countdown_arrays) + list(sentence_arrays)
    all_durations = countdown_durations + sentence_durations
    os.makedirs("animations/", exist_ok = True)
    imageio.mimsave(f"animations/animation_{index}.gif", all_arrays, duration = all_durations, loop = 1)

@st.cache_data(show_spinner = f"Loading current set of questions... (estimated time: {num_questions * 2} seconds)")
def save_all_animations(num_questions, num_frames_per_second):
    for i in range(num_questions):
        save_animation(i, countdown_arrays, interval = 1000/num_frames_per_second)
    return True

def setup_game(num_questions, num_frames_per_second, user_selected_categories_slug):
    select_sentences(clean_sentences, user_selected_categories_slug)
    save_all_animations(num_questions, num_frames_per_second)
    html_animations = []
    for i in range(num_questions):
        actual_sentence = st.session_state["selected_sentences"][i]
        category_slug = st.session_state["selected_categories"][i]
        category_readable = slug_to_readable_all[category_slug]
        html_animation = create_html_animation(actual_sentence, category_readable, i+1)
        html_animations.append(html_animation)

    # Session State
    if "answered_results" not in st.session_state:
        st.session_state["answered_results"] = []
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = 1
    if "check" not in st.session_state:
        st.session_state["check"] = False
    for i in range(num_questions):
        if f"start_time_{i+1}" not in st.session_state:
            st.session_state[f"start_time_{i+1}"] = None
        if f"input_sentence_{i+1}" not in st.session_state:
            st.session_state[f"input_sentence_{i+1}"] = None

    for i in range(num_questions):
        col1, col2 = st.columns([0.8, 0.2])
        with col1: st.header(f"Question {i+1}")
        with col2: start_button = st.button("Start", key = f"start_button_{i+1}")

        input_sentence = st.text_input("Type your answer here and click the Enter button on your keyboard", key = f"input_{i+1}")

        if start_button:
            if st.session_state["current_question"] <= i:
                st.info(f"Please answer Question {st.session_state['current_question']} first.")
            else:
                st.session_state[f"start_time_{i+1}"] = time.time()
        
        if st.session_state[f"start_time_{i+1}"] and st.session_state["current_question"] >= i:
            st.image(f"animations/animation_{i}.gif")
    
        if (input_sentence and input_sentence != st.session_state[f"input_sentence_{i+1}"]):
            st.session_state[f"input_sentence_{i+1}"] = input_sentence
            
            if st.session_state[f"start_time_{i+1}"]:
                end = time.time()
                start = st.session_state[f"start_time_{i+1}"]
                elapsed_time = round(end - start - 3, 2)
                st.write(f"Time taken: <b>{elapsed_time}</b> seconds")
                actual_sentence = st.session_state["selected_sentences"][i]
                st.session_state["answered_results"].append(
                    {
                        "question": i+1,
                        "time": elapsed_time,
                        **calculate_diff_score(input_sentence, actual_sentence),
                        "num_characters": len(actual_sentence.replace(" ", "")),
                        "num_characters_per_second": len(actual_sentence.replace(" ", "")) / elapsed_time
                    }
                )

                st.session_state["current_question"] += 1

        st.write("---")

    if st.button("Check!"):
        st.session_state["check"] = True

        st.header("Your Results")

        # Answered questions
        if len(st.session_state["answered_results"]) > 0:
            answered_df = pd.DataFrame(st.session_state["answered_results"])
            answered_questions = set(answered_df["question"])
        else:
            answered_df = pd.DataFrame(columns = ["question", "time", "insertions", "deletions", "replacements", "total_changes", "score", 
                                                  "input_sentence", "actual_sentence", "num_characters", "num_characters_per_second"])
            answered_questions = set()

        # Unanswered questions
        unanswered_questions = set(range(1, num_questions + 1)).difference(answered_questions)
        unanswered_results = []
        for question in unanswered_questions:
            unanswered_result = {
                col: np.nan for col in answered_df.columns
            }
            unanswered_result.update({"actual_sentence": st.session_state["selected_sentences"][question-1]})
            unanswered_result.update({"question": question})
            unanswered_results.append(unanswered_result)
        unanswered_df = pd.DataFrame(unanswered_results)

        # All questions
        all_df = pd.concat([answered_df, unanswered_df])
        all_df = all_df.sort_values(by = "question").reset_index(drop = True)
        all_df["speed"] = [st.session_state["num_frames_per_second"]] * all_df.shape[0]

        # Display dataframe
        all_df_display = all_df.copy()
        all_df_display.columns = ["Question", "Time", "Insertions", "Deletions", "Replacements", "Total Changes", "Score", "Input Sentence", "Actual Sentence", 
                                  "Number of Characters", "Number of Characters per Second", "Speed"]
        for col in ["Insertions", "Deletions", "Replacements", "Total Changes"]:
            all_df_display[col] = all_df_display[col].astype("Int32")
        all_df_display.index = all_df_display["Question"] 
        all_df_display.drop(columns = ["Question"], inplace = True)
        st.dataframe(all_df_display)

        # Statistics and download
        mean_time = all_df["time"].mean()
        if ~np.isnan(mean_time): 
            st.write(f"Mean Time: {mean_time:.2f}")
        mean_score = all_df["score"].mean()
        if ~np.isnan(mean_score):
            st.write(f"Mean Score: {mean_score:.2f}")
        now = datetime.datetime.now().strftime(format = "%Y%m%d_%H%M%S")
        st.download_button(data = all_df_display.to_csv(), label = "Download your results", file_name = f"asl_spelling_{now}.csv", mime = "text/csv")

        st.write("---")
        st.header("Answer Key") 
        with st.expander("How to use the animations"):
            st.info(dedent(dedent("""
            * -/+   : Increase or decrease the speed (default 2 frames per second)
            * ⏮/⏭  : Move to the start/end of the animation
            * |◀/▶|: Move one frame backward/forward
            * ◀/▶  : Play the animation backward/forward
            * ⏸     : Pause the animation
            * The radio buttons at the bottom refer to the playing of the animation once, on loop, or reflect (forward, then backward), in that order. 
            If you are in dark mode, you may not be able to see the radio buttons clearly.""")))
        for i in range(num_questions):
            input_sentence = st.session_state[f"input_sentence_{i+1}"]
            actual_sentence = st.session_state["selected_sentences"][i]
            st.components.v1.html(html_animations[i].to_jshtml(), height = 900)
            if input_sentence is None:
                st.error("No input provided. Diff cannot be done.")
            else:
                st.markdown(visualize_diff(input_sentence, actual_sentence), unsafe_allow_html = True)

#################### SUBMISSION AND NEW GAME ####################

with col3:
    submit_button = st.button("Submit", key = "submit")

if submit_button:
    st.session_state["num_questions"] = num_questions
    st.session_state["num_frames_per_second"] = num_frames_per_second
    st.session_state["user_selected_categories_readable"] = user_selected_categories_readable
    st.session_state["user_selected_categories_slug"] = user_selected_categories_slug

complete_inputs = st.session_state["num_questions"] > 0 and \
    st.session_state["num_frames_per_second"] is not None and \
    len(st.session_state["user_selected_categories_readable"]) > 0 and \
    len(st.session_state["user_selected_categories_slug"]) > 0

if complete_inputs:
    setup_game(st.session_state["num_questions"], st.session_state["num_frames_per_second"], st.session_state["user_selected_categories_slug"])

    if st.session_state["check"]:

        if st.button("New Game"):
            # Reset session state for user input
            st.session_state["num_questions"] = 0
            st.session_state["num_frames_per_second"] = None
            st.session_state["selected_categories"] = []
            st.session_state["selected_sentences"] = []

            # Reset session state for other variables
            st.session_state["answered_results"] = []
            st.session_state["current_question"] = 1
            st.session_state["check"] = False
            for i in range(20): # Maximum number of questions
                st.session_state[f"start_time_{i+1}"] = None
                st.session_state[f"input_sentence_{i+1}"] = ""

            # Remove saved animations
            save_all_animations.clear()

            st.info("Scroll back to the top of the page, set the number of questions, then click submit again")
