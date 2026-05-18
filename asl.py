import streamlit as st

# Utilities
import os
import string
import requests
import time
import difflib
from pprint import pprint

# Image processing and data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# Animation
import matplotlib.animation as animation

# Sentence selection
import nltk
from nltk.corpus import gutenberg, brown
import random

# Cache data
@st.cache_data
def load_images():
    characters = list(string.ascii_lowercase) + [str(i) for i in range(10)]
    char_sign_dict = {char: Image.open(f"asl_signs/{char}.png").convert("RGBA") for char in characters} #.resize(SIZE)
    all_sizes = [img.size for img in char_sign_dict.values()]
    max_w = max([s[0] for s in all_sizes])
    max_h = max([s[1] for s in all_sizes])
    # MAX_SIZE = (max_w, max_h)
    MAX_SIZE = (512, 512)

    def pad_to_max(img, max_size=MAX_SIZE, color=(255, 255, 255)):
        """Pad image to max_size (width, height) with white background."""
        w, h = img.size
        new_img = Image.new("RGB", max_size, color)
        offset = ((max_size[0] - w) // 2, (max_size[1] - h) // 2)
        new_img.paste(img, offset, mask = img)
        return new_img.convert("RGB")

    def make_countdown_frame(number, size=MAX_SIZE):
        """Create a white image with a big number in the center."""
        img = Image.new("RGB", size, color="white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 400)
        text = str(number)
        draw.text((140, 20), text, fill="black", font=font)
        return np.array(img)
    
    char_sign_dict_resized = {char: pad_to_max(char_sign_dict[char]) for char in char_sign_dict.keys()}
    countdown_images = [Image.fromarray(make_countdown_frame(num)) for num in range(3, 0, -1)]
    white_image = Image.new("RGB", MAX_SIZE, (255, 255, 255))
    black_image = Image.new("RGB", MAX_SIZE, (0, 0, 0))

    return char_sign_dict_resized, countdown_images, white_image, black_image

@st.cache_data
def load_sentences():
    sentences = gutenberg.sents() + brown.sents()

    clean_sentences = []
    for sentence in sentences:
        clean_sentence = " ".join(ch for ch in sentence if ch.isalpha() or ch == " " or ch == "'")
        clean_sentence = clean_sentence.replace(" ' ", "")
        length = len(clean_sentence)
        if length >= 30 and length <= 90:
            clean_sentences.append(clean_sentence)
    return clean_sentences

# Functions
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
        "input": s1,
        "actual": s2 
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

def create_html_animation(sentence, question_number, interval=500):
    sentence = sentence.lower()
    frames = [char_sign_dict_resized.get(char, white_image) for char in sentence]

    fig, ax = plt.subplots(figsize = (6, 6))
    im = ax.imshow(frames[0])
    ax.axis("off")

    # Update function
    sentence_split = split_sentence(sentence)
    def update(frame_idx):
        question_title = rf"$\mathbf{{QUESTION\ {question_number}}}$"
        if sentence_split[frame_idx] in [" ", "\n"]:
            sentence_title = rf"""{sentence_split[:frame_idx]} {sentence_split[frame_idx+1:]}"""
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
char_sign_dict_resized, countdown_images, white_image, black_image = load_images()
clean_sentences = load_sentences()
NUM_EXERCISES = 3

def create_test_sentences():
    if "test_sentences" not in st.session_state or st.session_state["test_sentences"] is None:
        st.session_state["test_sentences"] = [random.choice(clean_sentences) for _ in range(NUM_EXERCISES)]
    return True

def save_animation(index, countdown_images, interval = 500, repetitions = 2):
    sentence = st.session_state["test_sentences"][index].lower()

    countdown_durations = [1000] * 3
    sentence_images = ([char_sign_dict_resized.get(char, white_image) for char in sentence] + [black_image]) * repetitions
    sentence_durations = [interval] * (len(sentence_images) + 1) * repetitions

    all_images = countdown_images + sentence_images
    all_durations = countdown_durations + sentence_durations
    all_images[0].save(f"animations/animation_{index}.gif", save_all=True, append_images=all_images[1:], duration=all_durations, loop=1)

@st.cache_data
def save_all_animations():
    for i in range(NUM_EXERCISES):
        save_animation(i, countdown_images, interval = 700)
    return True

create_test_sentences()
save_all_animations()

# Session State
if "all_results" not in st.session_state:
    st.session_state["all_results"] = []
if "current_question" not in st.session_state:
    st.session_state["current_question"] = 0
for i in range(NUM_EXERCISES):
    if f"start_time_{i+1}" not in st.session_state:
        st.session_state[f"start_time_{i+1}"] = None
    if f"input_sentence_{i+1}" not in st.session_state:
        st.session_state[f"input_sentence_{i+1}"] = None



for i in range(NUM_EXERCISES):
    col1, col2 = st.columns([0.8, 0.2])
    with col1: st.header(f"Question {i+1}")
    with col2: start_button = st.button("Start", key = f"start_button_{i+1}")

    if start_button:
        st.session_state[f"start_time_{i+1}"] = time.time()

    if st.session_state[f"start_time_{i+1}"] and st.session_state["current_question"] >= i:
        st.image(f"animations/animation_{i}.gif")

    input_sentence = st.text_input("Type your answer here", key = f"input_{i+1}")

    if (input_sentence and input_sentence != st.session_state[f"input_sentence_{i+1}"]):
        st.session_state[f"input_sentence_{i+1}"] = input_sentence
        
        if st.session_state[f"start_time_{i+1}"]:
            end = time.time()
            start = st.session_state[f"start_time_{i+1}"]
            elapsed_time = round(end - start - 3, 2)
            st.write("Elapsed time", elapsed_time)
            st.session_state["all_results"].append(
                {
                    "time": elapsed_time,
                    **calculate_diff_score(input_sentence, st.session_state["test_sentences"][i])
                }
            )

            st.session_state["current_question"] += 1

    st.write("---")

if st.button("Check!"):

    st.header("Your Results")
    if st.session_state["all_results"] == []:
        df_results = pd.DataFrame()
        st.write("You did not attempt")
    else:
        df_results = pd.DataFrame(st.session_state["all_results"])
        st.write(df_results)
        st.write(f"Mean Time: {df_results['time'].mean().round(2)}")
        st.write(f"Mean Score: {df_results['score'].mean().round(2)}%")
        st.download_button(data = df_results.to_csv(), label = "Download your results", mime = "text/csv")

    st.write("---")

    st.header("Answer Key") 
    for i in range(NUM_EXERCISES):
        input_sentence = st.session_state[f"input_sentence_{i+1}"]
        test_sentence = st.session_state["test_sentences"][i]
        ani = create_html_animation(test_sentence, i+1)
        st.components.v1.html(ani.to_jshtml(), height = 700)
        if input_sentence is None:
            st.write("No input provided. Diff cannot be done.")
        else:
            st.markdown(visualize_diff(input_sentence, test_sentence), unsafe_allow_html=True)

if st.button("New Game"):
    # Reset session state
    st.session_state["all_results"] = []
    st.session_state["current_question"] = 0
    st.session_state["test_sentences"] = None

    for i in range(NUM_EXERCISES):
        st.session_state[f"start_time_{i}"] = None
        st.session_state[f"input_sentence_{i}"] = ""
        st.session_state[f"submit_clicked_{i}"] = False
        st.session_state[f"start_clicked_{i}"] = False

    # Regenerate questions and animations
    st.cache_data.clear()
    create_test_sentences()
    save_all_animations()
