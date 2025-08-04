# SimpleFeedbackDemo

This project demonstrates a feedback analysis pipeline that processes and summarizes user feedback about a healthcare records management system. The system uses multiple Large Language Models (LLMs) to generate sample feedback and then analyze it.

## Overview

The system works in two stages:
1. First, it uses the prompt in `Generate comments prompt` with an LLM to generate a set of realistic user feedback comments about a healthcare records management system. This feedback is stored in `FeedbackData/RawFeedback.txt`.
2. Then, it processes this feedback using another LLM to generate a comprehensive summary of the user sentiments, common issues, and positive aspects of the system.

## Features

- Realistic feedback generation using a carefully crafted prompt
- Processing of large sets of feedback comments (100+ entries)
- Analysis of both positive and negative feedback
- Identification of common themes and issues
- Generation of actionable insights from user feedback

## Project Structure

- `Generate comments prompt`: Contains the prompt used to generate realistic feedback data
- `SimpleFeedbackDemo.py`: Main script that processes and analyzes the feedback
- `requirements.txt`: Python dependencies
- `FeedbackData/RawFeedback.txt`: Contains the generated feedback data

## How It Works

1. **Feedback Generation:**
   - The system uses the prompt from `Generate comments prompt`
   - An LLM processes this prompt to generate 100 realistic pieces of feedback
   - The feedback simulates user experiences with a healthcare records management system
   - Generated feedback is stored in `FeedbackData/RawFeedback.txt`

2. **Feedback Analysis:**
   - The main script reads the raw feedback data
   - Another LLM processes the feedback to identify patterns and themes
   - The system generates a comprehensive summary of user sentiments
   - Key issues and positive aspects are highlighted
   - Actionable insights are provided based on the analysis

## Getting Started

1. **Install Python dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the feedback generator:**
   ```sh
   python SimpleFeedbackDemo.py generate
   ```

3. **Analyze the feedback:**
   ```sh
   python SimpleFeedbackDemo.py analyze
   ```

## Notes

- The system is designed to process both positive and negative feedback
- The generated feedback aims to be realistic and varied in length
- The analysis provides actionable insights for system improvement

## License

MIT
