---
title: Branchy Phi 2
emoji: ⚡
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Early Exit Computational Savings Demo

## Overview

This project demonstrates the concept of "early exiting" in deep learning models to save computational resources without significantly compromising on model performance. Early exit strategies allow a neural network to make predictions at intermediate layers for easy-to-classify instances, thus reducing the overall computation time and resources needed for inference. The application is built to run with Streamlit, offering an interactive web interface to explore the functionalities of the early exit model.

## Features

- **BranchyModel:** An implementation of a deep learning model with early exit points. This model architecture is designed to evaluate the performance and computational savings of using early exits.
- **Utility Functions:** A set of utilities to support the model's operation, including data preprocessing and performance evaluation metrics.
- **Streamlit Application:** A user-friendly web interface to interact with the model, visualize its performance, and understand the benefits of early exits.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed on your machine. You can install all the required dependencies via:

```bash
pip install -r requirements.txt
```

### Running the Application

To run the Streamlit application, execute the following command from the root of the project:

```bash
streamlit run app.py
```

The command will start a local web server and open the application in your default web browser, allowing you to interact with the BranchyModel and explore its features.

## Project Structure

- **app.py**: The main Streamlit application script.
- **requirements.txt**: Lists all the Python dependencies required by the project.
- **src/**: Contains the source code for the project.
