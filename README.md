# Multi-class Perceptron

- Machine Learning, Programming Assignment 1
- Spring 2025

## Purpose

This project is about practicing

1. Python/ML Basics
2. Perceptrons
3. Handle Multi-class Classification using One Vs. Rest methodology
4. Running experiments and interpretting/graphing the results

## Overview

Create a binary classification Perceptron from scratch. Then using your Perceptron code, create a one vs. rest multiclass Perceptron, and train it on the provided data. Interpret your results and findings to outline opportunities for growth and learning using provided questions.

## Dataset

The dataset is a wine dataset that has chemical analysis of different wines, with the goal of classifying what cultivar the wine was derived from. It has 13 features, and the class has 3 values.

Note: two of the features are on a vastly different scale than the others.

## Requirements

Create your Perceptron (binary) and MulticlassPerceptron python files.

After creating your new multi-class perceptron code, you will need to use this code to try to classify these three classes. Your code must do the following:

- Use the data provided
- Split data into appropriate sets using the correct sklearn function
- Train your new multi-class Perceptron using the appopriate data. Output the number of errors that occurred on each epoch for each model learned as a graph, and output the final learned bias and weights of each model in your multi-class Perceptron.
- Test your new multi-class Perceptron using the appropriate data on the set of models that worked best based on accuracy.
- Output the final confusion matrix and the classification report for the test data.
- Outline interpretation of ML process for outlining growth in interpretation.md
