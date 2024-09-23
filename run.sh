#!/bin/sh

echo "Training and Test errors for car dataset"
python3 decisionTree_car.py

echo "Training and Test errors for bank dataset"
python3 decisionTree_bank.py

echo "Training and Test errors for bank dataset with unknown replaced with most Common Values"
python3 decisionTree_bank_unknown_majority.py
