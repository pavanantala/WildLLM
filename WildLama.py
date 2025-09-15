def create_excel_mcc_analysis(all_results, output_dir):
    """Create Excel file with epoch-wise MCC analysis and graphs INCLUDING TEST RESULTS"""
    excel_path = os.path.join(output_dir, 'mcc_epoch_wise_analysis.xlsx')
    
    # Create a new workbook
    wb = openpyxl.Workbook()
    
    # Remove default sheet
    wb.remove(wb.active)
    
    # Create Sheet 1: Epoch-wise Data (INCLUDING TEST DATA)
    ws_data = wb.create_sheet("Epoch-wise MCC Data")
    
    # Headers for the data sheet (ADDED TEST COLUMNS)
    headers = ['Experiment_ID', 'Learning_Rate', 'Batch_Size', 'Epoch', 
               'Train_MCC', 'Val_MCC', 'Test_MCC',  # ADDED Test_MCC
               'Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy',  # ADDED Test_Accuracy
               'Train_F1', 'Val_F1', 'Test_F1',  # ADDED Test_F1
               'Train_AUC', 'Val_AUC', 'Test_AUC']  # ADDED Test_AUC
    
    # Write headers
    for col, header in enumerate(headers, 1):
        ws_data.cell(row=1, column=col, value=header)
    
    # Fill data
    row_idx = 2
    experiment_id = 1
    
    for result in all_results:
        lr = result['hyperparameters']['learning_rate']
        bs = result['hyperparameters']['batch_size']
        epochs = result['hyperparameters']['epochs']
        
        # Get epoch-wise data
        train_mcc_hist = result['epoch_wise_data']['train_mcc_history']
        val_mcc_hist = result['epoch_wise_data']['val_mcc_history']
        test_mcc_hist = result['epoch_wise_data']['test_mcc_history']  
        
        train_metrics_hist = result['epoch_wise_data']['train_metrics_history']
        val_metrics_hist = result['epoch_wise_data']['val_metrics_history']
        test_metrics_hist = result['epoch_wise_data']['test_metrics_history']  
        
        # Write data for each epoch
        for epoch in range(epochs):
            ws_data.cell(row=row_idx, column=1, value=f"Exp_{experiment_id}")
            ws_data.cell(row=row_idx, column=2, value=lr)
            ws_data.cell(row=row_idx, column=3, value=bs)
            ws_data.cell(row=row_idx, column=4, value=epoch + 1)
            
            # MCC values
            ws_data.cell(row=row_idx, column=5, value=round(train_mcc_hist[epoch], 4))
            ws_data.cell(row=row_idx, column=6, value=round(val_mcc_hist[epoch], 4))
            ws_data.cell(row=row_idx, column=7, value=round(test_mcc_hist[epoch], 4))  
            
            # Accuracy values
            ws_data.cell(row=row_idx, column=8, value=round(train_metrics_hist[epoch]['accuracy'], 4))
            ws_data.cell(row=row_idx, column=9, value=round(val_metrics_hist[epoch]['accuracy'], 4))
            ws_data.cell(row=row_idx, column=10, value=round(test_metrics_hist[epoch]['accuracy'], 4))  
            
            # F1 values
            ws_data.cell(row=row_idx, column=11, value=round(train_metrics_hist[epoch]['pos_f1'], 4))
            ws_data.cell(row=row_idx, column=12, value=round(val_metrics_hist[epoch]['pos_f1'], 4))
            ws_data.cell(row=row_idx, column=13, value=round(test_metrics_hist[epoch]['pos_f1'], 4))  
            
            # Handle AUC (might be None)
            train_auc = train_metrics_hist[epoch]['auc']
            val_auc = val_metrics_hist[epoch]['auc']
            test_auc = test_metrics_hist[epoch]['auc']  
            
            ws_data.cell(row=row_idx, column=14, value=round(train_auc, 4) if train_auc is not None else 0)
            ws_data.cell(row=row_idx, column=15, value=round(val_auc, 4) if val_auc is not None else 0)
            ws_data.cell(row=row_idx, column=16, value=round(test_auc, 4) if test_auc is not None else 0)  
            
            row_idx += 1
        
        experiment_id += 1
    
    # Auto-adjust column widths
    for column in ws_data.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 20)
        ws_data.column_dimensions[column_letter].width = adjusted_width
    
    # Create Sheet 2: Summary Statistics 
    ws_summary = wb.create_sheet("Summary Statistics")
    
    # Summary headers 
    summary_headers = ['Experiment_ID', 'Learning_Rate', 'Batch_Size', 'Epoch',
                      'Best_Epoch_Train_MCC', 'Best_Epoch_Val_MCC', 'Best_Epoch_Test_MCC',  # ADDED Best_Epoch_Test_MCC
                      'Train_MCC', 'Val_MCC', 'Test_MCC',  # EPOCH-WISE VALUES
                      'Train_Accuracy', 'Val_Accuracy', 'Test_Accuracy',  # EPOCH-WISE VALUES
                      'Train_F1', 'Val_F1', 'Test_F1',  # EPOCH-WISE VALUES
                      'Train_AUC', 'Val_AUC', 'Test_AUC',  # EPOCH-WISE VALUES
                      'Config_Name']
    
    for col, header in enumerate(summary_headers, 1):
        ws_summary.cell(row=1, column=col, value=header)
    
    # Fill summary data (ALL EPOCHS FOR ALL EXPERIMENTS)
    row_idx = 2
    for exp_idx, result in enumerate(all_results, 1):
        lr = result['hyperparameters']['learning_rate']
        bs = result['hyperparameters']['batch_size']
        epochs = result['hyperparameters']['epochs']
        config_name = f"lr_{lr}_bs_{bs}"
        
        # Find best epochs for train, val, and test MCC
        train_mcc_hist = result['epoch_wise_data']['train_mcc_history']
        val_mcc_hist = result['epoch_wise_data']['val_mcc_history']
        test_mcc_hist = result['epoch_wise_data']['test_mcc_history'] 
        
        best_train_epoch = train_mcc_hist.index(max(train_mcc_hist)) + 1
        best_val_epoch = val_mcc_hist.index(max(val_mcc_hist)) + 1
        best_test_epoch = test_mcc_hist.index(max(test_mcc_hist)) + 1 
        
        # Get epoch-wise data
        train_metrics_hist = result['epoch_wise_data']['train_metrics_history']
        val_metrics_hist = result['epoch_wise_data']['val_metrics_history']
        test_metrics_hist = result['epoch_wise_data']['test_metrics_history'] 
        
        # Write data for each epoch of this experiment
        for epoch in range(epochs):
            ws_summary.cell(row=row_idx, column=1, value=f"Exp_{exp_idx}")
            ws_summary.cell(row=row_idx, column=2, value=lr)
            ws_summary.cell(row=row_idx, column=3, value=bs)
            ws_summary.cell(row=row_idx, column=4, value=epoch + 1)  # Current epoch
            
            # Best epochs 
            ws_summary.cell(row=row_idx, column=5, value=best_train_epoch)
            ws_summary.cell(row=row_idx, column=6, value=best_val_epoch)
            ws_summary.cell(row=row_idx, column=7, value=best_test_epoch)  
            
            # Current epoch values
            ws_summary.cell(row=row_idx, column=8, value=round(train_mcc_hist[epoch], 4))
            ws_summary.cell(row=row_idx, column=9, value=round(val_mcc_hist[epoch], 4))
            ws_summary.cell(row=row_idx, column=10, value=round(test_mcc_hist[epoch], 4))  
            
            ws_summary.cell(row=row_idx, column=11, value=round(train_metrics_hist[epoch]['accuracy'], 4))
            ws_summary.cell(row=row_idx, column=12, value=round(val_metrics_hist[epoch]['accuracy'], 4))
            ws_summary.cell(row=row_idx, column=13, value=round(test_metrics_hist[epoch]['accuracy'], 4)) 
            
            ws_summary.cell(row=row_idx, column=14, value=round(train_metrics_hist[epoch]['pos_f1'], 4))
            ws_summary.cell(row=row_idx, column=15, value=round(val_metrics_hist[epoch]['pos_f1'], 4))
            ws_summary.cell(row=row_idx, column=16, value=round(test_metrics_hist[epoch]['pos_f1'], 4))  
            
            # Handle AUC
            train_auc = train_metrics_hist[epoch]['auc']
            val_auc = val_metrics_hist[epoch]['auc']
            test_auc = test_metrics_hist[epoch]['auc']  
            
            ws_summary.cell(row=row_idx, column=17, value=round(train_auc, 4) if train_auc is not None else 0)
            ws_summary.cell(row=row_idx, column=18, value=round(val_auc, 4) if val_auc is not None else 0)
            ws_summary.cell(row=row_idx, column=19, value=round(test_auc, 4) if test_auc is not None else 0)  
            
            ws_summary.cell(row=row_idx, column=20, value=config_name)  
            
            row_idx += 1
    
    # Auto-adjust column widths for summary
    for column in ws_summary.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 25)
        ws_summary.column_dimensions[column_letter].width = adjusted_width
    
    # Create Sheet 3: MCC Comparison Graphs 
    ws_graphs = wb.create_sheet("MCC Graphs")
    
    # Prepare data for graphs 
    graph_data_start_row = 2
    
    # Headers for graph data
    ws_graphs.cell(row=1, column=1, value="Epoch")
    col_idx = 2
    
    # Create columns for each experiment (Val, Test, and Train)
    experiment_labels = []
    for idx, result in enumerate(all_results):
        lr = result['hyperparameters']['learning_rate']
        bs = result['hyperparameters']['batch_size']
        val_label = f"Exp_{idx+1}_lr_{lr}_bs_{bs}_Val_MCC"
        test_label = f"Exp_{idx+1}_lr_{lr}_bs_{bs}_Test_MCC"  
        train_label = f"Exp_{idx+1}_lr_{lr}_bs_{bs}_Train_MCC"
        
        experiment_labels.extend([val_label, test_label, train_label])  # ADDED test_label
        ws_graphs.cell(row=1, column=col_idx, value=val_label)
        ws_graphs.cell(row=1, column=col_idx+1, value=test_label)  
        ws_graphs.cell(row=1, column=col_idx+2, value=train_label)  
        col_idx += 3  
    
    # Fill epoch data for graphs
    max_epochs = max([result['hyperparameters']['epochs'] for result in all_results])
    
    for epoch in range(1, max_epochs + 1):
        ws_graphs.cell(row=graph_data_start_row + epoch - 1, column=1, value=epoch)
        
        col_idx = 2
        for result in all_results:
            val_mcc_hist = result['epoch_wise_data']['val_mcc_history']
            test_mcc_hist = result['epoch_wise_data']['test_mcc_history'] 
            train_mcc_hist = result['epoch_wise_data']['train_mcc_history']
            
            # Validation MCC
            if epoch <= len(val_mcc_hist):
                val_mcc_value = round(val_mcc_hist[epoch - 1], 4)
            else:
                val_mcc_value = None
            ws_graphs.cell(row=graph_data_start_row + epoch - 1, column=col_idx, value=val_mcc_value)
            
            # Test MCC (NEW)
            if epoch <= len(test_mcc_hist):
                test_mcc_value = round(test_mcc_hist[epoch - 1], 4)
            else:
                test_mcc_value = None
            ws_graphs.cell(row=graph_data_start_row + epoch - 1, column=col_idx+1, value=test_mcc_value)
            
            # Training MCC
            if epoch <= len(train_mcc_hist):
                train_mcc_value = round(train_mcc_hist[epoch - 1], 4)
            else:
                train_mcc_value = None
            ws_graphs.cell(row=graph_data_start_row + epoch - 1, column=col_idx+2, value=train_mcc_value)
            
            col_idx += 3 
    
    # Create Line Chart for MCC comparison (ALL DATA)
    chart = LineChart()
    chart.title = "MCC Comparison Across Epochs (Train, Val, Test)"
    chart.style = 10
    chart.y_axis.title = 'MCC Score'
    chart.x_axis.title = 'Epoch'
    chart.width = 25 
    chart.height = 15
    
    # Add data to chart
    total_cols = len(all_results) * 3  # 3 series per experiment (train, val, test)
    data_range = Reference(ws_graphs, min_col=2, min_row=1, max_col=total_cols + 1, max_row=max_epochs + 1)
    categories = Reference(ws_graphs, min_col=1, min_row=2, max_row=max_epochs + 1)
    
    chart.add_data(data_range, titles_from_data=True)
    chart.set_categories(categories)
    
    # Add chart to worksheet
    ws_graphs.add_chart(chart, "B15")
    
    # Create Sheet 4: Best Performance Analysis 
    ws_best = wb.create_sheet("Best Performance Analysis")
    
    # Find best performing configurations
    best_by_test_mcc = max(all_results, key=lambda x: max(x['epoch_wise_data']['test_mcc_history']))  
    best_by_val_mcc = max(all_results, key=lambda x: max(x['epoch_wise_data']['val_mcc_history']))
    best_by_train_mcc = max(all_results, key=lambda x: max(x['epoch_wise_data']['train_mcc_history']))
    
    # Create analysis table
    analysis_headers = ['Metric', 'Best_Learning_Rate', 'Best_Batch_Size', 'Best_Value', 'Best_Epoch', 'Dataset']
    
    for col, header in enumerate(analysis_headers, 1):
        ws_best.cell(row=1, column=col, value=header)
    
    # Test MCC best
    best_test_mcc_val = max(best_by_test_mcc['epoch_wise_data']['test_mcc_history'])
    best_test_mcc_epoch = best_by_test_mcc['epoch_wise_data']['test_mcc_history'].index(best_test_mcc_val) + 1
    
    ws_best.cell(row=2, column=1, value="MCC")
    ws_best.cell(row=2, column=2, value=best_by_test_mcc['hyperparameters']['learning_rate'])
    ws_best.cell(row=2, column=3, value=best_by_test_mcc['hyperparameters']['batch_size'])
    ws_best.cell(row=2, column=4, value=round(best_test_mcc_val, 4))
    ws_best.cell(row=2, column=5, value=best_test_mcc_epoch)
    ws_best.cell(row=2, column=6, value="TEST")
    
    # Validation MCC best
    best_val_mcc_val = max(best_by_val_mcc['epoch_wise_data']['val_mcc_history'])
    best_val_mcc_epoch = best_by_val_mcc['epoch_wise_data']['val_mcc_history'].index(best_val_mcc_val) + 1
    
    ws_best.cell(row=3, column=1, value="MCC")
    ws_best.cell(row=3, column=2, value=best_by_val_mcc['hyperparameters']['learning_rate'])
    ws_best.cell(row=3, column=3, value=best_by_val_mcc['hyperparameters']['batch_size'])
    ws_best.cell(row=3, column=4, value=round(best_val_mcc_val, 4))
    ws_best.cell(row=3, column=5, value=best_val_mcc_epoch)
    ws_best.cell(row=3, column=6, value="VALIDATION")
    
    # Training MCC best
    best_train_mcc_val = max(best_by_train_mcc['epoch_wise_data']['train_mcc_history'])
    best_train_mcc_epoch = best_by_train_mcc['epoch_wise_data']['train_mcc_history'].index(best_train_mcc_val) + 1
    
    ws_best.cell(row=4, column=1, value="MCC")
    ws_best.cell(row=4, column=2, value=best_by_train_mcc['hyperparameters']['learning_rate'])
    ws_best.cell(row=4, column=3, value=best_by_train_mcc['hyperparameters']['batch_size'])
    ws_best.cell(row=4, column=4, value=round(best_train_mcc_val, 4))
    ws_best.cell(row=4, column=5, value=best_train_mcc_epoch)
    ws_best.cell(row=4, column=6, value="TRAINING")
    
    # Auto-adjust column widths for best performance sheet
    for column in ws_best.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 25)
        ws_best.column_dimensions[column_letter].width = adjusted_width
    
    # Add hyperparameter vs MCC analysis 
    ws_best.cell(row=7, column=1, value="Learning Rate vs MCC Analysis (All Datasets)")
    ws_best.cell(row=8, column=1, value="Learning_Rate")
    ws_best.cell(row=8, column=2, value="Batch_Size")
    ws_best.cell(row=8, column=3, value="Best_Train_MCC")
    ws_best.cell(row=8, column=4, value="Best_Val_MCC")
    ws_best.cell(row=8, column=5, value="Best_Test_MCC")  
    ws_best.cell(row=8, column=6, value="Config_Rank_by_Test_MCC")  
    
    # Sort results by best test MCC for ranking
    sorted_results = sorted(all_results, key=lambda x: max(x['epoch_wise_data']['test_mcc_history']), reverse=True)
    
    for idx, result in enumerate(sorted_results):
        row_num = 9 + idx
        ws_best.cell(row=row_num, column=1, value=result['hyperparameters']['learning_rate'])
        ws_best.cell(row=row_num, column=2, value=result['hyperparameters']['batch_size'])
        ws_best.cell(row=row_num, column=3, value=round(max(result['epoch_wise_data']['train_mcc_history']), 4))
        ws_best.cell(row=row_num, column=4, value=round(max(result['epoch_wise_data']['val_mcc_history']), 4))
        ws_best.cell(row=row_num, column=5, value=round(max(result['epoch_wise_data']['test_mcc_history']), 4))  
        ws_best.cell(row=row_num, column=6, value=idx + 1)
    
    # Save the Excel file
    try:
        wb.save(excel_path)
        print(f"Excel file with epoch-wise MCC analysis (including test data) saved to: {excel_path}")
        return excel_path
    except Exception as e:
        print(f"Error saving Excel file: {e}")
        return None
    

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import logging
import logging.config
from collections import Counter
import itertools
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import sys


try:
    import yaml
except ImportError:
    print("Warning: yaml package not found. Using alternative implementation.")

    class SimpleYAML:
        @staticmethod
        def dump(data, file):
            for key, value in data.items():
                file.write(f"{key}: {value}\n")
    yaml = SimpleYAML()

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("Warning: transformers package not found. Please install with: pip install transformers")
    
try:
    from peft import get_peft_model, LoraConfig, TaskType
except ImportError:
    print("Warning: peft package not found. Please install with: pip install peft")
    
try:
    from sklearn.metrics import (
        classification_report, confusion_matrix, roc_curve, auc, 
        precision_recall_curve, average_precision_score, roc_auc_score,
        matthews_corrcoef
    )
except ImportError:
    print("Warning: scikit-learn package not found. Please install with: pip install scikit-learn")
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib package not found. Please install with: pip install matplotlib")

try:
    import openpyxl
    from openpyxl.chart import LineChart, Reference
    from openpyxl.chart.axis import DateAxis
    from openpyxl.chart.label import DataLabelList
except ImportError:
    print("Warning: openpyxl package not found. Please install with: pip install openpyxl")
    
try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm package not found. Using simple progress indicator instead.")
    # Simple tqdm alternative
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        total = len(iterable) if hasattr(iterable, '__len__') else kwargs.get('total', None)
        print(f"Starting {desc}...")
        for i, item in enumerate(iterable):
            if i % 10 == 0 and total:
                print(f"{desc}: {i}/{total} ({i/total*100:.1f}%)")
            yield item
        print(f"Finished {desc}!")

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Base configuration
base_config = {
    'seed': 1,
    'num_workers': 2,
    'cv': 1,
    'rerun': 1,
    'use_cuda': True,
    'clip_grad': 10.0,
    'gamma': 0.95,
    'use_desc': False,
    'use_ocr': False,
    'use_synthetic': False,
    'weighted_0': 1.0,
    'weighted_1': 10.0,
    'weighted_loss': True,
    'dataPath': "data/",
    'textPath': ".csv",
    'num_classes': 2,
    'resultPath': "result.npy",
    'lossPath': "loss.pdf",
    'configWritePath': "config.yml",
    'errorAnalysisPath': "error_analysis.txt",
    'hidden_dim': 512,
    'epoch': 8  
}

# Hyperparameters
hyperparams = {
    'lr': [2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4],
    'batch_size': [8, 16, 32]
}

CLASSIFICATION_PROMPT = """Classify if this tweet indicates wildlife trafficking or illegal ivory trade. Positive: selling, buying, pricing, auctioning, or trading ivory or protected animal products. Negative: conservation news, history, or ivory as color reference."""

# Dataset class
class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128, config=None):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.config = config if config is not None else {}
        
        prompt_tokens = len(tokenizer.encode(CLASSIFICATION_PROMPT))
        print(f"Classification prompt uses {prompt_tokens} tokens")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tweet = self.data.iloc[idx]['tweet_text_cleaned']
        
        # Include user description if specified
        if self.config.get('use_desc', False) and 'user_description_cleaned' in self.data.columns and not pd.isna(self.data.iloc[idx]['user_description_cleaned']):
            user_desc = self.data.iloc[idx]['user_description_cleaned']
            tweet_content = f"Tweet: {tweet} User description: {user_desc}"
        else:
            tweet_content = f"Tweet: {tweet}"
        
        # Use classification prompt
        text = f"{CLASSIFICATION_PROMPT} {tweet_content}"
            
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Get input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Get label
        label = torch.tensor(self.data.iloc[idx]['label'], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

# Training function with probability outputs
def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=None):
    model.train()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Get predictions and probabilities
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().float().numpy())
    
    return epoch_loss / len(dataloader), all_preds, all_labels, all_probs

# Evaluation function with probability outputs
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss
            logits = outputs.logits
            
            epoch_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.detach().cpu().float().numpy())
    
    return epoch_loss / len(dataloader), all_preds, all_labels, all_probs

def calculate_metrics(true_labels, predicted_labels, predicted_probs=None):
    """Calculate comprehensive classification metrics with MCC as the primary focus"""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # Calculate MCC (Matthews Correlation Coefficient) - PRIMARY METRIC
    mcc = matthews_corrcoef(true_labels, predicted_labels)
    
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
    
    # Calculate macro average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Calculate confusion matrix components
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # For binary classification
    if len(np.unique(true_labels)) == 2:
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
            if cm.shape == (1, 1):
                if np.unique(predicted_labels)[0] == 0:
                    tn = cm[0, 0]
                else:
                    tp = cm[0, 0]
        
        # Calculate AUC if probabilities are provided
        auc_score = None
        if predicted_probs is not None:
            try:
                # Use probabilities for positive class (class 1)
                if isinstance(predicted_probs, list):
                    predicted_probs = np.array(predicted_probs)
                
                if predicted_probs.ndim == 2 and predicted_probs.shape[1] >= 2:
                    pos_probs = predicted_probs[:, 1]  
                else:
                    pos_probs = predicted_probs
                
                auc_score = roc_auc_score(true_labels, pos_probs)
            except Exception as e:
                print(f"Warning: Could not calculate AUC score: {e}")
                auc_score = None
        
        # Get metrics for WLT class
        pos_precision = precision[1] if len(precision) > 1 else (tp / (tp + fp) if (tp + fp) > 0 else 0)
        pos_recall = recall[1] if len(recall) > 1 else (tp / (tp + fn) if (tp + fn) > 0 else 0)
        pos_f1 = f1[1] if len(f1) > 1 else (2 * pos_precision * pos_recall / (pos_precision + pos_recall) if (pos_precision + pos_recall) > 0 else 0)
        
    else:
        # Multi-class case - set binary-specific metrics to None
        tn = fp = fn = tp = None
        auc_score = None  
        pos_precision = None
        pos_recall = None
        pos_f1 = None
    
    return {
        'mcc': mcc,  
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall, 
        'macro_f1': macro_f1,
        'pos_precision': pos_precision,
        'pos_recall': pos_recall,
        'pos_f1': pos_f1,
        'auc': auc_score,
        'confusion_matrix': cm.tolist(),
        'tn': int(tn) if tn is not None else None,
        'fp': int(fp) if fp is not None else None,
        'fn': int(fn) if fn is not None else None,
        'tp': int(tp) if tp is not None else None
    }

def save_predictions_with_errors(all_preds, all_labels, all_probs, dataset_df, output_path, dataset_name, config=None):
    """Save predictions with enhanced data for separate error analysis"""
    
    # Create DataFrame with predictions and enhanced information
    results_df = dataset_df.copy()
    results_df['predicted_label'] = all_preds
    results_df['true_label'] = all_labels
    results_df['prob_class_0'] = [prob[0] for prob in all_probs]
    results_df['prob_class_1'] = [prob[1] for prob in all_probs]
    results_df['prediction_confidence'] = [max(prob) for prob in all_probs]
    
    # Add classification prompt and formatted text for error analysis
    enhanced_texts = []
    for idx, row in results_df.iterrows():
        tweet = row['tweet_text_cleaned']
        if config and config.get('use_desc', False) and 'user_description_cleaned' in results_df.columns and not pd.isna(row['user_description_cleaned']):
            user_desc = row['user_description_cleaned']
            tweet_content = f"Tweet: {tweet} User description: {user_desc}"
        else:
            tweet_content = f"Tweet: {tweet}"
        
        formatted_text = f"{CLASSIFICATION_PROMPT}\n\nText to classify: {tweet_content}\n\nLabel (0 or 1):"
        enhanced_texts.append(formatted_text)
    
    results_df['formatted_input_text'] = enhanced_texts
    results_df['original_tweet_only'] = results_df['tweet_text_cleaned']
    
    # Add classification type
    classification_types = []
    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label == 1 and pred_label == 1:
            classification_types.append('True Positive')
        elif true_label == 0 and pred_label == 0:
            classification_types.append('True Negative')
        elif true_label == 0 and pred_label == 1:
            classification_types.append('False Positive')  
        elif true_label == 1 and pred_label == 0:
            classification_types.append('False Negative')  
    
    results_df['classification_type'] = classification_types
    results_df['is_correct'] = results_df['true_label'] == results_df['predicted_label']
    
    # Add dataset name for tracking
    results_df['dataset_split'] = dataset_name
    
    # Save results to CSV for error analysis
    results_df.to_csv(output_path, index=False)
    
    # Create separate files for each classification type with all necessary columns
    error_analysis_columns = [
        'original_tweet_only', 'formatted_input_text', 'true_label', 'predicted_label', 
        'prob_class_0', 'prob_class_1', 'prediction_confidence', 'classification_type', 
        'is_correct', 'dataset_split'
    ]
    
    # Include additional columns if they exist
    optional_columns = ['user_description_cleaned', 'tweet_id', 'user_id']
    for col in optional_columns:
        if col in results_df.columns:
            error_analysis_columns.insert(-4, col)  
    
    tp_df = results_df[results_df['classification_type'] == 'True Positive'][error_analysis_columns]
    tn_df = results_df[results_df['classification_type'] == 'True Negative'][error_analysis_columns]
    fp_df = results_df[results_df['classification_type'] == 'False Positive'][error_analysis_columns]
    fn_df = results_df[results_df['classification_type'] == 'False Negative'][error_analysis_columns]
    
    # Save classification type files
    if len(tp_df) > 0:
        tp_path = output_path.replace('.csv', '_true_positives_for_analysis.csv')
        tp_df.to_csv(tp_path, index=False)
        print(f"True Positives for analysis saved to: {tp_path} ({len(tp_df)} instances)")
    
    if len(tn_df) > 0:
        tn_path = output_path.replace('.csv', '_true_negatives_for_analysis.csv')
        tn_df.to_csv(tn_path, index=False)
        print(f"True Negatives for analysis saved to: {tn_path} ({len(tn_df)} instances)")
    
    if len(fp_df) > 0:
        fp_path = output_path.replace('.csv', '_false_positives_for_analysis.csv')
        fp_df.to_csv(fp_path, index=False)
        print(f"False Positives for analysis saved to: {fp_path} ({len(fp_df)} instances)")
    
    if len(fn_df) > 0:
        fn_path = output_path.replace('.csv', '_false_negatives_for_analysis.csv')
        fn_df.to_csv(fn_path, index=False)
        print(f"False Negatives for analysis saved to: {fn_path} ({len(fn_df)} instances)")
    
    # Save a comprehensive error analysis file combining all errors
    errors_df = pd.concat([fp_df, fn_df], ignore_index=True)
    if len(errors_df) > 0:
        errors_path = output_path.replace('.csv', '_all_errors_for_analysis.csv')
        errors_df.to_csv(errors_path, index=False)
        print(f"All errors for analysis saved to: {errors_path} ({len(errors_df)} instances)")
    
    return results_df, tp_df, tn_df, fp_df, fn_df


def analyze_classification_patterns(tp_df, tn_df, fp_df, fn_df, dataset_name):
    """Analyze patterns in all four classification types"""
    
    print(f"\n{dataset_name} Complete Classification Analysis:")
    print("="*60)
    
    total_instances = len(tp_df) + len(tn_df) + len(fp_df) + len(fn_df)
    correct_instances = len(tp_df) + len(tn_df)
    
    print(f"Total instances: {total_instances}")
    print(f"Correct predictions: {correct_instances} ({correct_instances/total_instances*100:.1f}%)")
    print(f"Incorrect predictions: {len(fp_df) + len(fn_df)} ({(len(fp_df) + len(fn_df))/total_instances*100:.1f}%)")
    
    # Use the correct column name based on what's available
    text_column = 'original_tweet_only' if 'original_tweet_only' in (tp_df.columns if len(tp_df) > 0 else tn_df.columns if len(tn_df) > 0 else fp_df.columns if len(fp_df) > 0 else fn_df.columns) else 'tweet_text_cleaned'
    
    # True Positives Analysis
    if len(tp_df) > 0:
        print(f"\nTrue Positives Analysis ({len(tp_df)} instances, {len(tp_df)/total_instances*100:.1f}%):")
        print(f"Average confidence: {tp_df['prob_class_1'].mean():.4f}")
        tp_df_copy = tp_df.copy()
        tp_df_copy['text_length'] = tp_df_copy[text_column].str.len()
        print(f"Text length - Mean: {tp_df_copy['text_length'].mean():.1f}, Median: {tp_df_copy['text_length'].median():.1f}")
        
        print(f"Sample True Positives (correctly identified as positive):")
        for i, row in tp_df.head(2).iterrows():
            print(f"  - Text: '{row[text_column][:80]}...'")
            print(f"    Confidence: {row['prob_class_1']:.4f}")
    
    # True Negatives Analysis
    if len(tn_df) > 0:
        print(f"\nTrue Negatives Analysis ({len(tn_df)} instances, {len(tn_df)/total_instances*100:.1f}%):")
        print(f"Average confidence: {tn_df['prob_class_0'].mean():.4f}")
        tn_df_copy = tn_df.copy()
        tn_df_copy['text_length'] = tn_df_copy[text_column].str.len()
        print(f"Text length - Mean: {tn_df_copy['text_length'].mean():.1f}, Median: {tn_df_copy['text_length'].median():.1f}")
        
        print(f"Sample True Negatives (correctly identified as negative):")
        for i, row in tn_df.head(2).iterrows():
            print(f"  - Text: '{row[text_column][:80]}...'")
            print(f"    Confidence: {row['prob_class_0']:.4f}")
    
    # False Positives Analysis
    if len(fp_df) > 0:
        print(f"\nFalse Positives Analysis ({len(fp_df)} instances, {len(fp_df)/total_instances*100:.1f}%):")
        print(f"Average confidence: {fp_df['prob_class_1'].mean():.4f}")
        fp_df_copy = fp_df.copy()
        fp_df_copy['text_length'] = fp_df_copy[text_column].str.len()
        print(f"Text length - Mean: {fp_df_copy['text_length'].mean():.1f}, Median: {fp_df_copy['text_length'].median():.1f}")
        
        print(f"Sample False Positives (wrongly predicted as positive):")
        for i, row in fp_df.head(2).iterrows():
            print(f"  - Text: '{row[text_column][:80]}...'")
            print(f"    Confidence: {row['prob_class_1']:.4f}")
    
    # False Negatives Analysis
    if len(fn_df) > 0:
        print(f"\nFalse Negatives Analysis ({len(fn_df)} instances, {len(fn_df)/total_instances*100:.1f}%):")
        print(f"Average confidence: {fn_df['prob_class_0'].mean():.4f}")
        fn_df_copy = fn_df.copy()
        fn_df_copy['text_length'] = fn_df_copy[text_column].str.len()
        print(f"Text length - Mean: {fn_df_copy['text_length'].mean():.1f}, Median: {fn_df_copy['text_length'].median():.1f}")
        
        print(f"Sample False Negatives (wrongly predicted as negative):")
        for i, row in fn_df.head(2).iterrows():
            print(f"  - Text: '{row[text_column][:80]}...'")
            print(f"    Confidence: {row['prob_class_0']:.4f}")
            

def create_classification_summary_report(results_df, output_path):
    """Create a detailed summary report of classification results"""
    
    with open(output_path, 'w') as f:
        f.write("CLASSIFICATION ANALYSIS SUMMARY REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Overall statistics
        total = len(results_df)
        correct = len(results_df[results_df['is_correct']])
        accuracy = correct / total if total > 0 else 0
        
        f.write(f"Overall Statistics:\n")
        f.write(f"Total instances: {total}\n")
        f.write(f"Correct predictions: {correct} ({accuracy*100:.2f}%)\n")
        f.write(f"Incorrect predictions: {total-correct} ({(1-accuracy)*100:.2f}%)\n\n")
        
        # Classification breakdown
        tp_count = len(results_df[results_df['classification_type'] == 'True Positive'])
        tn_count = len(results_df[results_df['classification_type'] == 'True Negative'])
        fp_count = len(results_df[results_df['classification_type'] == 'False Positive'])
        fn_count = len(results_df[results_df['classification_type'] == 'False Negative'])
        
        f.write(f"Classification Breakdown:\n")
        f.write(f"True Positives (TP):  {tp_count} ({tp_count/total*100:.1f}%)\n")
        f.write(f"True Negatives (TN):  {tn_count} ({tn_count/total*100:.1f}%)\n")
        f.write(f"False Positives (FP): {fp_count} ({fp_count/total*100:.1f}%)\n")
        f.write(f"False Negatives (FN): {fn_count} ({fn_count/total*100:.1f}%)\n\n")
    
    print(f"Classification summary report saved to: {output_path}")

def create_consolidated_error_analysis_file(output_dir):
    """Create a consolidated file with errors from all experiments for comprehensive analysis"""
    
    all_errors = []
    experiment_dirs = [d for d in os.listdir(output_dir) if d.startswith('lr_') and os.path.isdir(os.path.join(output_dir, d))]
    
    for exp_dir in experiment_dirs:
        exp_path = os.path.join(output_dir, exp_dir)
        error_file = os.path.join(exp_path, 'test_predictions_all_errors_for_analysis.csv')
        
        if os.path.exists(error_file):
            error_df = pd.read_csv(error_file)
            error_df['experiment_name'] = exp_dir
            all_errors.append(error_df)
    
    if all_errors:
        consolidated_errors = pd.concat(all_errors, ignore_index=True)
        consolidated_path = os.path.join(output_dir, 'consolidated_errors_all_experiments.csv')
        consolidated_errors.to_csv(consolidated_path, index=False)
        print(f"Consolidated error analysis file saved to: {consolidated_path}")
        return consolidated_path
    else:
        print("No error files found to consolidate")
        return None

def run_experiment(config, train_df, val_df, test_df):
    """Run a single experiment with given hyperparameters (MODIFIED TO INCLUDE COMPLETE CLASSIFICATION ANALYSIS)"""
    # Create experiment directory
    experiment_dir = os.path.join(config['outputPath'], f"lr_{config['lr']}_bs_{config['batch_size']}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging for this experiment
    log_file = os.path.join(experiment_dir, "log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    logger = logging.getLogger(f"exp_lr_{config['lr']}_bs_{config['batch_size']}")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    
    # Save config
    config_path = os.path.join(experiment_dir, config['configWritePath'])
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and config['use_cuda'] else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    if tokenizer.pad_token is None:
        # Use the EOS token as the pad token
        tokenizer.pad_token = tokenizer.eos_token
        model_config = {'pad_token_id': tokenizer.eos_token_id}
    
    # Create datasets
    train_dataset = TweetDataset(train_df, tokenizer, config=config)
    val_dataset = TweetDataset(val_df, tokenizer, config=config)
    test_dataset = TweetDataset(test_df, tokenizer, config=config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Set up PEFT with LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        num_labels=config['num_classes'],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **model_config  # Pass the pad token configuration
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params*100:.2f}% of total)")
    
    # Move model to device
    model = model.to(device)
    
    # Set up weighted loss
    if config['weighted_loss']:
        weights = torch.tensor([config['weighted_0'], config['weighted_1']], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    
    # Training loop - MODEL SELECTION BASED ON MCC (MODIFIED TO INCLUDE TEST EVALUATION)
    train_losses = []
    val_losses = []
    test_losses = []  
    
    train_mcc_history = []  # Store epoch-wise MCC for training
    val_mcc_history = []    # Store epoch-wise MCC for validation
    test_mcc_history = []   # Store epoch-wise MCC for test
    
    train_metrics_history = []  # Store all training metrics per epoch
    val_metrics_history = []    # Store all validation metrics per epoch
    test_metrics_history = []   # Store all test metrics per epoch
    
    best_val_mcc = -1.0  # Changed from loss to MCC
    best_val_metrics = None
    best_model_path = os.path.join(experiment_dir, "best_model.pkl")
    
    logger.info("Starting training with epoch-wise test evaluation...")
    
    for epoch in range(config['epoch']):
        logger.info(f"Epoch {epoch+1}/{config['epoch']}")
        
        # Train
        train_loss, train_preds, train_labels, train_probs = train_epoch(
            model, train_loader, optimizer, criterion, device, config['clip_grad']
        )
        
        # Validate
        val_loss, val_preds, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        
        # Evaluate on test set (NEW - ADDED FOR EACH EPOCH)
        test_loss, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)  
        
        # Calculate metrics
        train_metrics = calculate_metrics(train_labels, train_preds, train_probs)
        val_metrics = calculate_metrics(val_labels, val_preds, val_probs)
        test_metrics = calculate_metrics(test_labels, test_preds, test_probs)  
        
        # Store epoch-wise metrics
        train_mcc_history.append(train_metrics['mcc'])
        val_mcc_history.append(val_metrics['mcc'])
        test_mcc_history.append(test_metrics['mcc'])  
        
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        test_metrics_history.append(test_metrics)  
        
        # Print metrics with MCC emphasis 
        train_auc_str = f"{train_metrics['auc']:.4f}" if train_metrics['auc'] is not None else 'N/A'
        val_auc_str = f"{val_metrics['auc']:.4f}" if val_metrics['auc'] is not None else 'N/A'
        test_auc_str = f"{test_metrics['auc']:.4f}" if test_metrics['auc'] is not None else 'N/A'  
        
        logger.info(f"Train MCC: {train_metrics['mcc']:.4f}, Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['pos_f1']:.4f}, AUC: {train_auc_str}")
        logger.info(f"Val   MCC: {val_metrics['mcc']:.4f}, Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['pos_f1']:.4f}, AUC: {val_auc_str}")
        logger.info(f"Test  MCC: {test_metrics['mcc']:.4f}, Loss: {test_loss:.4f}, Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['pos_f1']:.4f}, AUC: {test_auc_str}")  
        
        # Save best model based on validation MCC 
        if val_metrics['mcc'] > best_val_mcc:
            best_val_mcc = val_metrics['mcc']
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved! (Val MCC: {best_val_mcc:.4f})")
    
    # Plot loss curves 
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')  
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training, Validation and Test Loss (lr={config["lr"]}, bs={config["batch_size"]})')
    plt.legend()
    plt.savefig(os.path.join(experiment_dir, config['lossPath']))
    plt.close()
    
    # Plot MCC curves 
    plt.figure(figsize=(12, 8))
    plt.plot(train_mcc_history, label='Train MCC', marker='o')
    plt.plot(val_mcc_history, label='Validation MCC', marker='s')
    plt.plot(test_mcc_history, label='Test MCC', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('MCC Score')
    plt.title(f'MCC Progression Across Epochs (lr={config["lr"]}, bs={config["batch_size"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random Performance')
    plt.savefig(os.path.join(experiment_dir, 'mcc_progression.pdf'))
    plt.close()
    
    # Load best model for final evaluation 
    model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation on test set using best model
    logger.info("Evaluating best model on test set...")
    final_test_loss, final_test_preds, final_test_labels, final_test_probs = evaluate(model, test_loader, criterion, device)
    
    # Also get validation predictions for complete analysis
    final_val_loss, final_val_preds, final_val_labels, final_val_probs = evaluate(model, val_loader, criterion, device)
    
    final_test_metrics = calculate_metrics(final_test_labels, final_test_preds, final_test_probs)

    # Save instance-level predictions with complete classification analysis
    test_predictions_path = os.path.join(experiment_dir, 'test_predictions_complete_analysis.csv')
    val_predictions_path = os.path.join(experiment_dir, 'val_predictions_complete_analysis.csv')

    test_results_df, test_tp_df, test_tn_df, test_fp_df, test_fn_df = save_predictions_with_errors(
        final_test_preds, final_test_labels, final_test_probs, 
        test_df, test_predictions_path, 'test', config
    )

    val_results_df, val_tp_df, val_tn_df, val_fp_df, val_fn_df = save_predictions_with_errors(
        final_val_preds, final_val_labels, final_val_probs,
        val_df, val_predictions_path, 'validation', config
    )
    # Save metadata for error analysis
    metadata = {
        'experiment_config': config,
        'classification_prompt': CLASSIFICATION_PROMPT,
        'model_info': {
            'model_name': "meta-llama/Llama-3.1-8B",
            'best_epoch_by_val_mcc': val_mcc_history.index(max(val_mcc_history)) + 1,
            'best_val_mcc': max(val_mcc_history),
            'final_test_mcc': final_test_metrics['mcc']
        },
        'dataset_stats': {
            'test_size': len(test_df),
            'val_size': len(val_df),
            'train_size': len(train_df)
        }
    }
    
    metadata_path = os.path.join(experiment_dir, 'error_analysis_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)

    print(f"Error analysis metadata saved to: {metadata_path}")

    # Log complete classification analysis
    logger.info(f"Test set classification analysis:")
    logger.info(f"  True Positives: {len(test_tp_df)} instances")
    logger.info(f"  True Negatives: {len(test_tn_df)} instances")
    logger.info(f"  False Positives: {len(test_fp_df)} instances")
    logger.info(f"  False Negatives: {len(test_fn_df)} instances")
    logger.info(f"  Total Correct: {len(test_tp_df) + len(test_tn_df)} / {len(test_results_df)}")
    
    logger.info(f"Validation set classification analysis:")
    logger.info(f"  True Positives: {len(val_tp_df)} instances")
    logger.info(f"  True Negatives: {len(val_tn_df)} instances")
    logger.info(f"  False Positives: {len(val_fp_df)} instances")
    logger.info(f"  False Negatives: {len(val_fn_df)} instances")
    logger.info(f"  Total Correct: {len(val_tp_df) + len(val_tn_df)} / {len(val_results_df)}")
    
    # Perform detailed pattern analysis
    analyze_classification_patterns(test_tp_df, test_tn_df, test_fp_df, test_fn_df, 'Test')
    analyze_classification_patterns(val_tp_df, val_tn_df, val_fp_df, val_fn_df, 'Validation')
    
    # Create summary reports
    test_summary_path = os.path.join(experiment_dir, 'test_classification_summary.txt')
    val_summary_path = os.path.join(experiment_dir, 'val_classification_summary.txt')
    create_classification_summary_report(test_results_df, test_summary_path)
    create_classification_summary_report(val_results_df, val_summary_path)
    
    # Calculate final metrics
    final_test_metrics = calculate_metrics(final_test_labels, final_test_preds, final_test_probs)
    
    # Print final test metrics
    final_auc_str = f"{final_test_metrics['auc']:.4f}" if final_test_metrics['auc'] is not None else 'N/A'
    logger.info(f"Final Test MCC: {final_test_metrics['mcc']:.4f} (PRIMARY METRIC)")
    logger.info(f"Final Test Loss: {final_test_loss:.4f}")
    logger.info(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
    logger.info(f"Final Test F1 Score: {final_test_metrics['pos_f1']:.4f}")
    logger.info(f"Final Test AUC: {final_auc_str}")
    logger.info(f"Confusion Matrix - TP: {final_test_metrics['tp']}, TN: {final_test_metrics['tn']}, FP: {final_test_metrics['fp']}, FN: {final_test_metrics['fn']}")
    
    # Save results summary 
    results = {
        'hyperparameters': {
            'learning_rate': config['lr'],
            'batch_size': config['batch_size'],
            'epochs': config['epoch']
        },
        'best_val_mcc': best_val_mcc,  
        'best_val_metrics': best_val_metrics,
        'test_loss': final_test_loss,  #
        'test_metrics': final_test_metrics,  # This is final test metrics with best model
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,  
        'epoch_wise_data': {  # UPDATED: Store epoch-wise data for Excel INCLUDING TEST
            'train_mcc_history': train_mcc_history,
            'val_mcc_history': val_mcc_history,
            'test_mcc_history': test_mcc_history,  
            'train_metrics_history': train_metrics_history,
            'val_metrics_history': val_metrics_history,
            'test_metrics_history': test_metrics_history  
        }
    }
    
    with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    return results

def plot_comparative_results(all_results, output_dir):
    """Create comparative visualizations of all experiments with MCC as primary focus"""
    # Create a results directory if it doesn't exist
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Extract results for plotting
    configs = []
    mcc_scores = []  
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    val_mccs = []  
    auc_scores = []
    
    for result in all_results:
        hp = result['hyperparameters']
        config_name = f"lr={hp['learning_rate']}, bs={hp['batch_size']}"
        configs.append(config_name)
        mcc_scores.append(result['test_metrics']['mcc'])
        accuracies.append(result['test_metrics']['accuracy'])
        f1_scores.append(result['test_metrics']['pos_f1'])
        precisions.append(result['test_metrics']['pos_precision'])
        recalls.append(result['test_metrics']['pos_recall'])
        val_mccs.append(result['best_val_mcc'])
        auc_scores.append(result['test_metrics']['auc'] if result['test_metrics']['auc'] else 0)
    
    # Sort results by MCC score for better visualization
    sorted_indices = np.argsort(mcc_scores)[::-1]  
    configs = [configs[i] for i in sorted_indices]
    mcc_scores = [mcc_scores[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    val_mccs = [val_mccs[i] for i in sorted_indices]
    auc_scores = [auc_scores[i] for i in sorted_indices]
    
    # 1. MCC Score Comparison (PRIMARY CHART)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(configs, mcc_scores, color='darkblue')
    plt.xlabel('Hyperparameters')
    plt.ylabel('MCC Score')
    plt.title('MCC Score Comparison Across Different Hyperparameters (PRIMARY METRIC)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-1.0, 1.0) 
    
    # Add value labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', rotation=0, fontweight='bold')
    
    # Add horizontal line at MCC=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random Performance (MCC=0)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'mcc_comparison_primary.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'mcc_comparison_primary.png'), dpi=300)
    plt.close()
    
    # 2. Enhanced Multiple Metrics Comparison (MCC emphasized)
    plt.figure(figsize=(16, 10))
    x = np.arange(len(configs))
    width = 0.13
    
    plt.bar(x - width*2.5, mcc_scores, width, label='MCC (Primary)', color='darkblue', edgecolor='black', linewidth=1.5)
    plt.bar(x - width*1.5, accuracies, width, label='Accuracy', color='skyblue')
    plt.bar(x - width/2, f1_scores, width, label='F1 Score', color='orange')
    plt.bar(x + width/2, precisions, width, label='Precision', color='green')
    plt.bar(x + width*1.5, recalls, width, label='Recall', color='red')
    plt.bar(x + width*2.5, auc_scores, width, label='AUC', color='purple')
    
    plt.xlabel('Hyperparameters')
    plt.ylabel('Score')
    plt.title('Comprehensive Performance Metrics Comparison (MCC as Primary Metric)')
    plt.xticks(x, configs, rotation=45, ha='right')
    plt.ylim(-1.0, 1.0)  
    plt.legend()
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'comprehensive_metrics_mcc_focused.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'comprehensive_metrics_mcc_focused.png'), dpi=300)
    plt.close()
    
    # 3. MCC vs Validation MCC scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(val_mccs, mcc_scores, s=100, alpha=0.7, c='darkblue')
    plt.xlabel('Validation MCC')
    plt.ylabel('Test MCC')
    plt.title('Test MCC vs Validation MCC')
    
    # Add diagonal line for reference
    min_mcc = min(min(val_mccs), min(mcc_scores))
    max_mcc = max(max(val_mccs), max(mcc_scores))
    plt.plot([min_mcc, max_mcc], [min_mcc, max_mcc], 'r--', alpha=0.5, label='Perfect Correlation')
    
    # Annotate points with config names
    for i, config in enumerate(configs):
        plt.annotate(f'{i+1}', (val_mccs[i], mcc_scores[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'mcc_validation_vs_test.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'mcc_validation_vs_test.png'), dpi=300)
    plt.close()
    
    # 4. Heatmap by hyperparameters (MCC focused)
    # Reorganize data for heatmap
    lr_values = sorted(list(set([res['hyperparameters']['learning_rate'] for res in all_results])))
    bs_values = sorted(list(set([res['hyperparameters']['batch_size'] for res in all_results])))
    
    # Create matrices for each metric (MCC emphasized)
    mcc_matrix = np.zeros((len(lr_values), len(bs_values)))
    accuracy_matrix = np.zeros((len(lr_values), len(bs_values)))
    f1_matrix = np.zeros((len(lr_values), len(bs_values)))
    auc_matrix = np.zeros((len(lr_values), len(bs_values)))
    
    # Fill matrices
    for result in all_results:
        lr_idx = lr_values.index(result['hyperparameters']['learning_rate'])
        bs_idx = bs_values.index(result['hyperparameters']['batch_size'])
        mcc_matrix[lr_idx, bs_idx] = result['test_metrics']['mcc']
        accuracy_matrix[lr_idx, bs_idx] = result['test_metrics']['accuracy']
        f1_matrix[lr_idx, bs_idx] = result['test_metrics']['pos_f1']
        auc_matrix[lr_idx, bs_idx] = result['test_metrics']['auc'] if result['test_metrics']['auc'] else 0
    
    # Create heatmaps with MCC as primary
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # MCC Heatmap (PRIMARY)
    sns.heatmap(mcc_matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                xticklabels=bs_values, yticklabels=[f'{lr:.1e}' for lr in lr_values],
                vmin=-1, vmax=1, ax=axes[0,0], center=0)
    axes[0,0].set_xlabel('Batch Size')
    axes[0,0].set_ylabel('Learning Rate')
    axes[0,0].set_title('MCC by Hyperparameters (PRIMARY METRIC)')
    
    # F1 Score Heatmap
    sns.heatmap(f1_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=bs_values, yticklabels=[f'{lr:.1e}' for lr in lr_values],
                vmin=0, vmax=1, ax=axes[0,1])
    axes[0,1].set_xlabel('Batch Size')
    axes[0,1].set_ylabel('Learning Rate')
    axes[0,1].set_title('F1 Score by Hyperparameters')
    
    # Accuracy Heatmap
    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=bs_values, yticklabels=[f'{lr:.1e}' for lr in lr_values],
                vmin=0, vmax=1, ax=axes[1,0])
    axes[1,0].set_xlabel('Batch Size')
    axes[1,0].set_ylabel('Learning Rate')
    axes[1,0].set_title('Accuracy by Hyperparameters')
    
    # AUC Heatmap
    sns.heatmap(auc_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=bs_values, yticklabels=[f'{lr:.1e}' for lr in lr_values],
                vmin=0, vmax=1, ax=axes[1,1])
    axes[1,1].set_xlabel('Batch Size')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].set_title('AUC by Hyperparameters')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'all_metrics_heatmap_mcc_focused.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'all_metrics_heatmap_mcc_focused.png'), dpi=300)
    plt.close()
    
    # 5. Learning rate comparison (grouped by batch size) - MCC Enhanced
    plt.figure(figsize=(15, 10))
    
    # Create subplots for different metrics with MCC as primary
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics_data = {
        'MCC': (mcc_scores, 'MCC Score (PRIMARY)'),
        'F1 Score': (f1_scores, 'F1 Score (Class 1)'),
        'Accuracy': (accuracies, 'Accuracy'),
        'AUC': (auc_scores, 'AUC Score')
    }
    
    for idx, (metric_name, (metric_values, ylabel)) in enumerate(metrics_data.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Group by batch size
        for bs in bs_values:
            bs_results = [r for r in all_results if r['hyperparameters']['batch_size'] == bs]
            bs_results.sort(key=lambda x: x['hyperparameters']['learning_rate'])
            
            lr_vals = [r['hyperparameters']['learning_rate'] for r in bs_results]
            
            if metric_name == 'MCC':
                metric_vals = [r['test_metrics']['mcc'] for r in bs_results]
                line_style = 'o-'
                line_width = 3 if metric_name == 'MCC' else 1
            elif metric_name == 'F1 Score':
                metric_vals = [r['test_metrics']['pos_f1'] for r in bs_results]
                line_style = 's-'
                line_width = 1
            elif metric_name == 'Accuracy':
                metric_vals = [r['test_metrics']['accuracy'] for r in bs_results]
                line_style = '^-'
                line_width = 1
            elif metric_name == 'AUC':
                metric_vals = [r['test_metrics']['auc'] if r['test_metrics']['auc'] else 0 for r in bs_results]
                line_style = 'd-'
                line_width = 1
            
            ax.plot(lr_vals, metric_vals, line_style, label=f'Batch Size = {bs}', linewidth=line_width)
        
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs Learning Rate by Batch Size')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        # Special formatting for MCC
        if metric_name == 'MCC':
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random Performance')
            ax.set_ylim(-1.0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'metrics_vs_lr_mcc_focused.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'metrics_vs_lr_mcc_focused.png'), dpi=300)
    plt.close()
    
    # 6. Best model highlight with MCC emphasis
    best_model = max(all_results, key=lambda x: x['test_metrics']['mcc']) 
    best_config = f"lr={best_model['hyperparameters']['learning_rate']}, bs={best_model['hyperparameters']['batch_size']}"
    
    plt.figure(figsize=(12, 8))
    metrics = ['MCC (Primary)', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC']
    best_values = [
        best_model['test_metrics']['mcc'],
        best_model['test_metrics']['accuracy'],
        best_model['test_metrics']['pos_f1'],
        best_model['test_metrics']['pos_precision'],
        best_model['test_metrics']['pos_recall'],
        best_model['test_metrics']['auc'] if best_model['test_metrics']['auc'] else 0
    ]
    
    colors = ['darkblue', 'skyblue', 'orange', 'green', 'red', 'purple']
    bars = plt.bar(metrics, best_values, color=colors)
    
    # Emphasize MCC bar
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(3)
    
    for bar, value in zip(bars, best_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold' if bar == bars[0] else 'normal')
    
    plt.ylim(-1.0, 1.0)  # Accommodate MCC range
    plt.title(f'Best Model Performance (Selected by MCC): {best_config}')
    plt.ylabel('Score')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'best_model_mcc_focused.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'best_model_mcc_focused.png'), dpi=300)
    plt.close()
    
    # 7. Confusion Matrix Visualization for Best Model (MCC-selected)
    cm = np.array(best_model['test_metrics']['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix - Best Model (Selected by MCC)\n{best_config}\nMCC: {best_model["test_metrics"]["mcc"]:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add TP, TN, FP, FN annotations
    if cm.shape == (2, 2):
        plt.text(0.5, -0.1, f'TN: {best_model["test_metrics"]["tn"]}', 
                ha='center', transform=plt.gca().transAxes)
        plt.text(1.5, -0.1, f'FP: {best_model["test_metrics"]["fp"]}', 
                ha='center', transform=plt.gca().transAxes)
        plt.text(0.5, -0.15, f'FN: {best_model["test_metrics"]["fn"]}', 
                ha='center', transform=plt.gca().transAxes)
        plt.text(1.5, -0.15, f'TP: {best_model["test_metrics"]["tp"]}', 
                ha='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'confusion_matrix_best_mcc.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'confusion_matrix_best_mcc.png'), dpi=300)
    plt.close()
    
    # 8. MCC Distribution Analysis
    plt.figure(figsize=(12, 8))
    
    # Create subplot for MCC analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MCC histogram
    ax1.hist(mcc_scores, bins=10, color='darkblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('MCC Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of MCC Scores Across All Experiments')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Random Performance')
    ax1.axvline(x=np.mean(mcc_scores), color='green', linestyle='-', alpha=0.7, label=f'Mean MCC: {np.mean(mcc_scores):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MCC vs other metrics scatter
    ax2.scatter(mcc_scores, f1_scores, alpha=0.7, s=100, c='orange', label='F1 Score')
    ax2.scatter(mcc_scores, accuracies, alpha=0.7, s=100, c='skyblue', label='Accuracy')
    ax2.set_xlabel('MCC Score')
    ax2.set_ylabel('Other Metrics')
    ax2.set_title('MCC vs Other Performance Metrics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'mcc_distribution_analysis.pdf'))
    plt.savefig(os.path.join(comparison_dir, 'mcc_distribution_analysis.png'), dpi=300)
    plt.close()
    
    # 9. Summary table in text file with MCC emphasis
    with open(os.path.join(comparison_dir, 'mcc_focused_results_summary.txt'), 'w') as f:
        f.write("MCC-FOCUSED HYPERPARAMETER TUNING RESULTS SUMMARY\n")
        f.write("==================================================\n\n")
        
        f.write("Best Model Configuration (Selected by MCC):\n")
        f.write(f"Learning Rate: {best_model['hyperparameters']['learning_rate']}\n")
        f.write(f"Batch Size: {best_model['hyperparameters']['batch_size']}\n")
        f.write(f"Epochs: {best_model['hyperparameters']['epochs']}\n\n")
        
        f.write("Best Model Performance:\n")
        f.write(f"MCC (PRIMARY METRIC): {best_model['test_metrics']['mcc']:.4f}\n")
        f.write(f"Accuracy: {best_model['test_metrics']['accuracy']:.4f}\n")
        f.write(f"F1 Score (Class 1): {best_model['test_metrics']['pos_f1']:.4f}\n")
        f.write(f"Precision (Class 1): {best_model['test_metrics']['pos_precision']:.4f}\n")
        f.write(f"Recall (Class 1): {best_model['test_metrics']['pos_recall']:.4f}\n")
        auc_str = f"{best_model['test_metrics']['auc']:.4f}" if best_model['test_metrics']['auc'] is not None else 'N/A'
        f.write(f"AUC: {auc_str}\n\n")
        
        f.write("Confusion Matrix Components:\n")
        f.write(f"True Positives (TP): {best_model['test_metrics']['tp']}\n")
        f.write(f"True Negatives (TN): {best_model['test_metrics']['tn']}\n")
        f.write(f"False Positives (FP): {best_model['test_metrics']['fp']}\n")
        f.write(f"False Negatives (FN): {best_model['test_metrics']['fn']}\n\n")
        
        f.write("MCC Analysis:\n")
        f.write(f"Mean MCC across all experiments: {np.mean(mcc_scores):.4f}\n")
        f.write(f"Standard deviation of MCC: {np.std(mcc_scores):.4f}\n")
        f.write(f"Best MCC: {max(mcc_scores):.4f}\n")
        f.write(f"Worst MCC: {min(mcc_scores):.4f}\n")
        f.write(f"Number of experiments with MCC > 0: {sum(1 for mcc in mcc_scores if mcc > 0)}/{len(mcc_scores)}\n\n")
        
        f.write("All Configurations (Sorted by MCC Score):\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'Rank':<4} {'Configuration':<25} {'MCC':<7} {'Acc':<6} {'F1':<6} {'AUC':<6} {'TP':<4} {'TN':<4} {'FP':<4} {'FN':<4}\n")
        f.write("-" * 85 + "\n")
        
        for i, result in enumerate(sorted(all_results, key=lambda x: x['test_metrics']['mcc'], reverse=True)):
            config_name = f"lr={result['hyperparameters']['learning_rate']:.1e}, bs={result['hyperparameters']['batch_size']}"
            auc_str = f"{result['test_metrics']['auc']:.3f}" if result['test_metrics']['auc'] is not None else 'N/A'
            f.write(f"{i+1:<4} {config_name:<25} ")
            f.write(f"{result['test_metrics']['mcc']:<7.3f} ")
            f.write(f"{result['test_metrics']['accuracy']:<6.3f} ")
            f.write(f"{result['test_metrics']['pos_f1']:<6.3f} ")
            f.write(f"{auc_str:<6} ")
            f.write(f"{result['test_metrics']['tp']:<4} ")
            f.write(f"{result['test_metrics']['tn']:<4} ")
            f.write(f"{result['test_metrics']['fp']:<4} ")
            f.write(f"{result['test_metrics']['fn']:<4}\n")
    
    return best_model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run MCC-focused hyperparameter tuning for Llama LoRA')
    parser.add_argument('--data_path', type=str, default='data/',
                    help='Path to the data directory containing train.csv, val.csv, and test.csv')
    parser.add_argument('--output_dir', type=str, default=None,
                    help='Output directory (default: 2/2_1_original/hyperparameter_tuning_TIMESTAMP)')
    parser.add_argument('--subset', action='store_true',
                    help='Run only a subset of hyperparameters for testing')
    parser.add_argument('--no_cuda', action='store_true',
                    help='Disable CUDA (use CPU only)')
    args = parser.parse_args()
    
    # Set up main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("2/2_1_original/mcc_focused_hyperparameter_tuning.log"),
            logging.StreamHandler()
        ]
    )
    main_logger = logging.getLogger(__name__)
    
    # Create output directory based on timestamp 
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"2/2_1_original/hyperparameter_tuning_{timestamp}"
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Update base config with command line args
    base_config['dataPath'] = args.data_path
    base_config['outputPath'] = output_dir
    base_config['use_cuda'] = not args.no_cuda
    
    # Load data
    main_logger.info(f"Loading data from {base_config['dataPath']}")
    try:
        train_df = pd.read_csv(os.path.join(base_config['dataPath'], 'train.csv'))
        test_df = pd.read_csv(os.path.join(base_config['dataPath'], 'test.csv'))
        val_df = pd.read_csv(os.path.join(base_config['dataPath'], 'val.csv'))
    except FileNotFoundError as e:
        main_logger.error(f"Error loading data: {e}")
        print(f"Error: Could not find the required CSV files in {base_config['dataPath']}")
        print("Please make sure train.csv, val.csv, and test.csv exist in the specified data path.")
        sys.exit(1)
    
    main_logger.info(f"Train data shape: {train_df.shape}")
    main_logger.info(f"Test data shape: {test_df.shape}")
    main_logger.info(f"Validation data shape: {val_df.shape}")
    
    # Use subset of hyperparameters if requested
    if args.subset:
        # Use only 3 learning rates and 2 batch sizes for faster testing
        lr_values = [hyperparams['lr'][0], hyperparams['lr'][4], hyperparams['lr'][-1]]  
        bs_values = [hyperparams['batch_size'][0], hyperparams['batch_size'][-1]]        
        main_logger.info("Running with subset of hyperparameters for testing")
    else:
        lr_values = hyperparams['lr']
        bs_values = hyperparams['batch_size']
    
    # Generate hyperparameter combinations
    param_combinations = list(itertools.product(lr_values, bs_values))
    main_logger.info(f"Running {len(param_combinations)} experiments with different hyperparameter combinations")
    main_logger.info("PRIMARY EVALUATION METRIC: MCC (Matthews Correlation Coefficient)")
    
    # Store all results
    all_results = []
    
    # Run experiments for each hyperparameter combination
    for lr, batch_size in param_combinations:
        # Update config for this run
        run_config = base_config.copy()
        run_config['lr'] = lr
        run_config['batch_size'] = batch_size
        
        # Log experiment start
        main_logger.info(f"Starting experiment with lr={lr}, batch_size={batch_size}")
        
        # Run the experiment
        result = run_experiment(run_config, train_df, val_df, test_df)
        
        # Store results
        all_results.append(result)
        
        # Log experiment completion with MCC emphasis
        auc_str = f"{result['test_metrics']['auc']:.4f}" if result['test_metrics']['auc'] is not None else 'N/A'
        main_logger.info(f"Completed experiment with lr={lr}, batch_size={batch_size}")
        main_logger.info(f"Test MCC: {result['test_metrics']['mcc']:.4f} (PRIMARY), F1: {result['test_metrics']['pos_f1']:.4f}, AUC: {auc_str}")
    
    # After all experiments, create Excel analysis
    main_logger.info("Creating Excel file with epoch-wise MCC analysis...")
    excel_path = create_excel_mcc_analysis(all_results, output_dir)
    if excel_path:
        main_logger.info(f"Excel analysis saved to: {excel_path}")
    
    # Create consolidated error analysis file
    main_logger.info("Creating consolidated error analysis file...")
    consolidated_path = create_consolidated_error_analysis_file(output_dir)
    if consolidated_path:
        main_logger.info(f"Consolidated error analysis saved to: {consolidated_path}")

    # After all experiments, plot comparative results
    main_logger.info("All experiments completed. Generating MCC-focused comparative analysis...")
    best_model = plot_comparative_results(all_results, output_dir)
    
    # Display best model with MCC emphasis
    auc_str = f"{best_model['test_metrics']['auc']:.4f}" if best_model['test_metrics']['auc'] is not None else 'N/A'
    main_logger.info(f"Best model configuration (by MCC): lr={best_model['hyperparameters']['learning_rate']}, batch_size={best_model['hyperparameters']['batch_size']}")
    main_logger.info(f"Best model metrics:")
    main_logger.info(f"  MCC (PRIMARY): {best_model['test_metrics']['mcc']:.4f}")
    main_logger.info(f"  Accuracy: {best_model['test_metrics']['accuracy']:.4f}")
    main_logger.info(f"  F1 Score: {best_model['test_metrics']['pos_f1']:.4f}")
    main_logger.info(f"  AUC: {auc_str}")
    main_logger.info(f"  TP: {best_model['test_metrics']['tp']}, TN: {best_model['test_metrics']['tn']}, FP: {best_model['test_metrics']['fp']}, FN: {best_model['test_metrics']['fn']}")
    
    # Copy best model to root output directory
    best_model_source = os.path.join(
        output_dir, 
        f"lr_{best_model['hyperparameters']['learning_rate']}_bs_{best_model['hyperparameters']['batch_size']}", 
        "best_model.pkl"
    )
    best_model_dest = os.path.join(output_dir, "best_model.pkl")
    import shutil
    shutil.copy2(best_model_source, best_model_dest)
    main_logger.info(f"Best model copied to {best_model_dest}")
    
    # Save all results to a single file
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    
    main_logger.info(f"All results saved to {os.path.join(output_dir, 'all_results.json')}")
    main_logger.info("MCC-focused hyperparameter tuning completed successfully!")
    
    # Calculate MCC statistics
    mcc_scores = [result['test_metrics']['mcc'] for result in all_results]
    
    auc_str = f"{best_model['test_metrics']['auc']:.4f}" if best_model['test_metrics']['auc'] is not None else 'N/A'
    print("\n" + "="*70)
    print(f"MCC-Focused Hyperparameter Tuning Completed Successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Best model (by MCC): lr={best_model['hyperparameters']['learning_rate']}, batch_size={best_model['hyperparameters']['batch_size']}")
    print(f"Best model metrics:")
    print(f"  MCC (PRIMARY METRIC): {best_model['test_metrics']['mcc']:.4f}")
    print(f"  Accuracy: {best_model['test_metrics']['accuracy']:.4f}")
    print(f"  F1 Score: {best_model['test_metrics']['pos_f1']:.4f}")
    print(f"  AUC: {auc_str}")
    print(f"  Confusion Matrix: TP={best_model['test_metrics']['tp']}, TN={best_model['test_metrics']['tn']}, FP={best_model['test_metrics']['fp']}, FN={best_model['test_metrics']['fn']}")
    print(f"\nMCC Statistics across all experiments:")
    print(f"  Mean MCC: {np.mean(mcc_scores):.4f}")
    print(f"  Best MCC: {max(mcc_scores):.4f}")
    print(f"  Experiments with MCC > 0: {sum(1 for mcc in mcc_scores if mcc > 0)}/{len(mcc_scores)}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()