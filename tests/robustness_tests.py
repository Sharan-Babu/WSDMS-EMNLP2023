"""
Robustness testing implementation based on the CheckList methodology
(Beyond Accuracy: Behavioral Testing of NLP Models with CheckList - Ribeiro et al., 2020)
"""

import argparse
import json
import os
from typing import List, Dict, Any

import numpy as np
import torch
from transformers import BertTokenizer

from HotCakeModels import HotCakeForSequenceClassification
from data_loader import tok2int_sent

class CheckListTester:
    """Implementation of CheckList testing methodology for WSDMS model."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        self.model = HotCakeForSequenceClassification.from_pretrained('bert-base-cased')
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        
    def predict(self, text: str) -> float:
        """Get model prediction for a single text."""
        inputs = tok2int_sent(text, self.tokenizer, max_seq_length=128)
        input_ids = torch.tensor([inputs[0]]).to(self.device)
        attention_mask = torch.tensor([inputs[1]]).to(self.device)
        token_type_ids = torch.tensor([inputs[2]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model({'input_ids': input_ids, 
                                'attention_mask': attention_mask,
                                'token_type_ids': token_type_ids})
            probs = torch.softmax(outputs, dim=1)
            return probs[0][1].item()  # Probability of being fake news

    def minimum_functionality_test(self) -> Dict[str, Any]:
        """Run minimum functionality tests."""
        results = {
            'vocabulary': [],
            'named_entities': [],
            'negation': []
        }
        
        # Vocabulary tests
        vocab_tests = [
            "This is a factual news article about politics.",
            "This article contains misleading information about politics.",
            "The president made an announcement today.",
            "The president made a false announcement today."
        ]
        
        for text in vocab_tests:
            pred = self.predict(text)
            results['vocabulary'].append({
                'text': text,
                'prediction': pred
            })
            
        # Named entity tests
        entity_tests = [
            "John Smith announced new policy.",
            "Jane Doe announced new policy.",
            "The event happened in New York.",
            "The event happened in Los Angeles."
        ]
        
        for text in entity_tests:
            pred = self.predict(text)
            results['named_entities'].append({
                'text': text,
                'prediction': pred
            })
            
        # Negation tests
        negation_tests = [
            "The company made a profit.",
            "The company did not make a profit.",
            "The study confirms the hypothesis.",
            "The study does not confirm the hypothesis."
        ]
        
        for text in negation_tests:
            pred = self.predict(text)
            results['negation'].append({
                'text': text,
                'prediction': pred
            })
            
        return results

    def invariance_test(self) -> Dict[str, Any]:
        """Run invariance tests."""
        results = {
            'location': [],
            'names': [],
            'numbers': [],
            'neutral_phrases': []
        }
        
        # Location perturbation tests
        base_text = "The incident occurred in New York."
        locations = ["Los Angeles", "Chicago", "Houston", "Miami"]
        
        base_pred = self.predict(base_text)
        for loc in locations:
            perturbed_text = f"The incident occurred in {loc}."
            pred = self.predict(perturbed_text)
            results['location'].append({
                'base_text': base_text,
                'perturbed_text': perturbed_text,
                'base_pred': base_pred,
                'perturbed_pred': pred
            })
            
        # Add more invariance tests here
        return results

    def directional_expectation_test(self) -> Dict[str, Any]:
        """Run directional expectation tests."""
        results = {
            'negation': [],
            'intensifiers': []
        }
        
        # Negation direction tests
        base_texts = [
            "The study confirms the results.",
            "The company reported profits.",
            "The official made the statement."
        ]
        
        for text in base_texts:
            base_pred = self.predict(text)
            negated_text = text.replace("confirms", "does not confirm")
            neg_pred = self.predict(negated_text)
            
            results['negation'].append({
                'base_text': text,
                'negated_text': negated_text,
                'base_pred': base_pred,
                'negated_pred': neg_pred
            })
            
        # Intensifier tests
        intensifiers = ["definitely", "absolutely", "completely", "totally"]
        base_text = "This is a true statement."
        base_pred = self.predict(base_text)
        
        for intensifier in intensifiers:
            intensified_text = f"This is {intensifier} a true statement."
            int_pred = self.predict(intensified_text)
            
            results['intensifiers'].append({
                'base_text': base_text,
                'intensified_text': intensified_text,
                'base_pred': base_pred,
                'intensified_pred': int_pred
            })
            
        return results

def main():
    parser = argparse.ArgumentParser(description='Run robustness tests on WSDMS model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--test_type', type=str, default='all',
                      choices=['all', 'mft', 'inv', 'dir'],
                      help='Type of test to run')
    parser.add_argument('--output_dir', type=str, default='test_results',
                      help='Directory to save test results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = CheckListTester(args.model_path)
    
    # Run tests based on argument
    results = {}
    if args.test_type in ['all', 'mft']:
        results['minimum_functionality'] = tester.minimum_functionality_test()
    if args.test_type in ['all', 'inv']:
        results['invariance'] = tester.invariance_test()
    if args.test_type in ['all', 'dir']:
        results['directional_expectation'] = tester.directional_expectation_test()
        
    # Save results
    output_file = os.path.join(args.output_dir, f'{args.test_type}_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Test results saved to {output_file}")

if __name__ == "__main__":
    main()
