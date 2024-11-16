import os
import json
import torch

def process_politifact_data():
    # Read the raw text file
    with open('politifact_news_articles.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Process each line and create label dictionary
    labels_dict = {}
    processed_articles = []
    
    for line in lines:
        try:
            # Each line is tab-separated with ID and text
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
                
            article_id, article_text = parts[0], parts[1]
            
            # Create a filename for this article
            filename = f"{article_id}.json"
            
            # For now, let's assume articles with certain keywords are fake
            fake_keywords = ['viral', 'incredible', 'amazing', 'shocking', 'unbelievable', 'you won\'t believe']
            text_lower = article_text.lower()
            label = 1 if any(keyword in text_lower for keyword in fake_keywords) else 0
            
            # Store label
            labels_dict[filename] = label
            
            # Create processed article
            processed_article = {
                'claim': article_text[:200],  # First 200 chars as claim
                'evidence': [article_text],
                'label': 'fake' if label == 1 else 'real'
            }
            
            # Save individual article
            with open(os.path.join('raw', filename), 'w', encoding='utf-8') as f:
                json.dump(processed_article, f)
            
            processed_articles.append(filename)
            
        except Exception as e:
            print(f"Error processing line: {e}")
            continue
    
    # Save labels tensor
    torch.save(labels_dict, os.path.join('raw', 'politifact_labels.pt'))
    print(f"Processed {len(processed_articles)} articles")
    print(f"Real articles: {sum(1 for v in labels_dict.values() if v == 0)}")
    print(f"Fake articles: {sum(1 for v in labels_dict.values() if v == 1)}")

if __name__ == "__main__":
    # Create raw directory if it doesn't exist
    os.makedirs('raw', exist_ok=True)
    process_politifact_data()
