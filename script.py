import argparse
import PyPDF2
import re
from transformers import BertTokenizer , BertModel,BertForSequenceClassification
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import shutil
import pandas as pd


label_mapping = {
    0: 'INFORMATION-TECHNOLOGY',\
    1: 'ENGINEERING',\
    2: 'BUSINESS-DEVELOPMENT',\
    3: 'SALES',\
    4: 'HR',\
    5: 'FITNESS',\
    6: 'ARTS', \
    7: 'ADVOCATE', \
    8: 'CONSTRUCTION', \
    9: 'AVIATION',\
    10: 'FINANCE', \
    11: 'CHEF', \
    12: 'ACCOUNTANT',\
    13: 'BANKING',\
    14: 'HEALTHCARE', \
    15: 'CONSULTANT', \
    16: 'PUBLIC-RELATIONS', \
    17: 'DESIGNER', \
    18: 'TEACHER',  \
    19: 'APPAREL', \
    20: 'DIGITAL-MEDIA', \
    21: 'AGRICULTURE', \
    22: 'AUTOMOBILE', \
    23: 'BPO'
    }






device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model_root = BertModel.from_pretrained('bert-base-cased')

# Define the model architecture with dropout and L2 regularization
class TextModel(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.3, l2_reg=1e-5):
        super(TextModel, self).__init__()
        self.bert = model_root
        self.intermediate_layer = nn.Linear(768, 512)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer added
        self.output_layer = nn.Linear(512, num_classes)
        
        # L2 regularization added to linear layers
        self.intermediate_layer.weight.data = nn.init.kaiming_normal_(self.intermediate_layer.weight.data)
        self.intermediate_layer.bias.data.fill_(0)
        self.output_layer.weight.data = nn.init.kaiming_normal_(self.output_layer.weight.data)
        self.output_layer.bias.data.fill_(0)
        
        self.l2_reg = l2_reg
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)[1]
        intermediate = self.intermediate_layer(outputs)
        intermediate = self.dropout(intermediate)  # Apply dropout
        logits = self.output_layer(intermediate)
        return logits

def predict(ids,masks,ckpt):
    model = TextModel(num_classes=24)
    model.to(device)
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    # Make predictions
    with torch.no_grad():
        outputs = model(ids, attention_mask=masks)
    prediction = torch.argmax(outputs, dim=1).tolist()
    return prediction
        
    

def text_to_tensor(text):
    # Tokenize the input text
    encoded_input = tokenizer.encode_plus(text, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=256)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    # Move tensors to the appropriate device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids , attention_mask
        


def preprocessing(resumeText):
    resumeText = resumeText.lower()
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub('[^a-zA-Z]', ' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main(args):
    
    filename = []
    category = []
    ROOT = "categorized_resume"
    model_path = args.ckpt
    file_path = args.file_path
    pdf_files = os.listdir(file_path)
    for pdf in tqdm(pdf_files):
        pdf_path = os.path.join(file_path,pdf)
        text = extract_text_from_pdf(pdf_path)
        resume = preprocessing(text)
        ids , masks = text_to_tensor(resume)
        prediction = predict(ids=ids, masks=masks, ckpt=model_path)
        pred_class = label_mapping[prediction[0]]
        
        category_dir = os.path.join(ROOT,pred_class)
        
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        new_file_path = os.path.join(category_dir, pdf)
        shutil.move(pdf_path, new_file_path)
        
        filename.append(pdf)
        category.append(pred_class)
    categorized_resumes = pd.DataFrame({"filename":filename, "category":category})
    categorized_resumes.to_csv("categorized_resumes.csv")
    print("###########complete#############")
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with command-line arguments")
    parser.add_argument("--ckpt", type=str, default="./model_ckpt/ckpt.pt", help="model checkpoint")
    parser.add_argument("--file_path", type=str, help="Resume file path")
    args = parser.parse_args()
    main(args)
