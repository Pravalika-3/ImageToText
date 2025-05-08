import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
try:
    import pytesseract
except ImportError:
    st.error("Please install pytesseract: `pip install pytesseract`.")
    st.stop()
try:
    import PyPDF2
except ImportError:
    st.error("Please install PyPDF2: `pip install PyPDF2`.")
    st.stop()
try:
    from pdf2image import convert_from_bytes
except ImportError:
    st.error("Please install pdf2image: `pip install pdf2image`. Also ensure Poppler is installed (see Help).")
    st.stop()
try:
    from docx import Document
except ImportError:
    st.error("Please install python-docx: `pip install python-docx`.")
    st.stop()
try:
    import cv2
except ImportError:
    st.error("Please install opencv-python-headless: `pip install opencv-python-headless`.")
    st.stop()
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
import re
import pandas as pd
import os
import tempfile
import io
import numpy as np
import logging
from transformers import DistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if not os.path.exists(TESSERACT_PATH):
    st.error(f"Tesseract OCR not found at {TESSERACT_PATH}. Install from https://github.com/UB-Mannheim/tesseract/wiki.")
    st.stop()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

@st.cache_resource
def load_qa_model():
    try:
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    except Exception as e:
        st.error(f"Failed to load question-answering model: {str(e)}")
        return None

def check_image_quality(image):
    """Check if image resolution is sufficient for OCR."""
    width, height = image.size
    if width * height < 500 * 500:
        return False, f"Image resolution too low ({width}x{height}). Minimum 500x500 pixels recommended."
    return True, "Image quality sufficient."

def extract_pdf_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text())
        return text.strip() or "No text detected in PDF"
    except Exception as e:
        return f"PDF Text Extraction Error: {str(e)}"

def extract_pdf_images(pdf_file):
    try:
        images = convert_from_bytes(pdf_file.read())
        return [Image.frombytes("RGB", img.size, img.rgb) for img in images]
    except Exception as e:
        return f"PDF Image Extraction Error: {str(e)}"

def extract_docx_text(docx_file):
    try:
        doc = Document(docx_file)
        text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        return text.strip() or "No text detected in DOCX"
    except Exception as e:
        return f"DOCX Text Extraction Error: {str(e)}"

def extract_text_with_ocr(image, lang='eng'):
    try:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name)
            text = pytesseract.image_to_string(tmp.name, lang=lang)
        os.unlink(tmp.name)
        text = ' '.join(text.split())
        return text or "No text detected"
    except Exception as e:
        return f"OCR Error: {str(e)}"

def generate_image_description(image, processor, model, device):
    try:
        inputs = processor(image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_length=50, num_beams=5, min_length=5)
        return processor.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Image Description Error: {str(e)}"

def detect_language(text):
    language_markers = {
        'cyrillic': set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'),
        'cjk': set('的一是不了在人有我他这个们中来上大为和国地到以说时要就出'),
        'arabic': set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي'),
        'devanagari': set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह')
    }
    char_counts = {lang: 0 for lang in language_markers}
    for char in text.lower():
        for lang, char_set in language_markers.items():
            if char in char_set:
                char_counts[lang] += 1
    max_lang = max(char_counts.items(), key=lambda x: x[1], default=('unknown', 0))
    language_mapping = {
        'cyrillic': ('ru', 'Russian'),
        'cjk': ('zh-cn', 'Chinese/Japanese/Korean'),
        'arabic': ('ar', 'Arabic'),
        'devanagari': ('hi', 'Hindi'),
        'unknown': ('en', 'English (default)')
    }
    return language_mapping.get(max_lang[0], ('en', 'English (default)')) if max_lang[1] >= 5 else ('en', 'English (default)')

def get_tesseract_lang(lang_code):
    return {
        'en': 'eng', 'fr': 'fra', 'es': 'spa', 'de': 'deu', 'it': 'ita',
        'pt': 'por', 'ru': 'rus', 'zh-cn': 'chi_sim', 'ja': 'jpn', 'ko': 'kor',
        'ar': 'ara', 'hi': 'hin'
    }.get(lang_code, 'eng')

def extract_invoice_info(text):
    invoice_data = {
        'invoice_number': None, 'date': None, 'due_date': None,
        'total_amount': None, 'currency': None, 'sender': None,
        'recipient': None, 'items': []
    }
    patterns = {
        'invoice_number': [
            r'(?:Invoice|INVOICE).*?(?:#|No\.?|Number|NUM)[\s:]*([A-Z0-9][-A-Z0-9]*)',
            r'(?:Invoice|INVOICE)[\s:]*([A-Z0-9][-A-Z0-9]*)'
        ],
        'date': [
            r'(?:Date|DATE)[\s:]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(?:Date|DATE)[\s:]*(\d{1,2}[\s][A-Za-z]{3,9}[\s,]\d{2,4})'
        ],
        'due_date': [
            r'(?:Due|DUE)[\s:]*(?:Date|DATE)[\s:]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
            r'(?:Due|DUE)[\s:]*(?:Date|DATE)[\s:]*(\d{1,2}[\s][A-Za-z]{3,9}[\s,]\d{2,4})'
        ],
        'amount': [
            r'(?:Total|TOTAL|Amount|AMOUNT|Due|DUE)[\s:]*([£€$])\s*(\d+(?:,\d+)?\.?\d*)',
            r'(?:Total|TOTAL|Amount|AMOUNT|Due|DUE)[\s:]*(\d+(?:,\d+)?\.?\d*)\s*([£€$])',
            r'(?:Total|TOTAL|Amount|AMOUNT|Due|DUE)[\s:]*(\d+(?:,\d+)?\.?\d*)'
        ],
        'sender': [
            r'^([A-Za-z0-9][\w\s&\.,\']*(?:Inc|LLC|Ltd|Co|Corporation|Company|GmbH|SA|SL|BV|Limited)[\.,]?)$',
            r'^([A-Za-z0-9][\w\s&\.,\']{5,50})$'
        ],
        'recipient': [
            r'(?:Bill|BILL|Invoice|INVOICE)[\s](?:To|TO)[\s:]*([A-Za-z0-9][\w\s&\.,\']{5,50})',
            r'(?:To|TO)[\s:]*([A-Za-z0-9][\w\s&\.,\']{5,50})'
        ]
    }
    for key, pattern_list in patterns.items():
        if key in ['sender', 'recipient']:
            lines = text.split('\n')[:5] if key == 'sender' else [text]
            for line in lines:
                for pattern in pattern_list:
                    match = re.search(pattern, line.strip() if key == 'sender' else line, re.IGNORECASE)
                    if match:
                        invoice_data[key] = match.group(1)
                        break
                if invoice_data[key]:
                    break
        else:
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if key == 'amount':
                        invoice_data['total_amount'] = match.group(2 if len(match.groups()) > 1 else 1)
                        invoice_data['currency'] = match.group(1) if len(match.groups()) > 1 else None
                    else:
                        invoice_data[key] = match.group(1)
                    break
    item_section_match = re.search(r'(?:Item|ITEM|Description|DESCRIPTION|Service|SERVICE).*?(?:Amount|AMOUNT|Total|TOTAL)', text, re.DOTALL)
    if item_section_match:
        for item in re.findall(r'(\d+)\s+([A-Za-z0-9][\w\s\-&\.\']+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)|([A-Za-z0-9][\w\s\-&\.\']{5,50})\s+(\d+(?:\.\d+)?)', item_section_match.group(0)):
            if item[0]:
                invoice_data['items'].append({'quantity': item[0], 'description': item[1], 'unit_price': item[2], 'total': item[3]})
            else:
                invoice_data['items'].append({'description': item[4], 'total': item[5]})
    return invoice_data

def detect_form_fields(image, debug=False):
    """Detect form fields in an image using OCR bounding boxes."""
    fields = []
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        d = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        n_boxes = len(d['level'])
        if debug:
            logger.info(f"OCR Data: {d}")
        for i in range(n_boxes):
            if d['text'][i].strip():
                label = d['text'][i].lower()
                if any(keyword in label for keyword in ['name', 'date', 'address', 'phone', 'email', 'signature']):
                    x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                    fields.append({
                        'label': d['text'][i],
                        'x': x,
                        'y': y + h + 10,
                        'width': w,
                        'height': h
                    })
    except Exception as e:
        st.error(f"Error detecting form fields: {e}")
    return fields

def fill_form(image, fields, user_inputs):
    """Fill form fields on the image with user inputs."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for field in fields:
        label = field['label']
        label_key = re.sub(r'[^a-zA-Z0-9]', '', label.lower())
        for input_label, value in user_inputs.items():
            input_key = re.sub(r'[^a-zA-Z0-9]', '', input_label.lower())
            if input_key in label_key and value:
                draw.text((field['x'], field['y']), value, fill="black", font=font)
    return image

def split_text_into_chunks(text, max_tokens=400):
    """Split text into chunks suitable for DistilBERT (max 512 tokens, accounting for question)."""
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk += sentence + ". "
            current_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
            current_tokens = sentence_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def answer_contract_question(contract_text, question, qa_pipeline, debug=False):
    """Answer a question about the contract using NLP, handling long texts."""
    if debug:
        logger.info(f"Processing question: {question}")
        logger.info(f"Contract text length: {len(contract_text)} characters")
    
    try:
        if not question.strip():
            return "Error: Question is empty.", 0.0
        if not contract_text or len(contract_text.strip()) < 10:
            return "Error: Contract text is too short or empty.", 0.0
        if not qa_pipeline:
            return "Error: Question-answering model failed to load.", 0.0
        
        # Split text into chunks to handle long contexts
        chunks = split_text_into_chunks(contract_text)
        best_answer = None
        best_score = 0.0
        
        for i, chunk in enumerate(chunks):
            if debug:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:100]}...")
            try:
                result = qa_pipeline(question=question, context=chunk)
                if result['score'] > best_score:
                    best_answer = result['answer']
                    best_score = result['score']
                if debug:
                    logger.info(f"Chunk {i+1} result: {result}")
            except Exception as e:
                if debug:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                continue
        
        if best_answer:
            return best_answer, best_score
        else:
            return "No relevant answer found in the document.", 0.0
    
    except Exception as e:
        if debug:
            logger.error(f"QA Error: {str(e)}")
        return f"Error answering question: {str(e)}", 0.0

st.set_page_config(page_title="Document & Invoice Processor", layout="wide")

with st.sidebar:
    st.title("📄 Document Processor")
    uploaded_file = st.file_uploader("📤 Upload Document/Invoice", type=["png", "jpg", "jpeg", "pdf", "docx"])
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension in ['png', 'jpg', 'jpeg']:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Document", use_column_width=True)
        else:
            st.write(f"Uploaded {file_extension.upper()}: {uploaded_file.name}")

tab1, tab2, tab3, tab4 = st.tabs(["Invoice Understanding Bot", "Multilingual Document Processor", "Form Auto-Filler", "Contract Insight Tool"])

with tab1:
    st.title("📊 Invoice Understanding Bot")
    st.markdown("Extracts invoice details (amount, sender, due date, items) from PNG, JPG, PDF, or DOCX.")
    if uploaded_file:
        if st.button("🧾 Extract Invoice Details"):
            with st.spinner("Processing invoice..."):
                file_extension = uploaded_file.name.split('.')[-1].lower()
                text, images = "", []
                processor, model, device = load_blip_model()
                if file_extension in ['png', 'jpg', 'jpeg']:
                    image = Image.open(uploaded_file).convert("RGB")
                    text = extract_text_with_ocr(image)
                    images = [image]
                elif file_extension == 'pdf':
                    uploaded_file.seek(0)
                    text = extract_pdf_text(uploaded_file)
                    uploaded_file.seek(0)
                    image_result = extract_pdf_images(uploaded_file)
                    images = image_result if isinstance(image_result, list) else []
                    if text.startswith("PDF Text Extraction Error") and images:
                        text = extract_text_with_ocr(images[0])
                elif file_extension == 'docx':
                    text = extract_docx_text(uploaded_file)
                if not text.startswith(("OCR Error", "PDF Text Extraction Error", "DOCX Text Extraction Error")):
                    invoice_data = extract_invoice_info(text)
                    image_desc = generate_image_description(images[0], processor, model, device) if images else "No image available"
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📄 Raw Text")
                        st.info(text)
                        if images:
                            st.subheader("🖼️ Image Description")
                            st.success(image_desc)
                    with col2:
                        st.subheader("📝 Invoice Details")
                        for key, label in [
                            ('invoice_number', "Invoice Number"), ('date', "Date"), ('due_date', "Due Date"),
                            ('total_amount', "Total Amount"), ('sender', "Sender"), ('recipient', "Recipient")
                        ]:
                            value = invoice_data[key] or 'Not found'
                            if key == 'total_amount' and invoice_data['currency']:
                                value = f"{invoice_data['currency']}{value}"
                            st.markdown(f"**{label}:** {value}")
                    if invoice_data['items']:
                        st.subheader("📋 Line Items")
                        st.dataframe(pd.DataFrame(invoice_data['items']))
                    else:
                        st.subheader("📋 Line Items")
                        st.markdown("No line items detected.")
                    st.subheader("💡 Summary")
                    if all([invoice_data['sender'], invoice_data['total_amount'], invoice_data['date']]):
                        summary = f"Invoice from {invoice_data['sender']} dated {invoice_data['date']} for {invoice_data['currency'] or ''}{invoice_data['total_amount']}"
                        if invoice_data['due_date']:
                            summary += f", due by {invoice_data['due_date']}."
                        st.success(summary)
                    else:
                        st.warning("Incomplete invoice details.")
                else:
                    st.error(f"Processing failed: {text}")

with tab2:
    st.title("🌐 Multilingual Document Processor")
    st.markdown("Processes PNG, JPG, PDF, or DOCX in multiple languages for global media/legal workflows.")
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        text, images = "", []
        processor, model, device = load_blip_model()
        if file_extension in ['png', 'jpg', 'jpeg']:
            image = Image.open(uploaded_file).convert("RGB")
            text = extract_text_with_ocr(image)
            images = [image]
        elif file_extension == 'pdf':
            uploaded_file.seek(0)
            text = extract_pdf_text(uploaded_file)
            uploaded_file.seek(0)
            image_result = extract_pdf_images(uploaded_file)
            images = image_result if isinstance(image_result, list) else []
            if text.startswith("PDF Text Extraction Error") and images:
                text = extract_text_with_ocr(images[0])
        elif file_extension == 'docx':
            text = extract_docx_text(uploaded_file)
        if not text.startswith(("OCR Error", "PDF Text Extraction Error", "DOCX Text Extraction Error")):
            lang_code, language_name = detect_language(text)
            tesseract_lang = get_tesseract_lang(lang_code)
            st.info(f"Detected language: {language_name}")
            languages = {
                'eng': 'English', 'fra': 'French', 'spa': 'Spanish', 'deu': 'German',
                'ita': 'Italian', 'por': 'Portuguese', 'rus': 'Russian', 'chi_sim': 'Chinese (Simplified)',
                'jpn': 'Japanese', 'kor': 'Korean', 'ara': 'Arabic', 'hin': 'Hindi'
            }
            selected_lang = st.selectbox(
                "Confirm or select language:",
                options=list(languages.keys()),
                format_func=lambda x: languages.get(x, x),
                index=list(languages.keys()).index(tesseract_lang) if tesseract_lang in languages else 0
            )
            if st.button("🔍 Process Document"):
                with st.spinner(f"Processing in {languages.get(selected_lang, selected_lang)}..."):
                    if file_extension in ['png', 'jpg', 'jpeg', 'pdf'] and images:
                        text = extract_text_with_ocr(images[0], lang=selected_lang)
                    if not text.startswith(("OCR Error", "PDF Text Extraction Error", "DOCX Text Extraction Error")):
                        image_desc = generate_image_description(images[0], processor, model, device) if images else "No image available"
                        st.subheader("📄 Extracted Text")
                        st.info(text)
                        if images:
                            st.subheader("🖼️ Image Description")
                            st.success(image_desc)
                        st.subheader("📊 Analysis")
                        word_count = len(text.split())
                        is_form = len(re.findall(r'(?:Name|Date|Address|Phone|Email|Signature)[\s:]*___+', text, re.IGNORECASE)) > 0
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Word Count", word_count)
                            st.metric("Character Count", len(text))
                        with col2:
                            st.markdown(f"**Document Type:** {'Form' if is_form else 'General Document'}")
                            st.markdown(f"**Language:** {languages.get(selected_lang, selected_lang)}")
                        st.subheader("💡 Summary")
                        st.markdown(f"A {languages.get(selected_lang, selected_lang)} {'form' if is_form else 'document'} with {word_count} words.")
                    else:
                        st.error(f"Processing failed: {text}")
        else:
            st.error("Text detection failed. Try a clearer document.")

with tab3:
    st.title("📝 Form Auto-Filler")
    st.markdown("Upload a scanned form (PNG, JPG, or PDF) to extract fields and fill them digitally.")
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in ['png', 'jpg', 'jpeg', 'pdf']:
            st.error("Please upload a PNG, JPG, JPEG, or PDF file for form filling.")
        else:
            debug_mode = st.checkbox("Enable Debug Mode (Log OCR Data)")
            if st.button("📋 Detect Form Fields"):
                with st.spinner("Detecting form fields..."):
                    images = []
                    if file_extension in ['png', 'jpg', 'jpeg']:
                        form_image = Image.open(uploaded_file).convert("RGB")
                        is_valid, quality_message = check_image_quality(form_image)
                        if not is_valid:
                            st.error(quality_message)
                            st.stop()
                        images = [form_image]
                    elif file_extension == 'pdf':
                        uploaded_file.seek(0)
                        image_result = extract_pdf_images(uploaded_file)
                        images = image_result if isinstance(image_result, list) else []
                    if not images:
                        st.error("No images found for form filling.")
                        st.stop()
                    form_image = images[0]
                    is_valid, quality_message = check_image_quality(form_image)
                    if not is_valid:
                        st.error(quality_message)
                        st.stop()
                    fields = detect_form_fields(form_image, debug=debug_mode)
                    if not fields:
                        st.warning("No form fields detected. Try a different form with clear labels (e.g., Name, Date).")
                        st.stop()
                    st.subheader("Detected Form Fields")
                    df_fields = pd.DataFrame(fields)
                    st.dataframe(df_fields[['label', 'x', 'y']])
                    st.subheader("Enter Values")
                    with st.form(key="form_filler"):
                        user_inputs = {}
                        for i, field in enumerate(fields):
                            unique_key = f"{field['label']}_{i}"
                            user_inputs[field['label']] = st.text_input(
                                f"Enter {field['label']}",
                                key=unique_key
                            )
                        submit = st.form_submit_button("Fill Form")
                        if submit:
                            filled_image = fill_form(form_image.copy(), fields, user_inputs)
                            st.subheader("Filled Form")
                            st.image(filled_image, use_column_width=True)
                            img_byte_arr = io.BytesIO()
                            filled_image.save(img_byte_arr, format="PNG")
                            st.download_button(
                                label="Download Filled Form",
                                data=img_byte_arr.getvalue(),
                                file_name="filled_form.png",
                                mime="image/png"
                            )

with tab4:
    st.title("🔍 Contract Insight Tool")
    st.markdown("Upload a contract (PNG, JPG, PDF, or DOCX) and ask questions about its content.")
    
    # Initialize session state for extracted text and conversation history
    if 'extracted_contract_text' not in st.session_state:
        st.session_state.extracted_contract_text = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if uploaded_file:
        debug_mode = st.checkbox("Enable Debug Mode (Log QA Data)")
        
        # Step 1: Extract contract text
        if st.button("📜 Extract Contract Text"):
            with st.spinner("Extracting contract text..."):
                file_extension = uploaded_file.name.split('.')[-1].lower()
                text = ""
                images = []
                if file_extension in ['png', 'jpg', 'jpeg']:
                    image = Image.open(uploaded_file).convert("RGB")
                    is_valid, quality_message = check_image_quality(image)
                    if not is_valid:
                        st.error(quality_message)
                        st.stop()
                    text = extract_text_with_ocr(image)
                    images = [image]
                elif file_extension == 'pdf':
                    uploaded_file.seek(0)
                    text = extract_pdf_text(uploaded_file)
                    uploaded_file.seek(0)
                    image_result = extract_pdf_images(uploaded_file)
                    images = image_result if isinstance(image_result, list) else []
                    if text.startswith("PDF Text Extraction Error") and images:
                        is_valid, quality_message = check_image_quality(images[0])
                        if not is_valid:
                            st.error(quality_message)
                            st.stop()
                        text = extract_text_with_ocr(images[0])
                elif file_extension == 'docx':
                    text = extract_docx_text(uploaded_file)
                
                if not text.startswith(("OCR Error", "PDF Text Extraction Error", "DOCX Text Extraction Error")):
                    st.session_state.extracted_contract_text = text
                    with st.expander("View Extracted Contract Text"):
                        st.text_area("Contract Text", text, height=200)
                    st.subheader("Debug Info")
                    st.write(f"Extracted text length: {len(text)} characters")
                else:
                    st.error(f"Text extraction failed: {text}")
                    st.session_state.extracted_contract_text = None
        
        # Step 2: Ask a question if text is extracted
        if st.session_state.extracted_contract_text:
            st.subheader("Ask a Question")
            with st.form(key="question_form"):
                question = st.text_input("Enter your question (e.g., 'What’s the penalty clause?')", key="contract_question")
                submit_question = st.form_submit_button("Submit Question")
            
            if submit_question:
                if not question:
                    st.error("Please enter a question.")
                else:
                    with st.spinner("Processing question..."):
                        try:
                            if debug_mode:
                                st.write(f"Question submitted: {question}")
                            qa_pipeline = load_qa_model()
                            answer, score = answer_contract_question(
                                st.session_state.extracted_contract_text, question, qa_pipeline, debug=debug_mode
                            )
                            confidence_threshold = 0.1  
                            if score < confidence_threshold:
                                st.warning("Answer confidence is low. Try rephrasing the question or checking the document.")
                            st.markdown(f"**Answer:** {answer}\n\n**Confidence:** {score:.2%}")
                            st.session_state.conversation.append({"question": question, "answer": answer, "score": score})
                        except Exception as e:
                            st.error(f"Failed to process question: {str(e)}")
            
            # Display conversation history
            if st.session_state.conversation:
                st.subheader("Conversation History")
                for entry in st.session_state.conversation:
                    st.markdown(f"**Q:** {entry['question']}\n\n**A:** {entry['answer']} (Confidence: {entry['score']:.2%})")
                
                # Add buttons for clearing and downloading history
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Clear Conversation History", key="clear_history"):
                        st.session_state.conversation = []
                        st.experimental_rerun()
                with col2:
                    history_text = "\n\n".join(
                        f"Q: {entry['question']}\nA: {entry['answer']} (Confidence: {entry['score']:.2%})"
                        for entry in st.session_state.conversation
                    )
                    st.download_button(
                        label="Download Conversation History",
                        data=history_text,
                        file_name="conversation_history.txt",
                        mime="text/plain",
                        key="download_history"
                    )
        else:
            st.info("Please extract contract text before asking a question.")

with st.sidebar:
    with st.expander("Help & Instructions"):
        st.markdown("""
            ## How to Use
            1. Upload a document/invoice (PNG, JPG, PDF, or DOCX).
            2. Choose a tab:
               - **Invoice Understanding Bot**: Extracts invoice details and describes images.
               - **Multilingual Document Processor**: Processes multilingual documents and describes images.
               - **Form Auto-Filler**: Detects and fills form fields in scanned forms (PNG, JPG, PDF).
               - **Contract Insight Tool**: Extracts text from contracts and answers questions about content.
            3. Click process to analyze.
            ## Tips
            - Use clear, high-resolution images/documents (min 300 DPI).
            - Verify detected language for multilingual documents.
            - Ensure invoices show key fields (amount, sender, dates).
            - For forms, ensure fields like Name, Date, Address are clearly labeled.
            - For contracts, ask specific questions for best results.
            - Enable Debug Mode in Form Auto-Filler or Contract Insight Tool to log data for troubleshooting.
            ## Prerequisites
            - Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
            - Poppler: https://github.com/oschwartz10612/poppler-windows (Windows) or `apt-get install poppler-utils` (Linux) or `brew install poppler` (Mac)
            - Python packages: `pip install pytesseract pillow torch transformers streamlit pandas PyPDF2 pdf2image python-docx opencv-python-headless`
        """)