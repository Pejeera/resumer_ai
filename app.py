from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
from collections import Counter
import json
import requests
from PyPDF2 import PdfReader
import io
from werkzeug.utils import secure_filename
import os
from docx import Document

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ollama API endpoint (ใช้ localhost ถ้า Ollama รันอยู่ที่เครื่องเดียวกัน)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"  # ใช้โมเดลขนาดเล็กที่ติดตั้งอยู่แล้ว

# ฐานข้อมูลตำแหน่งงาน (เหลือ 1 ตำแหน่ง)
JOB_POSITIONS_DATABASE = [
    {
        "title": "Full-Stack Developer",
        "description": """We are looking for a Full-Stack Developer to join our team.
Requirements:
- Experience with JavaScript, Python, or similar programming languages
- Knowledge of front-end frameworks (React, Vue, Angular)
- Back-end development experience (Node.js, Django, Flask)
- Database knowledge (SQL, MongoDB)
- Experience with RESTful APIs
- Version control (Git)
- Problem-solving skills
- Ability to work in a team environment"""
    }
]

def call_llama(prompt, model=None, max_retries=2):
    """เรียกใช้ Llama 3.2 ผ่าน Ollama API (มี retry mechanism)"""
    # ใช้ model ที่ส่งมา หรือใช้ default
    selected_model = model if model else OLLAMA_MODEL
    
    # แปลง model name เป็น format ที่ Ollama ต้องการ
    # รองรับหลายรูปแบบ: llama-3.2-1b, llama3.2:1b, llama3.2-1b
    if selected_model == 'llama-3.2-1b' or selected_model == 'llama3.2:1b' or selected_model == 'llama3.2-1b':
        ollama_model = 'llama3.2:1b'
    elif selected_model == 'llama-3.2-latest' or selected_model == 'llama3.2:latest' or selected_model == 'llama3.2-latest':
        ollama_model = 'llama3.2:latest'  # ใช้ latest (3.2B)
    elif selected_model == 'llama-3.2-8b' or selected_model == 'llama3.2:8b' or selected_model == 'llama3.2-8b':
        # ใช้ llama3.1:8b เพราะ llama3.2:8b ยังไม่มีใน Ollama library
        ollama_model = 'llama3.1:8b'
    elif selected_model == 'llama-3-8b' or selected_model == 'llama3:8b' or selected_model == 'llama3-8b':
        ollama_model = 'llama3:8b'
    elif selected_model == 'gemma-3-4b' or selected_model == 'gemma3:4b' or selected_model == 'gemma3-4b':
        ollama_model = 'gemma3:4b'
    else:
        # ถ้าเป็น format อื่นที่ตรงกับ OLLAMA_MODEL หรือเป็น string ที่มี 'llama'
        if 'gemma' in str(selected_model).lower() and '4b' in str(selected_model).lower():
            ollama_model = 'gemma3:4b'
        elif 'llama3:8b' in str(selected_model).lower() or (selected_model.startswith('llama-3-8b')):
            ollama_model = 'llama3:8b'
        elif '8b' in str(selected_model).lower() or '8b' in str(selected_model):
            # ใช้ llama3.1:8b เพราะ llama3.2:8b ยังไม่มีใน Ollama library
            ollama_model = 'llama3.1:8b'
        elif 'latest' in str(selected_model).lower() or '3b' in str(selected_model).lower():
            ollama_model = 'llama3.2:latest'
        elif '1b' in str(selected_model).lower() or '1b' in str(selected_model):
            ollama_model = 'llama3.2:1b'
        else:
            ollama_model = OLLAMA_MODEL  # fallback to default
    
    # Log model ที่ใช้
    print(f"🤖 ใช้โมเดล: {ollama_model}")
    
    for attempt in range(max_retries + 1):
        try:
            payload = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,  # ลดเพื่อให้ดึงข้อมูลแม่นกว่าเดิม
                    "top_p": 0.9,
                    "top_k": 40,  # จำกัดคำตอบให้เลือกจาก top 40 tokens
                    "num_predict": 2048,  # maxTokens: 2048
                    "repeat_penalty": 1.15,  # ลดการซ้ำคำ
                    "num_ctx": 4096  # เพิ่ม context window
                }
            }
            
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
            response.raise_for_status()
            
            result = response.json()
            llama_response = result.get("response", "").strip()
            
            if llama_response:
                return llama_response
            else:
                if attempt < max_retries:
                    print(f"⚠️  Llama API return empty response, retrying... ({attempt + 1}/{max_retries})")
                    continue
                else:
                    print(f"⚠️  Llama API return empty response after {max_retries + 1} attempts")
                    return None
                    
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                print(f"⚠️  Timeout, retrying... ({attempt + 1}/{max_retries})")
                continue
            else:
                print(f"❌ Error calling {ollama_model}: Timeout after {max_retries + 1} attempts")
                return None
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                print(f"⚠️  Connection error, retrying... ({attempt + 1}/{max_retries})")
                continue
            else:
                print(f"❌ Error calling {ollama_model}: Connection error - {e}")
                print(f"   ตรวจสอบว่า Ollama service กำลังทำงานอยู่ที่ {OLLAMA_API_URL}")
                return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"⚠️  Request error, retrying... ({attempt + 1}/{max_retries})")
                continue
            else:
                print(f"❌ Error calling {ollama_model}: {e}")
                return None
    
    return None

def extract_text_from_pdf(pdf_file):
    """อ่านข้อความจากไฟล์ PDF"""
    try:
        text = ""
        pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def extract_text_from_docx(docx_file):
    """อ่านข้อความจากไฟล์ DOCX"""
    try:
        # Reset file pointer to beginning
        docx_file.seek(0)
        doc = Document(io.BytesIO(docx_file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return None

def clean_resume_text(text):
    """ทำความสะอาดและจัดรูปแบบ resume text เพื่อให้ Llama เข้าใจง่ายขึ้น"""
    if not text:
        return ""
    
    # ลบ whitespace ที่มากเกินไป แต่เก็บ newlines ไว้
    text = re.sub(r'[ \t]+', ' ', text)  # ลบ spaces/tabs ที่ซ้ำ
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # ลบบรรทัดว่างที่ซ้ำ
    
    # แยกส่วนสำคัญ (ถ้ามี) เพื่อให้ Llama เข้าใจโครงสร้าง
    sections = {}
    lines = text.split('\n')
    
    current_section = None
    section_content = []
    
    # Keywords สำหรับแต่ละ section
    section_keywords = {
        'summary': ['summary', 'objective', 'profile', 'เกี่ยวกับ', 'ประวัติ', 'overview'],
        'experience': ['experience', 'ประสบการณ์', 'work', 'employment', 'employment history', 'ประวัติการทำงาน'],
        'education': ['education', 'การศึกษา', 'qualification', 'qualifications', 'academic', 'การศึกษา'],
        'skills': ['skills', 'ทักษะ', 'technical skills', 'technical', 'ability', 'abilities', 'competencies', 'ความสามารถ']
    }
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        line_lower = line_stripped.lower()
        
        # ตรวจสอบว่าเป็น section header หรือไม่
        is_section_header = False
        for key, keywords in section_keywords.items():
            # ตรวจสอบว่า line นี้เป็น header (สั้นและมี keyword)
            if any(kw in line_lower for kw in keywords) and len(line_stripped) < 80:
                # ตรวจสอบว่าไม่ใช่เนื้อหา (เช่น "3 years of experience")
                if not any(char.isdigit() for char in line_stripped[:20]):
                    if current_section:
                        sections[current_section] = ' '.join(section_content)
                    current_section = key
                    section_content = []
                    is_section_header = True
                    break
        
        if not is_section_header:
            section_content.append(line_stripped)
    
    # เก็บ section สุดท้าย
    if current_section:
        sections[current_section] = ' '.join(section_content)
    
    # ถ้าแยก section ได้ ให้จัดรูปแบบใหม่
    if sections:
        formatted = []
        # เรียงลำดับตามความสำคัญ
        for key in ['summary', 'experience', 'education', 'skills']:
            if key in sections and sections[key]:
                formatted.append(f"=== {key.upper()} ===\n{sections[key]}")
        
        # เพิ่มส่วนอื่นๆ ที่ไม่ได้อยู่ใน categories หลัก
        other_content = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped and not any(
                any(kw in line_stripped.lower() for kw in keywords) 
                for keywords in section_keywords.values()
            ):
                other_content.append(line_stripped)
        
        if other_content:
            formatted.append(f"=== OTHER ===\n{' '.join(other_content[:10])}")  # จำกัดความยาว
        
        return '\n\n'.join(formatted) if formatted else text.strip()
    
    return text.strip()

def calculate_match_percentage(resume_text, jd_text, llama_result=None):
    """คำนวณ match_percentage จากข้อมูลจริง"""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # หาทักษะพื้นฐาน
    common_skills = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'vue', 'angular', 
                     'node.js', 'aws', 'docker', 'git', 'excel', 'power bi', 'tableau', 'machine learning',
                     'typescript', 'mongodb', 'postgresql', 'mysql', 'redis', 'kubernetes', 'jenkins',
                     'flask', 'django', 'express', 'spring', 'laravel', 'php', 'ruby', 'go', 'rust']
    
    # หา skills จาก resume
    resume_skills = [s.lower() for s in common_skills if s.lower() in resume_lower]
    
    # หา skills ที่ต้องการจาก job description
    jd_skills = [s.lower() for s in common_skills if s.lower() in jd_lower]
    
    # ใช้ skills จาก Llama ถ้ามี
    if llama_result and 'skills_detected' in llama_result:
        llama_skills = [s.lower() for s in llama_result['skills_detected'] if isinstance(s, str)]
        # รวม skills จาก resume และ Llama
        all_resume_skills = list(set(resume_skills + llama_skills))
    else:
        all_resume_skills = resume_skills
    
    # คำนวณ match
    matched_skills = set(all_resume_skills).intersection(set(jd_skills))
    missing_skills = set(jd_skills) - set(all_resume_skills)
    
    # คำนวณ percentage
    if jd_skills:
        base_percentage = int((len(matched_skills) / len(jd_skills)) * 100)
    else:
        base_percentage = 0
    
    # ปรับตาม strengths จาก Llama
    if llama_result and 'strengths' in llama_result:
        strengths = llama_result['strengths']
        if isinstance(strengths, list):
            strengths_count = len(strengths)
            # เพิ่ม 2-3% ต่อ strength (สูงสุด +15%)
            base_percentage += min(strengths_count * 3, 15)
        elif isinstance(strengths, str) and strengths:
            # ถ้าเป็น string ให้เพิ่ม 5%
            base_percentage += 5
    
    # ลดตาม skill_gaps
    if llama_result and 'skill_gaps' in llama_result:
        gaps = llama_result['skill_gaps']
        if isinstance(gaps, list):
            gaps_count = len(gaps)
            # ลด 2% ต่อ gap (สูงสุด -20%)
            base_percentage -= min(gaps_count * 2, 20)
        elif isinstance(gaps, str) and gaps:
            # ถ้าเป็น string ให้ลด 5%
            base_percentage -= 5
    
    # ลดตาม missing_skills
    if missing_skills:
        base_percentage -= min(len(missing_skills) * 2, 15)
    
    # จำกัดค่าระหว่าง 0-95%
    final_percentage = max(0, min(base_percentage, 95))
    
    return f"{final_percentage}%"

def enhance_llama_result(result, resume_text, jd_text):
    """ปรับปรุงและตรวจสอบผลลัพธ์จาก Llama ด้วยการตรวจสอบกับข้อมูลจริง"""
    if not result:
        return result
    
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # ตรวจสอบและปรับปรุง skills_detected
    if 'skills_detected' in result:
        verified_skills = []
        for skill in result['skills_detected']:
            if not isinstance(skill, str):
                continue
            skill_lower = skill.lower().strip()
            # ตรวจสอบว่ามี skill นี้ใน resume จริงหรือไม่
            # ตรวจสอบทั้งชื่อเต็มและคำสำคัญ
            skill_words = skill_lower.split()
            if (skill_lower in resume_lower or 
                any(word in resume_lower for word in skill_words if len(word) > 2)):
                verified_skills.append(skill)
        
        result['skills_detected'] = verified_skills
    
    # ตรวจสอบและปรับปรุง skill_gaps
    if 'skill_gaps' in result:
        verified_gaps = []
        for gap in result['skill_gaps']:
            if not isinstance(gap, str):
                continue
            gap_lower = gap.lower().strip()
            # ตรวจสอบว่ามี skill นี้ใน JD จริงหรือไม่
            gap_words = gap_lower.split()
            if (gap_lower in jd_lower or 
                any(word in jd_lower for word in gap_words if len(word) > 2)):
                verified_gaps.append(gap)
        result['skill_gaps'] = verified_gaps
    
    # ตรวจสอบและปรับปรุง strengths
    if 'strengths' in result:
        if isinstance(result['strengths'], list):
            # กรอง strengths ที่เกี่ยวข้องจริง
            verified_strengths = []
            for strength in result['strengths']:
                if isinstance(strength, str) and strength.strip():
                    verified_strengths.append(strength.strip())
            result['strengths'] = verified_strengths[:5]  # จำกัดไม่เกิน 5
    
    return result

def extract_personal_info_from_resume(resume_text):
    """ดึงข้อมูลส่วนตัวจาก Resume ด้วย regex (เข้มงวด - ดึงเฉพาะที่ปรากฏจริงเท่านั้น)
    
    กฎสำคัญ:
    1. ห้ามใช้ชื่อสถานที่, บริษัท, หน่วยงาน, โรงเรียน, โครงการ
    2. ต้องเป็นชื่อบุคคลที่มีความเป็นไปได้เท่านั้น
    3. ห้ามใช้หมวดหมู่, เมนู, หัวข้อ
    4. เบอร์โทรต้องเริ่มด้วย 0 หรือ +66 และมีตัวเลข 9-10 หลัก
    """
    personal_info = {
        'full_name': '',
        'email': '',
        'phone': ''
    }
    
    if not resume_text:
        return personal_info
    
    # รายชื่อสถานที่ที่พบบ่อย (ห้ามใช้แทนชื่อผู้สมัคร)
    common_places = [
        'bangkok', 'phuket', 'chiang mai', 'pattaya', 'hat yai', 'udon thani',
        'khon kaen', 'nakhon ratchasima', 'surat thani', 'rayong', 'chonburi',
        'bang bo', 'bang na', 'bang sue', 'bang rak', 'bang kapi', 'bang khae',
        'thailand', 'thai', 'asia', 'europe', 'america', 'province', 'จังหวัด',
        'เขต', 'อำเภอ', 'ตำบล', 'district', 'amphoe', 'tambon'
    ]
    
    # คำที่บ่งบอกว่าไม่ใช่ชื่อคน (บริษัท, หน่วยงาน, โรงเรียน, โครงการ, หมวดหมู่)
    non_person_keywords = [
        'company', 'corporation', 'corp', 'ltd', 'limited', 'co.,', 'co.,ltd',
        'บริษัท', 'จำกัด', 'มหาชน', 'องค์กร', 'หน่วยงาน', 'department', 'division',
        'school', 'university', 'college', 'institute', 'มหาวิทยาลัย', 'โรงเรียน',
        'project', 'program', 'โครงการ', 'โปรแกรม', 'section', 'section', 'หมวดหมู่',
        'menu', 'category', 'topic', 'subject', 'หัวข้อ', 'เรื่อง', 'title',
        'address', 'ที่อยู่', 'location', 'สถานที่', 'office', 'สำนักงาน'
    ]
    
    # หา name จาก resume (มักจะอยู่บรรทัดแรกๆ)
    # Pattern: ชื่อที่ขึ้นต้นด้วยตัวพิมพ์ใหญ่ ตามด้วยตัวพิมพ์เล็ก และมีอย่างน้อย 1-2 คำ
    name_patterns = [
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',  # ชื่อภาษาอังกฤษ (1-2 คำ)
        r'^([ก-๙]{2,}(?:\s+[ก-๙]{2,})?)',  # ชื่อภาษาไทย (1-2 คำ แต่ละคำอย่างน้อย 2 ตัวอักษร)
        r'^([ก-๙]{3,})',  # ชื่อภาษาไทย 1 คำ (อย่างน้อย 3 ตัวอักษร)
    ]
    
    # ลองหาชื่อจากบรรทัดแรกๆ ก่อน (มักจะอยู่ 15 บรรทัดแรก)
    lines = resume_text.split('\n')[:15]
    resume_start = '\n'.join(lines)
    
    for pattern in name_patterns:
        name_match = re.search(pattern, resume_start, re.MULTILINE)
        if name_match:
            candidate_name = name_match.group(1).strip()
            
            # ตรวจสอบความยาว (ชื่อควรมีความยาวสมเหตุสมผล)
            if len(candidate_name) < 2 or len(candidate_name) > 100:
                continue
            
            name_lower = candidate_name.lower()
            
            # ตรวจสอบว่าไม่ใช่ชื่อสถานที่
            is_place = any(place in name_lower for place in common_places)
            if is_place:
                continue
            
            # ตรวจสอบว่าไม่ใช่คำที่บ่งบอกว่าไม่ใช่ชื่อคน
            is_non_person = any(keyword in name_lower for keyword in non_person_keywords)
            if is_non_person:
                continue
            
            # ตรวจสอบว่าไม่ใช่คำที่ขึ้นต้นด้วย "Mr.", "Mrs.", "Ms.", "Dr." หรือคำอื่นๆ
            if re.match(r'^(Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam|นาย|นาง|นางสาว|ดร\.|อาจารย์)', candidate_name, re.IGNORECASE):
                continue
            
            # ตรวจสอบว่ามีตัวเลขหรือสัญลักษณ์พิเศษหรือไม่ (ถ้ามีอาจไม่ใช่ชื่อ)
            if re.search(r'[0-9@#$%^&*()_+=\[\]{}|;:,.<>?/\\]', candidate_name):
                continue
            
            # ตรวจสอบว่าไม่ใช่คำที่ขึ้นต้นด้วยตัวพิมพ์ใหญ่ทั้งหมด (อาจเป็นหัวข้อ)
            if candidate_name.isupper() and len(candidate_name.split()) > 2:
                continue
            
            # ตรวจสอบว่าไม่ใช่คำที่ขึ้นต้นด้วยตัวพิมพ์เล็ก (อาจไม่ใช่ชื่อ)
            if candidate_name[0].islower():
                continue
            
            # ตรวจสอบว่าไม่ใช่คำที่ยาวเกินไป (อาจเป็นประโยค)
            words = candidate_name.split()
            if len(words) > 4:
                continue
            
            # ตรวจสอบว่าแต่ละคำมีความยาวสมเหตุสมผล (ไม่ยาวเกินไป)
            if any(len(word) > 20 for word in words):
                continue
            
            # ตรวจสอบว่าไม่ใช่คำที่ขึ้นต้นด้วยตัวเลขหรือสัญลักษณ์
            if re.match(r'^[0-9\W]', candidate_name):
                continue
            
            personal_info['full_name'] = candidate_name
            break
    
    # หา email จาก resume (ต้องเป็นรูปแบบ name@example.com เท่านั้น)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    email_matches = re.findall(email_pattern, resume_text)
    if email_matches:
        # ใช้ email แรกที่พบ
        email = email_matches[0]
        # ตรวจสอบว่าเป็นรูปแบบที่ถูกต้อง (มี @ และ .)
        if '@' in email and '.' in email.split('@')[1]:
            personal_info['email'] = email
    
    # หา phone จาก resume (ต้องเริ่มด้วย 0 หรือ +66 และมีตัวเลข 9-10 หลัก)
    # รูปแบบที่รองรับ:
    # - 0[689]x-xxx-xxxx (10 หลัก: 0 + 9 หลัก) หรือ 0[689]x-xxx-xxx (9 หลัก: 0 + 8 หลัก)
    # - 0[689]xxxxxxxx (10 หลัก) หรือ 0[689]xxxxxxx (9 หลัก)
    # - +66[689]x-xxx-xxxx (12 หลัก: +66 + 9 หลัก) หรือ +66[689]x-xxx-xxx (11 หลัก: +66 + 8 หลัก)
    # - +66[689]xxxxxxxx (12 หลัก) หรือ +66[689]xxxxxxx (11 หลัก)
    phone_patterns = [
        r'0[689]\d{1}[-.\s]?\d{3}[-.\s]?\d{3,4}',  # 08x-xxx-xxxx หรือ 08x-xxx-xxx (9-10 หลัก)
        r'0[689]\d{7,8}',  # 08xxxxxxxx หรือ 08xxxxxxx (9-10 หลัก)
        r'\+66[-.\s]?[689]\d{1}[-.\s]?\d{3}[-.\s]?\d{3,4}',  # +66-8x-xxx-xxxx หรือ +66-8x-xxx-xxx (11-12 หลัก)
        r'\+66[-.\s]?[689]\d{7,8}',  # +66[689]xxxxxxxx หรือ +66[689]xxxxxxx (11-12 หลัก)
        r'\(?0[689]\d{1}\)?[-.\s]?\d{3}[-.\s]?\d{3,4}',  # (08x) xxx-xxxx หรือ (08x) xxx-xxx (9-10 หลัก)
    ]
    
    for pattern in phone_patterns:
        phone_matches = re.finditer(pattern, resume_text)
        for phone_match in phone_matches:
            phone = phone_match.group(0).strip()
            # ลบ whitespace, dash, dot, parentheses
            phone_clean = re.sub(r'[-.\s()]', '', phone)
            
            # ตรวจสอบว่าต้องเริ่มด้วย 0 หรือ +66
            if phone_clean.startswith('0'):
                # ต้องมี 9-10 หลัก (0 + 8-9 หลัก)
                if len(phone_clean) >= 9 and len(phone_clean) <= 10:
                    # ตรวจสอบว่าเป็นเบอร์โทรไทยที่ถูกต้อง (0 ตามด้วย 6, 8, หรือ 9)
                    if phone_clean[1] in ['6', '8', '9']:
                        personal_info['phone'] = phone
                        break
            elif phone_clean.startswith('+66'):
                # ต้องมี 11-12 หลัก (+66 + 8-9 หลัก)
                if len(phone_clean) >= 11 and len(phone_clean) <= 12:
                    # ตรวจสอบว่าเป็นเบอร์โทรไทยที่ถูกต้อง (66 ตามด้วย 6, 8, หรือ 9)
                    if phone_clean[3] in ['6', '8', '9']:
                        personal_info['phone'] = phone_clean
                        break
            elif phone_clean.startswith('66'):
                # ต้องมี 10-11 หลัก (66 + 8-9 หลัก)
                if len(phone_clean) >= 10 and len(phone_clean) <= 11:
                    # ตรวจสอบว่าเป็นเบอร์โทรไทยที่ถูกต้อง (66 ตามด้วย 6, 8, หรือ 9)
                    if phone_clean[2] in ['6', '8', '9']:
                        # แปลงเป็นรูปแบบ +66
                        personal_info['phone'] = '+' + phone_clean
                        break
        
        if personal_info['phone']:
            break
    
    return personal_info

def extract_personal_info_with_llama(resume_text, model=None):
    """ดึงข้อมูลส่วนตัวจาก Resume ด้วย Llama 3.2 Instruct
    
    ดึงข้อมูล:
    - Full name
    - Phone number
    - Email
    - Highest education level
    """
    if not resume_text:
        return {
            "name": None,
            "phone": None,
            "email": None,
            "education_level": None
        }
    
    # จำกัดความยาวเพื่อไม่ให้ prompt ยาวเกินไป
    resume_limited = resume_text[:2000] if len(resume_text) > 2000 else resume_text
    
    # Prompt สำหรับ Llama 3.2 ตามที่ระบุ
    prompt = f"""Information to extract from resume:

1. Full name 
   - Extract only the real full name
   - Avoid job titles, company names, or positions
   - If formatted like "Name – Position", extract only the name

2. Phone number 
   - Extract the actual phone number found in the resume

3. Email 
   - Extract valid email only

4. Highest education level
   - Extract the actual highest level (e.g., ปริญญาตรี, ปวส., ม.6)

Rules:
- Do not guess. If no information is found, return null.
- Return JSON only.

Resume text:

{resume_limited}"""
    
    response = call_llama(prompt, model=model)
    
    if not response:
        return {
            "name": None,
            "phone": None,
            "email": None,
            "education_level": None
        }
    
    # พยายามดึง JSON จาก response
    try:
        # หา JSON object
        start_idx = response.find('{')
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", response, 0)
        
        # นับ brackets เพื่อหา JSON object ที่สมบูรณ์
        bracket_count = 0
        in_string = False
        escape_next = False
        end_idx = start_idx
        
        for i in range(start_idx, len(response)):
            char = response[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
        
        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)
        
        # แปลง key names ให้ตรงกับ format ที่ต้องการ
        personal_info = {
            "name": result.get("name") or result.get("full_name") or result.get("fullName"),
            "phone": result.get("phone") or result.get("phone_number") or result.get("phoneNumber"),
            "email": result.get("email"),
            "education_level": result.get("education_level") or result.get("educationLevel") or result.get("highest_education")
        }
        
        return personal_info
        
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"⚠️  Error parsing personal info from Llama response: {str(e)[:100]}")
        return {
            "name": None,
            "phone": None,
            "email": None,
            "education_level": None
        }

def analyze_with_llama(resume_text, jd_text, job_title="", model=None):
    """ใช้ Llama 3.2 วิเคราะห์ Resume และ Job Description"""
    
    # ทำความสะอาด resume text ก่อน
    resume_clean = clean_resume_text(resume_text)
    jd_clean = jd_text.strip()
    
    # จำกัดความยาวเพื่อไม่ให้ prompt ยาวเกินไป (Llama 3.2:1b มี context limit)
    if len(resume_clean) > 2000:
        resume_clean = resume_clean[:2000] + "..."
    if len(jd_clean) > 1000:
        jd_clean = jd_clean[:1000] + "..."
    
    # ดึงข้อมูลส่วนตัวจาก Resume ด้วย Llama 3.2 ก่อน
    llama_personal_info = extract_personal_info_with_llama(resume_text, model=model)
    
    # ใช้ข้อมูลจาก Llama ถ้ามี ถ้าไม่มีให้ใช้ regex fallback
    personal_info = {
        'full_name': llama_personal_info.get('name') or '',
        'email': llama_personal_info.get('email') or '',
        'phone': llama_personal_info.get('phone') or '',
        'education_level': llama_personal_info.get('education_level') or ''
    }
    
    # ถ้าข้อมูลจาก Llama ไม่ครบ ให้ใช้ regex fallback
    if not personal_info['full_name'] or not personal_info['email'] or not personal_info['phone']:
        regex_personal_info = extract_personal_info_from_resume(resume_text)
        if not personal_info['full_name']:
            personal_info['full_name'] = regex_personal_info.get('full_name', '')
        if not personal_info['email']:
            personal_info['email'] = regex_personal_info.get('email', '')
        if not personal_info['phone']:
            personal_info['phone'] = regex_personal_info.get('phone', '')
    
    job_title_part = f"Job Title: {job_title}\n\n" if job_title else ""
    
    # Prompt ที่ปรับปรุงแล้ว - ใช้ prompt ใหม่ที่ชัดเจนและมีข้อมูลส่วนตัวที่ดึงมาแล้ว
    prompt = f"""คุณคือระบบวิเคราะห์ใบสมัครงาน (AI Recruitment Analyst)

หน้าที่ของคุณคือวิเคราะห์ Resume เทียบกับ Job Description แล้วตอบกลับในรูปแบบ JSON เท่านั้น  

ห้ามมีข้อความใด ๆ นอกเหนือจาก JSON ที่กำหนด

=====================================================================
🔐 ข้อมูลส่วนตัวจากระบบ (ดึงด้วย regex – ห้ามแก้ไขแม้แต่นิดเดียว)
=====================================================================

full_name: "{personal_info['full_name']}"

email: "{personal_info['email']}"

phone: "{personal_info['phone']}"

คำสั่งสำคัญ:

- ถ้าค่าเหล่านี้ "ไม่ว่าง" → ใช้ตามนี้ ห้ามแก้ไข ห้ามตีความใหม่

- ถ้าค่าว่าง → ให้ค้นหาจาก Resume เท่านั้น ห้ามเดาเอง

=====================================================================
กฎสำคัญ:
=====================================================================

1. ต้องตอบเป็น JSON **เท่านั้น**

2. ห้ามใช้ markdown เช่น ``` หรือ ### 

3. ทุกฟิลด์ต้องมีข้อมูล (ห้ามปล่อยว่าง)

4. skills_detected ต้องเป็นทักษะที่พบใน Resume จริง แต่สามารถปรับคำให้อ่านง่ายได้

5. strengths ต้องอย่างน้อย 3 รายการ

6. skill_gaps ต้องอย่างน้อย 2 รายการ

7. match_percentage ต้องมีรูปแบบ เช่น "85%"

8. recommendation ต้องมีคำแนะนำที่มีประโยชน์

9. ห้ามเดาข้อมูลส่วนตัว นอกจากกรณีที่ระบบส่งมาเป็นค่าว่าง

=====================================================================
📌 รูปแบบ JSON ที่ต้องส่งกลับ (ห้ามเปลี่ยนโครงสร้าง)
=====================================================================

{{
  "full_name": "string",
  "email": "string",
  "phone": "string",
  "summary": "string",
  "skills_detected": ["string"],
  "strengths": ["string"],
  "skill_gaps": ["string"],
  "match_percentage": "string",
  "why_suitable": "string",
  "recommendation": "string"
}}

{job_title_part}=====================================================================
📄 ข้อความ Resume
=====================================================================

{resume_clean}

=====================================================================
📄 ข้อความ Job Description
=====================================================================

{jd_clean}

=====================================================================
โปรดตอบกลับเป็น JSON ตามแบบด้านล่างเท่านั้น:
=====================================================================

{{
  "full_name": "",
  "email": "",
  "phone": "",
  "summary": "",
  "skills_detected": [],
  "strengths": [],
  "skill_gaps": [],
  "match_percentage": "",
  "why_suitable": "",
  "recommendation": ""
}}"""

    response = call_llama(prompt, model=model)
    
    if not response:
        if job_title:
            print(f"⚠️  {job_title}: Llama API ไม่ได้ response หรือ response เป็น empty")
        return None
    
    # พยายามดึง JSON จาก response (อาจมีข้อความอื่นปนอยู่)
    try:
        # หา JSON object โดยนับ brackets เพื่อหา JSON object ที่สมบูรณ์
        start_idx = response.find('{')
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", response, 0)
        
        # นับ brackets เพื่อหา JSON object ที่สมบูรณ์
        bracket_count = 0
        in_string = False
        escape_next = False
        end_idx = start_idx
        
        for i in range(start_idx, len(response)):
            char = response[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i + 1
                        break
        
        json_str = response[start_idx:end_idx]
        
        # ลอง parse JSON
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            # ถ้า parse ไม่ได้ ลองทำการ clean nested JSON strings
            # หาและแก้ไข fields ที่มี nested JSON string เช่น "summary": "{\"key\": \"value\"}"
            json_str_clean = json_str
            
            # แก้ไข nested JSON string ใน string fields
            for field in ['summary', 'why_suitable', 'recommendation']:
                # Pattern: "field": "{\"...\"}"
                pattern = rf'"{field}"\s*:\s*"(\{{[^"]*)"([^"]*)"'
                match = re.search(pattern, json_str_clean)
                if match:
                    # เอาเฉพาะ value ที่อยู่หลัง nested JSON
                    value_part = match.group(2) if len(match.groups()) > 1 else ""
                    # แทนที่ด้วย plain text
                    json_str_clean = re.sub(
                        rf'"{field}"\s*:\s*"\{{[^"]*"[^"]*"',
                        f'"{field}": "{value_part}"',
                        json_str_clean,
                        count=1
                    )
            
            # ลอง parse อีกครั้ง
            try:
                result = json.loads(json_str_clean)
            except json.JSONDecodeError:
                # ถ้ายัง parse ไม่ได้ ให้ extract fields แบบ manual
                result = {}
                # Extract string fields (รองรับ nested JSON string)
                for field in ['full_name', 'email', 'phone', 'summary', 'why_suitable', 'recommendation', 'match_percentage']:
                    # ลองหาแบบปกติก่อน (รองรับ multiline และ nested JSON)
                    # Pattern ที่รองรับ: "field": "value" หรือ "field": "{\"key\": \"value\"}"
                    pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.|\\n)*)"'
                    match = re.search(pattern, json_str, re.DOTALL)
                    if match:
                        value = match.group(1)
                        # ถ้ามี nested JSON ลอง extract value จาก nested JSON
                        if value.startswith('{'):
                            # ลอง parse nested JSON
                            try:
                                nested_json = json.loads(value)
                                # ถ้า parse ได้ ให้หาค่าแรกที่เป็น string
                                if isinstance(nested_json, dict):
                                    # หา value แรกที่เป็น string
                                    for v in nested_json.values():
                                        if isinstance(v, str) and v:
                                            value = v
                                            break
                                    # ถ้าไม่เจอ string value ให้ใช้ key แรก
                                    if value.startswith('{'):
                                        first_key = list(nested_json.keys())[0] if nested_json else ""
                                        value = first_key
                                elif isinstance(nested_json, str):
                                    value = nested_json
                            except:
                                # ถ้า parse ไม่ได้ ให้ลอง extract แบบ manual
                                # รูปแบบ 1: {"key": "value"}
                                nested_match = re.search(r'"([^"]+)"\s*:\s*"([^"]*)"', value)
                                if nested_match:
                                    value = nested_match.group(2)
                                else:
                                    # รูปแบบ 2: {"key": "value with spaces"}
                                    nested_match = re.search(r':\s*"([^"]*)"', value)
                                    if nested_match:
                                        value = nested_match.group(1)
                                    else:
                                        # ถ้ายังไม่เจอ ให้ลบ JSON structure ออก
                                        value = re.sub(r'^\{"[^"]*"\s*:\s*"', '', value)
                                        value = re.sub(r'"\s*\}$', '', value)
                        
                        # Clean up value - แก้ไข escape sequences
                        value = value.replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n').replace('\\r', '').strip()
                        
                        # ถ้ายังมี JSON structure เหลืออยู่ ให้ลบออก
                        if value.startswith('{') and value.endswith('}'):
                            # ลอง extract text จาก JSON
                            text_match = re.search(r':\s*"([^"]*)"', value)
                            if text_match:
                                value = text_match.group(1)
                            else:
                                # ลบ JSON structure ทั้งหมด
                                value = re.sub(r'^\{"[^"]*"\s*:\s*"', '', value)
                                value = re.sub(r'"\s*\}$', '', value)
                        
                        # ลบ escape characters ที่เหลือ
                        value = value.replace('\\"', '"').replace('\\\\', '\\')
                        
                        if value and value != '{"Job Description:':
                            result[field] = value
                    else:
                        # ลองหาแบบไม่ต้องมี quotes (สำหรับ match_percentage)
                        if field == 'match_percentage':
                            pattern = rf'"{field}"\s*:\s*"([^"]*)"|"{field}"\s*:\s*(\d+%)'
                            match = re.search(pattern, json_str)
                            if match:
                                result[field] = match.group(1) or match.group(2) or "0%"
                
                # Extract array fields
                for field in ['skills_detected', 'strengths', 'skill_gaps']:
                    pattern = rf'"{field}"\s*:\s*\[(.*?)\]'
                    match = re.search(pattern, json_str, re.DOTALL)
                    if match:
                        items_str = match.group(1)
                        # Extract items from array
                        items = re.findall(r'"((?:[^"\\]|\\.)*)"', items_str)
                        result[field] = [item.replace('\\"', '"').replace('\\\\', '\\') for item in items]
                    else:
                        result[field] = []
                
                # Log extracted fields for debugging
                if job_title:
                    extracted_fields = list(result.keys())
                    print(f"   Extracted fields: {', '.join(extracted_fields)}")
        
        # แปลง why_suitable และ recommendation จาก array เป็น string ถ้าเป็น array
        if 'why_suitable' in result and isinstance(result['why_suitable'], list):
            result['why_suitable'] = ' '.join(result['why_suitable'])
        if 'recommendation' in result and isinstance(result['recommendation'], list):
            result['recommendation'] = ' '.join(result['recommendation'])
        
        # ทำความสะอาด string fields - ลบ JSON string ที่เหลืออยู่
        for field in ['full_name', 'email', 'phone', 'why_suitable', 'recommendation', 'summary']:
            if field in result and isinstance(result[field], str):
                value = result[field]
                # ถ้ายังมี JSON structure อยู่ ให้ลบออก
                if value.startswith('{') or value.startswith('{"'):
                    # ลอง parse เป็น JSON
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            # หา value แรกที่เป็น string
                            for v in parsed.values():
                                if isinstance(v, str) and v:
                                    result[field] = v
                                    break
                    except:
                        # ถ้า parse ไม่ได้ ให้ลบ JSON structure ออก
                        # ลบ pattern: {"key": "value"}
                        cleaned = re.sub(r'^\{"[^"]*"\s*:\s*"', '', value)
                        cleaned = re.sub(r'"\s*\}$', '', cleaned)
                        # ลบ escape characters
                        cleaned = cleaned.replace('\\"', '"').replace('\\\\', '\\').replace('\\n', ' ').strip()
                        if cleaned and not cleaned.startswith('{'):
                            result[field] = cleaned
                
                # ลบ escape sequences ที่เหลือ
                result[field] = result[field].replace('\\n', ' ').replace('\\r', '').strip()

        # ตรวจสอบว่ามี fields ที่จำเป็นครบหรือไม่
        if not result or len(result) == 0:
            if job_title:
                print(f"⚠️  {job_title}: JSON ไม่ครบถ้วน (result is empty) ใช้ fallback analysis")
            return None
        
        # ตรวจสอบว่ามี fields สำคัญอย่างน้อย 1 field หรือไม่
        important_fields = ['summary', 'why_suitable', 'recommendation', 'match_percentage']
        has_important_field = any(field in result for field in important_fields)
        
        if not has_important_field:
            if job_title:
                print(f"⚠️  {job_title}: ไม่มี fields สำคัญ ใช้ fallback analysis")
            return None

        # คำนวณ match_percentage จากข้อมูลจริงเสมอ (แทนการใช้ค่าจาก Llama)
        # เพื่อให้แต่ละตำแหน่งได้คะแนนที่แตกต่างกันตามข้อมูลจริง
        calculated_percentage = calculate_match_percentage(resume_text, jd_text, result)
        result['match_percentage'] = calculated_percentage
        
        # Log ถ้าค่าที่คำนวณได้แตกต่างจากค่าจาก Llama (ถ้ามี)
        if job_title:
            # ลองหา match_percentage จาก response เพื่อเปรียบเทียบ
            match_pattern = r'"match_percentage"\s*:\s*"([^"]*)"|"match_percentage"\s*:\s*(\d+%)|match_percentage["\s:]+([0-9]+%)'
            match = re.search(match_pattern, response, re.IGNORECASE)
            if match:
                llama_percentage = (match.group(1) or match.group(2) or match.group(3) or "0%")
                if llama_percentage != calculated_percentage:
                    print(f"   💡 {job_title}: คำนวณ match_percentage จากข้อมูลจริง: {calculated_percentage} (Llama: {llama_percentage})")
            else:
                print(f"   💡 {job_title}: คำนวณ match_percentage จากข้อมูลจริง: {calculated_percentage}")
        
        # Extract personal info จาก resume ถ้ายังไม่มี
        if 'full_name' not in result or not result.get('full_name'):
            # ลองหา name จาก resume (มักจะอยู่บรรทัดแรกๆ)
            name_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', resume_text, re.MULTILINE)
            if name_match:
                result['full_name'] = name_match.group(1).strip()
            else:
                result['full_name'] = "Not specified"
        
        if 'email' not in result or not result.get('email'):
            # ลองหา email จาก resume
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
            if email_match:
                result['email'] = email_match.group(0)
            else:
                result['email'] = "Not specified"
        
        if 'phone' not in result or not result.get('phone'):
            # ลองหา phone จาก resume (รองรับหลายรูปแบบ)
            phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}', resume_text)
            if phone_match:
                result['phone'] = phone_match.group(0).strip()
            else:
                result['phone'] = "Not specified"
        
        # ตรวจสอบ fields อื่นๆ ที่จำเป็น
        required_fields = ['full_name', 'email', 'phone', 'summary', 'skills_detected', 'strengths', 'skill_gaps', 'why_suitable', 'recommendation']
        missing_fields = [f for f in required_fields if f not in result]
        if missing_fields:
            # เติม default values สำหรับ fields ที่ขาด
            if 'summary' not in result or not result.get('summary'):
                result['summary'] = "ผู้สมัครมีประสบการณ์และทักษะที่เกี่ยวข้อง"
            if 'skills_detected' not in result:
                result['skills_detected'] = []
            if 'strengths' not in result or len(result.get('strengths', [])) < 3:
                # ต้องมีอย่างน้อย 3 strengths
                if not result.get('strengths'):
                    result['strengths'] = []
                # ถ้ามีน้อยกว่า 3 ให้เพิ่ม
                while len(result.get('strengths', [])) < 3:
                    result['strengths'].append("มีทักษะและประสบการณ์ที่เกี่ยวข้อง")
            if 'skill_gaps' not in result or len(result.get('skill_gaps', [])) < 2:
                # ต้องมีอย่างน้อย 2 skill_gaps
                if not result.get('skill_gaps'):
                    result['skill_gaps'] = []
                # ถ้ามีน้อยกว่า 2 ให้เพิ่ม
                while len(result.get('skill_gaps', [])) < 2:
                    result['skill_gaps'].append("ควรพัฒนาทักษะเพิ่มเติม")
            if 'why_suitable' not in result or not result.get('why_suitable'):
                # ลองสร้าง why_suitable จากข้อมูลที่มี
                if result.get('strengths'):
                    strengths_str = ', '.join(result['strengths'][:3]) if isinstance(result['strengths'], list) else str(result['strengths'])
                    result['why_suitable'] = f"ผู้สมัครมีจุดแข็ง ได้แก่ {strengths_str}"
                else:
                    result['why_suitable'] = "ผู้สมัครมีทักษะและประสบการณ์ที่เกี่ยวข้องกับตำแหน่งนี้"
            if 'recommendation' not in result or not result.get('recommendation'):
                # ลองสร้าง recommendation จากข้อมูลที่มี
                if result.get('skill_gaps'):
                    gaps_str = ', '.join(result['skill_gaps'][:3]) if isinstance(result['skill_gaps'], list) else str(result['skill_gaps'])
                    result['recommendation'] = f"ผู้สมัครควรพัฒนาทักษะเพิ่มเติม ได้แก่ {gaps_str}"
                else:
                    result['recommendation'] = "ผู้สมัครเหมาะกับตำแหน่งนี้"
            
            if job_title:
                print(f"   เติม default values สำหรับ fields ที่ขาด: {', '.join(missing_fields)}")
        
        # ใช้ enhance_llama_result เพื่อตรวจสอบและปรับปรุงผลลัพธ์
        result = enhance_llama_result(result, resume_text, jd_text)
        
        return result
    except json.JSONDecodeError as e:
        # ถ้า parse ไม่ได้ ให้ใช้ fallback
        if job_title:
            print(f"⚠️  {job_title}: ไม่สามารถ parse JSON ได้")
            print(f"   Response preview: {response[:300]}...")
            print(f"   Error: {str(e)[:100]}")
        return None
    except Exception as e:
        # จัดการ error อื่นๆ
        if job_title:
            print(f"⚠️  {job_title}: เกิด error ในการ parse: {str(e)[:100]}")
        return None

# Global variable สำหรับเก็บ progress
analysis_progress = {
    'current': 0,
    'total': 0,
    'current_job': '',
    'status': 'idle'
}

def analyze_multiple_positions(resume_text, job_descriptions, model=None):
    """วิเคราะห์ Resume กับตำแหน่งงานหลายตำแหน่ง (ใช้ Llama ทั้งหมด)"""
    import time
    global analysis_progress
    
    # แสดงโมเดลที่ใช้
    if model == 'llama-3.2-1b' or model == 'llama3.2:1b':
        model_display = 'llama3.2:1b'
    elif model == 'llama-3.2-latest' or model == 'llama3.2:latest':
        model_display = 'llama3.2:latest (3.2B)'
    elif model == 'llama-3.2-8b' or model == 'llama3.2:8b':
        model_display = 'llama3.1:8b'  # ใช้ llama3.1:8b เพราะ llama3.2:8b ยังไม่มี
    elif model == 'llama-3-8b' or model == 'llama3:8b':
        model_display = 'llama3:8b'
    elif model == 'gemma-3-4b' or model == 'gemma3:4b':
        model_display = 'gemma3:4b'
    else:
        model_display = OLLAMA_MODEL
    
    results = []
    total_positions = len(job_descriptions)
    estimated_time_per_position = 45  # วินาที (ประมาณ 30-60 วินาทีต่อตำแหน่ง)
    estimated_total_time = total_positions * estimated_time_per_position
    initial_estimated_time = estimated_total_time  # เก็บเวลาเริ่มต้นไว้สำหรับ countdown
    
    # อัปเดต progress
    analysis_progress['total'] = total_positions
    analysis_progress['current'] = 0
    analysis_progress['status'] = 'analyzing'
    
    print("\n" + "="*60)
    print("🔍 เริ่มวิเคราะห์ Resume กับตำแหน่งงาน")
    print("="*60)
    print(f"📊 จำนวนตำแหน่งงานที่ต้องวิเคราะห์: {total_positions} ตำแหน่ง")
    print(f"🤖 AI Model: {model_display} (ใช้ทั้งหมด)")
    print(f"⏱️  เวลาที่คาดว่าจะใช้: ประมาณ {estimated_total_time // 60} นาที {estimated_total_time % 60} วินาที")
    print("-"*60)
    
    start_time = time.time()
    
    # ใช้ Llama วิเคราะห์ทุกตำแหน่ง
    for idx, jd_data in enumerate(job_descriptions):
        job_title = jd_data.get('title', f'ตำแหน่ง {idx + 1}')
        jd_text = jd_data.get('description', '')
        
        if not jd_text:
            continue
        
        # อัปเดต progress
        analysis_progress['current'] = idx + 1
        analysis_progress['current_job'] = job_title
        
        print(f"\n🔄 [{idx + 1}/{total_positions}] กำลังวิเคราะห์: {job_title}...")
        print(f"   ใช้ {model_display} วิเคราะห์...")
        
        # ใช้ Llama วิเคราะห์
        llama_result = analyze_with_llama(resume_text, jd_text, job_title, model=model)
        
        if llama_result:
            result = llama_result
            result['job_title'] = job_title
            result['job_index'] = idx
            
            # แปลง match_percentage เป็นตัวเลขเพื่อเรียงลำดับ
            try:
                match_num = int(result.get('match_percentage', '0').replace('%', ''))
                result['match_score'] = match_num
            except:
                result['match_score'] = 0
            
            results.append(result)
            
            elapsed = int(time.time() - start_time)
            remaining = initial_estimated_time - elapsed
            if remaining > 0:
                print(f"   ✅ {job_title}: {result.get('match_percentage', '0%')} ({model_display})")
                print(f"   ⏱️  ใช้เวลา: {elapsed} วินาที | เหลืออีกประมาณ {remaining // 60} นาที {remaining % 60} วินาที")
            else:
                print(f"   ✅ {job_title}: {result.get('match_percentage', '0%')} ({model_display})")
                print(f"   ⏱️  ใช้เวลา: {elapsed} วินาที")
        else:
            print(f"   ⚠️  {job_title}: ไม่สามารถใช้ {model_display} ได้")
            # ถ้า Llama ไม่ได้ ให้ใช้ fallback
            result = fallback_analysis(resume_text, jd_text)
            result['job_title'] = job_title
            result['job_index'] = idx
            try:
                match_num = int(result.get('match_percentage', '0').replace('%', ''))
                result['match_score'] = match_num
            except:
                result['match_score'] = 0
            results.append(result)
    
    # เรียงลำดับตามความเหมาะสม
    results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
    
    total_time = int(time.time() - start_time)
    
    print(f"\n📊 ผลการเรียงลำดับสุดท้าย:")
    for i, r in enumerate(results[:5], 1):
        print(f"   {i}. {r['job_title']}: {r.get('match_percentage', '0%')} ({model_display})")
    
    print("\n" + "="*60)
    print(f"✅ การวิเคราะห์เสร็จสมบูรณ์ (ใช้เวลา {total_time // 60} นาที {total_time % 60} วินาที)")
    print("="*60 + "\n")
    
    # Reset progress
    analysis_progress['status'] = 'completed'
    
    return results

def fallback_analysis(resume_text, jd_text):
    """Fallback analysis เมื่อ Llama ไม่สามารถใช้งานได้"""
    # ใช้วิธีง่ายๆ ในการวิเคราะห์
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # Extract personal info
    name_match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', resume_text, re.MULTILINE)
    full_name = name_match.group(1).strip() if name_match else "Not specified"
    
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    email = email_match.group(0) if email_match else "Not specified"
    
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}', resume_text)
    phone = phone_match.group(0).strip() if phone_match else "Not specified"
    
    # หาทักษะพื้นฐาน
    common_skills = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'vue', 'angular', 
                     'node.js', 'aws', 'docker', 'git', 'excel', 'power bi', 'tableau', 'machine learning']
    
    skills_detected = [s.title() for s in common_skills if s in resume_lower]
    jd_skills = [s.title() for s in common_skills if s in jd_lower]
    
    matched = set([s.lower() for s in skills_detected]).intersection(set([s.lower() for s in jd_skills]))
    gaps = set([s.lower() for s in jd_skills]) - set([s.lower() for s in skills_detected])
    
    match_percentage = int((len(matched) / len(jd_skills)) * 100) if jd_skills else 0
    
    # สร้าง strengths (อย่างน้อย 3 ข้อ)
    strengths = [f"มีทักษะด้าน {s.title()}" for s in list(matched)[:5]] if matched else ["มีประสบการณ์ที่เกี่ยวข้อง"]
    while len(strengths) < 3:
        strengths.append("มีทักษะและประสบการณ์ที่เกี่ยวข้อง")
    
    # สร้าง skill_gaps (อย่างน้อย 2 ข้อ)
    skill_gaps = [s.title() for s in list(gaps)[:10]] if gaps else ["ควรพัฒนาทักษะเพิ่มเติม"]
    while len(skill_gaps) < 2:
        skill_gaps.append("ควรพัฒนาทักษะเพิ่มเติม")
    
    why_suitable = ""
    if match_percentage >= 60:
        matched_skills_str = ", ".join([s.title() for s in list(matched)[:5]])
        why_suitable = f"ผู้สมัครมีทักษะที่ตรงกับความต้องการ ได้แก่ {matched_skills_str} ซึ่งเป็นทักษะสำคัญสำหรับตำแหน่งนี้"
    elif match_percentage >= 40:
        matched_skills_str = ", ".join([s.title() for s in list(matched)[:3]])
        why_suitable = f"ผู้สมัครมีทักษะพื้นฐานบางส่วนที่เกี่ยวข้อง ได้แก่ {matched_skills_str} แต่ยังขาดทักษะสำคัญบางอย่าง"
    else:
        why_suitable = "ผู้สมัครมีทักษะที่ตรงกับความต้องการน้อย ควรพัฒนาทักษะเพิ่มเติม"
    
    return {
        "full_name": full_name,
        "email": email,
        "phone": phone,
        "summary": "ผู้สมัครมีประสบการณ์และทักษะที่เกี่ยวข้อง",
        "skills_detected": skills_detected[:15],
        "strengths": strengths,
        "skill_gaps": skill_gaps,
        "match_percentage": f"{match_percentage}%",
        "why_suitable": why_suitable,
        "recommendation": f"ผู้สมัคร{'เหมาะ' if match_percentage >= 60 else 'อาจไม่เหมาะ'}กับตำแหน่งนี้" + (f" ควรพัฒนาด้าน {', '.join([s.title() for s in list(gaps)[:3]])}" if gaps else "")
    }

# สร้างโฟลเดอร์ uploads ถ้ายังไม่มี
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    """รับไฟล์ PDF หรือ DOCX และแปลงเป็นข้อความ"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'ไม่มีไฟล์'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
        
        filename_lower = file.filename.lower()
        
        # ตรวจสอบประเภทไฟล์
        if filename_lower.endswith('.pdf'):
            # อ่านข้อความจาก PDF
            resume_text = extract_text_from_pdf(file)
            file_type = 'PDF'
        elif filename_lower.endswith('.docx'):
            # อ่านข้อความจาก DOCX
            resume_text = extract_text_from_docx(file)
            file_type = 'DOCX'
        else:
            return jsonify({'error': 'ไฟล์ต้องเป็น PDF หรือ DOCX เท่านั้น'}), 400
        
        if not resume_text:
            return jsonify({'error': f'ไม่สามารถอ่านไฟล์ {file_type} ได้'}), 400
        
        return jsonify({
            'success': True,
            'resume_text': resume_text,
            'message': f'อ่านไฟล์ {file_type} สำเร็จ',
            'file_type': file_type
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """วิเคราะห์ Resume กับ Job Description เดียว"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'ไม่มีข้อมูล'}), 400
        
        resume_text = data.get('resume', '')
        jd_text = data.get('job_description', '')
        
        if not resume_text or not jd_text:
            return jsonify({'error': 'กรุณาระบุ Resume และ Job Description'}), 400
        
        # ใช้ Llama 3.2 วิเคราะห์
        result = analyze_with_llama(resume_text, jd_text)
        
        # ถ้า Llama ไม่สามารถใช้งานได้ ให้ใช้ fallback
        if not result:
            print("Llama API ไม่สามารถใช้งานได้ ใช้ fallback analysis")
            result = fallback_analysis(resume_text, jd_text)
        
        # ตรวจสอบว่ามีฟิลด์ที่จำเป็นครบหรือไม่
        required_fields = ['full_name', 'email', 'phone', 'summary', 'skills_detected', 'strengths', 'skill_gaps', 'match_percentage', 'why_suitable', 'recommendation']
        for field in required_fields:
            if field not in result:
                # เติม default values ตามประเภทของ field
                if field in ['skills_detected', 'strengths', 'skill_gaps']:
                    result[field] = []
                elif field == 'match_percentage':
                    result[field] = "0%"
                else:
                    # สำหรับ text fields ให้ใช้ข้อความที่มีความหมาย
                    if field == 'summary':
                        result[field] = "ผู้สมัครมีประสบการณ์และทักษะที่เกี่ยวข้อง"
                    elif field == 'why_suitable':
                        result[field] = "ผู้สมัครมีทักษะและประสบการณ์ที่เกี่ยวข้องกับตำแหน่งนี้"
                    elif field == 'recommendation':
                        result[field] = "ผู้สมัครควรพัฒนาทักษะเพิ่มเติมเพื่อให้เหมาะสมกับตำแหน่งนี้มากขึ้น"

        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/analyze-positions', methods=['POST'])
def analyze_positions():
    """วิเคราะห์ Resume กับตำแหน่งงานหลายตำแหน่ง"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'ไม่มีข้อมูล'}), 400
        
        resume_text = data.get('resume', '')
        job_descriptions = data.get('job_descriptions', [])
        
        if not resume_text:
            return jsonify({'error': 'กรุณาระบุ Resume'}), 400
        
        if not job_descriptions or len(job_descriptions) == 0:
            return jsonify({'error': 'กรุณาระบุตำแหน่งงานอย่างน้อย 1 ตำแหน่ง'}), 400
        
        # วิเคราะห์ทุกตำแหน่ง
        results = analyze_multiple_positions(resume_text, job_descriptions)
        
        # หาตำแหน่งที่เหมาะสมที่สุด
        best_match = results[0] if results else None
        
        return jsonify({
            'full_name': best_match.get('full_name', '') if best_match else '',
            'email': best_match.get('email', '') if best_match else '',
            'phone': best_match.get('phone', '') if best_match else '',
            'summary': best_match.get('summary', '') if best_match else '',
            'skills_detected': best_match.get('skills_detected', []) if best_match else [],
            'all_positions': results,
            'best_match': best_match,
            'total_positions': len(results)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """ดึง progress การวิเคราะห์"""
    global analysis_progress
    progress_percent = 0
    if analysis_progress['total'] > 0:
        progress_percent = int((analysis_progress['current'] / analysis_progress['total']) * 100)
    
    return jsonify({
        'progress': progress_percent,
        'current': analysis_progress['current'],
        'total': analysis_progress['total'],
        'current_job': analysis_progress['current_job'],
        'status': analysis_progress['status']
    }), 200

@app.route('/api/analyze-auto', methods=['POST'])
def analyze_auto():
    """วิเคราะห์ Resume อัตโนมัติกับทุกตำแหน่งในฐานข้อมูล"""
    global analysis_progress
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'ไม่มีข้อมูล'}), 400
        
        resume_text = data.get('resume', '')
        model = data.get('model', 'llama-3.2-1b')  # default เป็น llama-3.2-1b
        
        if not resume_text:
            return jsonify({'error': 'กรุณาระบุ Resume'}), 400
        
        # Reset progress
        analysis_progress = {
            'current': 0,
            'total': 0,
            'current_job': '',
            'status': 'idle'
        }
        
        # ใช้ตำแหน่งงานจากฐานข้อมูล
        results = analyze_multiple_positions(resume_text, JOB_POSITIONS_DATABASE, model=model)
        
        # กรองเฉพาะตำแหน่งที่มีความเหมาะสม >= 40%
        suitable_positions = [r for r in results if r.get('match_score', 0) >= 40]
        
        # หาตำแหน่งที่เหมาะสมที่สุด
        best_match = suitable_positions[0] if suitable_positions else results[0] if results else None
        
        # ดึงข้อมูลส่วนตัวจากผลลัพธ์แรก (ทุกตำแหน่งควรมีข้อมูลเดียวกัน)
        personal_info = {}
        if results:
            first_result = results[0]
            personal_info = {
                'full_name': first_result.get('full_name', ''),
                'email': first_result.get('email', ''),
                'phone': first_result.get('phone', '')
            }
        
        # สร้างผลลัพธ์แบบย่อ (ไม่รวม Job Description)
        suitable_positions_clean = []
        for pos in suitable_positions:
            suitable_positions_clean.append({
                'job_title': pos.get('job_title', ''),
                'match_percentage': pos.get('match_percentage', '0%'),
                'match_score': pos.get('match_score', 0),
                'summary': pos.get('summary', ''),
                'skills_detected': pos.get('skills_detected', []),
                'strengths': pos.get('strengths', []),
                'skill_gaps': pos.get('skill_gaps', []),
                'why_suitable': pos.get('why_suitable', ''),
                'recommendation': pos.get('recommendation', '')
            })
        
        best_match_clean = None
        if best_match:
            best_match_clean = {
                'job_title': best_match.get('job_title', ''),
                'match_percentage': best_match.get('match_percentage', '0%'),
                'match_score': best_match.get('match_score', 0),
                'summary': best_match.get('summary', ''),
                'skills_detected': best_match.get('skills_detected', []),
                'strengths': best_match.get('strengths', []),
                'skill_gaps': best_match.get('skill_gaps', []),
                'why_suitable': best_match.get('why_suitable', ''),
                'recommendation': best_match.get('recommendation', '')
            }
        
        return jsonify({
            'full_name': personal_info.get('full_name', ''),
            'email': personal_info.get('email', ''),
            'phone': personal_info.get('phone', ''),
            'summary': best_match_clean.get('summary', '') if best_match_clean else '',
            'skills_detected': best_match_clean.get('skills_detected', []) if best_match_clean else [],
            'suitable_positions': suitable_positions_clean,
            'best_match': best_match_clean,
            'total_analyzed': len(results),
            'total_suitable': len(suitable_positions_clean)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """แสดงรายการตำแหน่งงานทั้งหมดในฐานข้อมูล"""
    try:
        positions = []
        for idx, pos in enumerate(JOB_POSITIONS_DATABASE):
            positions.append({
                'id': idx + 1,
                'title': pos.get('title', ''),
                'description': pos.get('description', ''),
                'requirements': pos.get('description', '').split('\n') if pos.get('description') else []
            })
        
        return jsonify({
            'success': True,
            'total': len(positions),
            'positions': positions
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/analyze-detail', methods=['POST'])
def analyze_detail():
    """วิเคราะห์ Resume กับตำแหน่งงานและแสดงผลแบบละเอียด (เหมาะสำหรับเทส)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'ไม่มีข้อมูล'}), 400
        
        resume_text = data.get('resume', '')
        job_title = data.get('job_title', '')  # Optional: ระบุตำแหน่งเฉพาะ
        
        if not resume_text:
            return jsonify({'error': 'กรุณาระบุ Resume'}), 400
        
        # ถ้าระบุตำแหน่งเฉพาะ ให้วิเคราะห์เฉพาะตำแหน่งนั้น
        if job_title:
            # หาตำแหน่งที่ตรงกับ job_title
            selected_job = None
            for pos in JOB_POSITIONS_DATABASE:
                if pos.get('title', '').lower() == job_title.lower():
                    selected_job = pos
                    break
            
            if not selected_job:
                return jsonify({'error': f'ไม่พบตำแหน่งงาน: {job_title}'}), 404
            
            # วิเคราะห์เฉพาะตำแหน่งนี้
            result = analyze_with_llama(resume_text, selected_job.get('description', ''), selected_job.get('title', ''))
            
            if not result:
                result = fallback_analysis(resume_text, selected_job.get('description', ''))
            
            # เติมข้อมูลเพิ่มเติม
            result['job_title'] = selected_job.get('title', '')
            result['job_description'] = selected_job.get('description', '')
            
            return jsonify({
                'success': True,
                'full_name': result.get('full_name', ''),
                'email': result.get('email', ''),
                'phone': result.get('phone', ''),
                'resume_preview': resume_text[:200] + '...' if len(resume_text) > 200 else resume_text,
                'analysis': result,
                'job_info': {
                    'title': selected_job.get('title', ''),
                    'description': selected_job.get('description', '')
                }
            }), 200
        else:
            # วิเคราะห์กับทุกตำแหน่ง
            results = analyze_multiple_positions(resume_text, JOB_POSITIONS_DATABASE)
            
            # ดึงข้อมูลส่วนตัวจากผลลัพธ์แรก
            personal_info = {}
            if results:
                first_result = results[0]
                personal_info = {
                    'full_name': first_result.get('full_name', ''),
                    'email': first_result.get('email', ''),
                    'phone': first_result.get('phone', '')
                }
            
            # สร้างผลลัพธ์แบบละเอียด
            detailed_results = []
            for r in results:
                detailed_results.append({
                    'job_title': r.get('job_title', ''),
                    'match_percentage': r.get('match_percentage', '0%'),
                    'match_score': r.get('match_score', 0),
                    'summary': r.get('summary', ''),
                    'skills_detected': r.get('skills_detected', []),
                    'strengths': r.get('strengths', []),
                    'skill_gaps': r.get('skill_gaps', []),
                    'why_suitable': r.get('why_suitable', ''),
                    'recommendation': r.get('recommendation', ''),
                    'job_description': next(
                        (pos.get('description', '') for pos in JOB_POSITIONS_DATABASE 
                         if pos.get('title', '') == r.get('job_title', '')),
                        ''
                    )
                })
            
            # เรียงลำดับตาม match_score
            detailed_results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            
            # หาตำแหน่งที่ดีที่สุด
            best_match = detailed_results[0] if detailed_results else None
            
            return jsonify({
                'success': True,
                'full_name': personal_info.get('full_name', ''),
                'email': personal_info.get('email', ''),
                'phone': personal_info.get('phone', ''),
                'resume_preview': resume_text[:200] + '...' if len(resume_text) > 200 else resume_text,
                'total_positions': len(detailed_results),
                'best_match': best_match,
                'all_analyses': detailed_results,
                'ranking': [
                    {
                        'rank': idx + 1,
                        'job_title': r.get('job_title', ''),
                        'match_percentage': r.get('match_percentage', '0%'),
                        'match_score': r.get('match_score', 0)
                    }
                    for idx, r in enumerate(detailed_results)
                ]
            }), 200
            
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/extract-personal-info', methods=['POST'])
def extract_personal_info():
    """ดึงข้อมูลส่วนตัวจาก Resume (full_name, email, phone) - ดึงเฉพาะที่ปรากฏจริงเท่านั้น"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'ไม่มีข้อมูล'}), 400
        
        resume_text = data.get('resume', '')
        
        if not resume_text:
            return jsonify({'error': 'กรุณาระบุ Resume'}), 400
        
        # ดึงข้อมูลส่วนตัวจาก Resume
        personal_info = extract_personal_info_from_resume(resume_text)
        
        # ส่งกลับเป็น JSON ตามรูปแบบที่กำหนด
        return jsonify({
            'full_name': personal_info.get('full_name', ''),
            'email': personal_info.get('email', ''),
            'phone': personal_info.get('phone', '')
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

@app.route('/api/upload-and-analyze', methods=['POST'])
def upload_and_analyze():
    """อัปโหลด PDF และวิเคราะห์อัตโนมัติกับทุกตำแหน่งในฐานข้อมูล (API เดียว)"""
    global analysis_progress
    try:
        # ตรวจสอบว่ามีไฟล์หรือไม่
        if 'file' not in request.files:
            return jsonify({'error': 'ไม่มีไฟล์'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
        
        filename_lower = file.filename.lower()
        
        # ตรวจสอบประเภทไฟล์
        if filename_lower.endswith('.pdf'):
            # อ่านข้อความจาก PDF
            resume_text = extract_text_from_pdf(file)
            file_type = 'PDF'
        elif filename_lower.endswith('.docx'):
            # อ่านข้อความจาก DOCX
            resume_text = extract_text_from_docx(file)
            file_type = 'DOCX'
        else:
            return jsonify({'error': 'ไฟล์ต้องเป็น PDF หรือ DOCX เท่านั้น'}), 400
        
        # อ่าน model จาก form data (ถ้ามี)
        model = request.form.get('model', 'llama-3.2-1b')  # default เป็น llama-3.2-1b
        
        if not resume_text:
            return jsonify({'error': 'ไม่สามารถอ่านไฟล์ PDF ได้'}), 400
        
        # Reset progress
        analysis_progress = {
            'current': 0,
            'total': 0,
            'current_job': '',
            'status': 'idle'
        }
        
        # ใช้ตำแหน่งงานจากฐานข้อมูล
        results = analyze_multiple_positions(resume_text, JOB_POSITIONS_DATABASE, model=model)
        
        # กรองเฉพาะตำแหน่งที่มีความเหมาะสม >= 40%
        suitable_positions = [r for r in results if r.get('match_score', 0) >= 40]
        
        # หาตำแหน่งที่เหมาะสมที่สุด
        best_match = suitable_positions[0] if suitable_positions else results[0] if results else None
        
        # ดึงข้อมูลส่วนตัวจากผลลัพธ์แรก (ทุกตำแหน่งควรมีข้อมูลเดียวกัน)
        personal_info = {}
        if results:
            first_result = results[0]
            personal_info = {
                'full_name': first_result.get('full_name', ''),
                'email': first_result.get('email', ''),
                'phone': first_result.get('phone', '')
            }
        
        # สร้างผลลัพธ์แบบย่อ (ไม่รวม Job Description)
        suitable_positions_clean = []
        for pos in suitable_positions:
            suitable_positions_clean.append({
                'job_title': pos.get('job_title', ''),
                'match_percentage': pos.get('match_percentage', '0%'),
                'match_score': pos.get('match_score', 0),
                'summary': pos.get('summary', ''),
                'skills_detected': pos.get('skills_detected', []),
                'strengths': pos.get('strengths', []),
                'skill_gaps': pos.get('skill_gaps', []),
                'why_suitable': pos.get('why_suitable', ''),
                'recommendation': pos.get('recommendation', '')
            })
        
        best_match_clean = None
        if best_match:
            best_match_clean = {
                'job_title': best_match.get('job_title', ''),
                'match_percentage': best_match.get('match_percentage', '0%'),
                'match_score': best_match.get('match_score', 0),
                'summary': best_match.get('summary', ''),
                'skills_detected': best_match.get('skills_detected', []),
                'strengths': best_match.get('strengths', []),
                'skill_gaps': best_match.get('skill_gaps', []),
                'why_suitable': best_match.get('why_suitable', ''),
                'recommendation': best_match.get('recommendation', '')
            }
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'full_name': personal_info.get('full_name', ''),
            'email': personal_info.get('email', ''),
            'phone': personal_info.get('phone', ''),
            'resume_preview': resume_text[:200] + '...' if len(resume_text) > 200 else resume_text,
            'summary': best_match_clean.get('summary', '') if best_match_clean else '',
            'skills_detected': best_match_clean.get('skills_detected', []) if best_match_clean else [],
            'suitable_positions': suitable_positions_clean,
            'best_match': best_match_clean,
            'total_analyzed': len(results),
            'total_suitable': len(suitable_positions_clean)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

