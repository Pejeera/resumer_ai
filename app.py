from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
from collections import Counter
import json
import requests
import pdfplumber
import io
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ollama API endpoint (ใช้ localhost ถ้า Ollama รันอยู่ที่เครื่องเดียวกัน)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"  # ใช้โมเดลขนาดเล็กที่ติดตั้งอยู่แล้ว

# ฐานข้อมูลตำแหน่งงานที่พบบ่อย (เหลือ 3 ตำแหน่ง)
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
    },
    {
        "title": "Front-End Developer",
        "description": """Front-End Developer Position
Requirements:
- Strong knowledge of HTML, CSS, JavaScript
- Experience with React, Vue, or Angular
- Responsive design skills
- UI/UX understanding
- Version control (Git)
- Cross-browser compatibility
- Performance optimization"""
    },
    {
        "title": "Back-End Developer",
        "description": """Back-End Developer Role
Requirements:
- Strong programming skills (Python, Java, Node.js, or similar)
- Database design and optimization (SQL, NoSQL)
- API development (REST, GraphQL)
- Server management and deployment
- Security best practices
- System architecture knowledge
- Problem-solving abilities"""
    }
]

def call_llama(prompt, max_retries=2):
    """เรียกใช้ Llama 3.2 ผ่าน Ollama API (มี retry mechanism)"""
    for attempt in range(max_retries + 1):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
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
                print(f"❌ Error calling {OLLAMA_MODEL}: Timeout after {max_retries + 1} attempts")
                return None
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                print(f"⚠️  Connection error, retrying... ({attempt + 1}/{max_retries})")
                continue
            else:
                print(f"❌ Error calling {OLLAMA_MODEL}: Connection error - {e}")
                print(f"   ตรวจสอบว่า Ollama service กำลังทำงานอยู่ที่ {OLLAMA_API_URL}")
                return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"⚠️  Request error, retrying... ({attempt + 1}/{max_retries})")
                continue
            else:
                print(f"❌ Error calling {OLLAMA_MODEL}: {e}")
                return None
    
    return None

def extract_text_from_pdf(pdf_file):
    """อ่านข้อความจากไฟล์ PDF"""
    try:
        text = ""
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

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

def analyze_with_llama(resume_text, jd_text, job_title=""):
    """ใช้ Llama 3.2 วิเคราะห์ Resume และ Job Description"""
    
    job_title_part = f"ตำแหน่งงาน: {job_title}\n\n" if job_title else ""
    
    prompt = f"""คุณคือระบบประมวลผลใบสมัครงาน AI ที่ทำหน้าที่วิเคราะห์ Resume ของผู้สมัครและเปรียบเทียบกับ Job Description

กฎการทำงาน:
- ห้ามเพิ่มเติมข้อมูลที่ไม่มีใน resume
- วิเคราะห์จากข้อมูลจริงเท่านั้น
- อธิบายแบบชัดเจน กระชับ เข้าใจง่าย
- ตอบกลับในรูปแบบ JSON เท่านั้น

นี่คือข้อมูล Resume:
{resume_text}

{job_title_part}และนี่คือ Job Description:
{jd_text}

โปรดวิเคราะห์และตอบกลับในรูปแบบ JSON ตามโครงสร้างนี้เท่านั้น (ห้ามเพิ่มเติมฟิลด์อื่น):
{{
  "summary": "สรุปประวัติผู้สมัครแบบกระชับ",
  "skills_detected": ["skill1", "skill2", "skill3"],
  "strengths": ["จุดแข็ง1", "จุดแข็ง2"],
  "skill_gaps": ["ช่องว่าง1", "ช่องว่าง2"],
  "match_percentage": "75%",
  "why_suitable": "อธิบายว่าทำไมผู้สมัครเหมาะกับตำแหน่งนี้ โดยอ้างอิงจากทักษะและประสบการณ์ที่มีใน Resume",
  "recommendation": "คำแนะนำว่าผู้สมัครเหมาะกับตำแหน่งนี้หรือไม่ และควรพัฒนาอะไรเพิ่ม"
}}

ตอบกลับเฉพาะ JSON เท่านั้น ไม่ต้องมีข้อความอื่น:"""

    response = call_llama(prompt)
    
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
                for field in ['summary', 'why_suitable', 'recommendation', 'match_percentage']:
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
        
        # ทำความสะอาด why_suitable และ recommendation - ลบ JSON string ที่เหลืออยู่
        for field in ['why_suitable', 'recommendation', 'summary']:
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
        
        # ตรวจสอบ fields อื่นๆ ที่จำเป็น
        required_fields = ['summary', 'skills_detected', 'strengths', 'skill_gaps', 'why_suitable', 'recommendation']
        missing_fields = [f for f in required_fields if f not in result]
        if missing_fields:
            # เติม default values สำหรับ fields ที่ขาด
            if 'summary' not in result or not result.get('summary'):
                result['summary'] = "ผู้สมัครมีประสบการณ์และทักษะที่เกี่ยวข้อง"
            if 'skills_detected' not in result:
                result['skills_detected'] = []
            if 'strengths' not in result:
                result['strengths'] = []
            if 'skill_gaps' not in result:
                result['skill_gaps'] = []
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

def analyze_multiple_positions(resume_text, job_descriptions):
    """วิเคราะห์ Resume กับตำแหน่งงานหลายตำแหน่ง (ใช้ Llama ทั้งหมด)"""
    import time
    global analysis_progress
    
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
    print(f"🤖 AI Model: {OLLAMA_MODEL} (ใช้ทั้งหมด)")
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
        print(f"   ใช้ {OLLAMA_MODEL} วิเคราะห์...")
        
        # ใช้ Llama วิเคราะห์
        llama_result = analyze_with_llama(resume_text, jd_text, job_title)
        
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
                print(f"   ✅ {job_title}: {result.get('match_percentage', '0%')} ({OLLAMA_MODEL})")
                print(f"   ⏱️  ใช้เวลา: {elapsed} วินาที | เหลืออีกประมาณ {remaining // 60} นาที {remaining % 60} วินาที")
            else:
                print(f"   ✅ {job_title}: {result.get('match_percentage', '0%')} ({OLLAMA_MODEL})")
                print(f"   ⏱️  ใช้เวลา: {elapsed} วินาที")
        else:
            print(f"   ⚠️  {job_title}: ไม่สามารถใช้ {OLLAMA_MODEL} ได้")
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
        print(f"   {i}. {r['job_title']}: {r.get('match_percentage', '0%')} ({OLLAMA_MODEL})")
    
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
    
    # หาทักษะพื้นฐาน
    common_skills = ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'vue', 'angular', 
                     'node.js', 'aws', 'docker', 'git', 'excel', 'power bi', 'tableau', 'machine learning']
    
    skills_detected = [s.title() for s in common_skills if s in resume_lower]
    jd_skills = [s.title() for s in common_skills if s in jd_lower]
    
    matched = set([s.lower() for s in skills_detected]).intersection(set([s.lower() for s in jd_skills]))
    gaps = set([s.lower() for s in jd_skills]) - set([s.lower() for s in skills_detected])
    
    match_percentage = int((len(matched) / len(jd_skills)) * 100) if jd_skills else 0
    
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
        "summary": "ผู้สมัครมีประสบการณ์และทักษะที่เกี่ยวข้อง",
        "skills_detected": skills_detected[:15],
        "strengths": [f"มีทักษะด้าน {s.title()}" for s in list(matched)[:5]] if matched else ["มีประสบการณ์ที่เกี่ยวข้อง"],
        "skill_gaps": [s.title() for s in list(gaps)[:10]] if gaps else ["ไม่มีช่องว่างที่สำคัญ"],
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
    """รับไฟล์ PDF และแปลงเป็นข้อความ"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'ไม่มีไฟล์'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'ไม่ได้เลือกไฟล์'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'ไฟล์ต้องเป็น PDF เท่านั้น'}), 400
        
        # อ่านข้อความจาก PDF
        resume_text = extract_text_from_pdf(file)
        
        if not resume_text:
            return jsonify({'error': 'ไม่สามารถอ่านไฟล์ PDF ได้'}), 400
        
        return jsonify({
            'success': True,
            'resume_text': resume_text,
            'message': 'อ่านไฟล์ PDF สำเร็จ'
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
        required_fields = ['summary', 'skills_detected', 'strengths', 'skill_gaps', 'match_percentage', 'why_suitable', 'recommendation']
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
        results = analyze_multiple_positions(resume_text, JOB_POSITIONS_DATABASE)
        
        # กรองเฉพาะตำแหน่งที่มีความเหมาะสม >= 40%
        suitable_positions = [r for r in results if r.get('match_score', 0) >= 40]
        
        # หาตำแหน่งที่เหมาะสมที่สุด
        best_match = suitable_positions[0] if suitable_positions else results[0] if results else None
        
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
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'ไฟล์ต้องเป็น PDF เท่านั้น'}), 400
        
        # อ่านข้อความจาก PDF
        resume_text = extract_text_from_pdf(file)
        
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
        results = analyze_multiple_positions(resume_text, JOB_POSITIONS_DATABASE)
        
        # กรองเฉพาะตำแหน่งที่มีความเหมาะสม >= 40%
        suitable_positions = [r for r in results if r.get('match_score', 0) >= 40]
        
        # หาตำแหน่งที่เหมาะสมที่สุด
        best_match = suitable_positions[0] if suitable_positions else results[0] if results else None
        
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

