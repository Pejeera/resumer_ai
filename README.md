# ระบบประมวลผลใบสมัครงาน AI

ระบบวิเคราะห์ Resume ของผู้สมัครและเปรียบเทียบกับ Job Description เพื่อประเมินความเหมาะสมของตำแหน่งงาน

## คุณสมบัติ

- ✅ สรุปประวัติผู้สมัคร (Summary)
- ✅ ระบุ Skills หลักที่ผู้สมัครมี
- ✅ เทียบ Resume กับ Job Description
- ✅ คำนวณความตรงกันของทักษะ (Skill Match %)
- ✅ ระบุจุดแข็ง (Strengths)
- ✅ ระบุช่องว่างของทักษะ (Skill Gaps)
- ✅ แนะนำความเหมาะสมและสิ่งที่ควรพัฒนา

## การติดตั้ง

### 1. ติดตั้ง Ollama และ Llama 3.2

ระบบนี้ใช้ **Llama 3.2** ผ่าน Ollama API คุณต้องติดตั้ง Ollama ก่อน:

1. ดาวน์โหลดและติดตั้ง Ollama จาก: https://ollama.com
2. เปิด Terminal/PowerShell และรันคำสั่งเพื่อดาวน์โหลด Llama 3.2:
```bash
ollama pull llama3.2
```
หรือสำหรับโมเดลขนาดเล็ก (1B):
```bash
ollama pull llama3.2:1b
```

3. ตรวจสอบว่า Ollama ทำงานอยู่:
```bash
ollama serve
```

### 2. ติดตั้ง Python Dependencies

```bash
pip install -r requirements.txt
```

## การใช้งาน

**สำคัญ:** ต้องแน่ใจว่า Ollama กำลังทำงานอยู่ก่อน (รัน `ollama serve` ใน terminal อื่น)

1. รันเซิร์ฟเวอร์ Flask:
```bash
python app.py
```

2. เปิดเบราว์เซอร์ไปที่:
```
http://localhost:5000
```

3. วาง Resume และ Job Description ในช่องที่กำหนด

4. คลิกปุ่ม "วิเคราะห์ Resume" หรือกด Ctrl+Enter

5. ดูผลลัพธ์ในรูปแบบ JSON ที่แสดงบนหน้าเว็บ

## API Endpoint

### POST /api/analyze

วิเคราะห์ Resume และ Job Description

**Request Body:**
```json
{
  "resume": "ข้อความ Resume ของผู้สมัคร",
  "job_description": "ข้อความ Job Description"
}
```

**Response:**
```json
{
  "summary": "สรุปประวัติผู้สมัคร",
  "skills_detected": ["Skill1", "Skill2", ...],
  "strengths": ["จุดแข็ง1", "จุดแข็ง2", ...],
  "skill_gaps": ["ช่องว่าง1", "ช่องว่าง2", ...],
  "match_percentage": "75%",
  "recommendation": "คำแนะนำ"
}
```

## โครงสร้างโปรเจกต์

```
test_Llama3.2/
├── app.py                 # Flask backend API
├── templates/
│   └── index.html         # Frontend UI
├── requirements.txt       # Python dependencies
└── README.md             # เอกสารนี้
```

## หมายเหตุ

- **ระบบใช้ Llama 3.2 ผ่าน Ollama API** สำหรับการวิเคราะห์ที่แม่นยำ
- ระบบจะวิเคราะห์จากข้อมูลจริงใน Resume เท่านั้น ไม่ได้เพิ่มเติมข้อมูล
- ผลลัพธ์จะแสดงในรูปแบบ JSON ที่พร้อมใช้งาน
- ระบบรองรับทั้งภาษาไทยและภาษาอังกฤษ
- หาก Ollama ไม่สามารถใช้งานได้ ระบบจะใช้ fallback analysis แทน

## การแก้ไขปัญหา

### Ollama ไม่ทำงาน
- ตรวจสอบว่า Ollama ติดตั้งและรันอยู่: `ollama serve`
- ตรวจสอบว่าโมเดล llama3.2 ดาวน์โหลดแล้ว: `ollama list`
- หากใช้ port อื่น แก้ไข `OLLAMA_API_URL` ใน `app.py`

### เปลี่ยนโมเดล
แก้ไข `OLLAMA_MODEL` ใน `app.py` เป็นโมเดลที่ต้องการ เช่น:
- `llama3.2` (default)
- `llama3.2:1b` (ขนาดเล็ก)
- `llama3.2:3b` (ขนาดกลาง)

## License

MIT License

