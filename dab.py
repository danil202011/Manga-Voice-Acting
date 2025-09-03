import os
import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes
from uuid import uuid4
import subprocess
import requests
from deepface import DeepFace
import logging
import base64
import re
import asyncio
import edge_tts
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TMP_DIR = 'tmp'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

USERS_FILE = 'users.txt'


OPENROUTER_API_KEY = "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"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


EDGE_TTS_VOICES = {
    "ru": {
        "male": "ru-RU-DmitryNeural",
        "female": "ru-RU-SvetlanaNeural",
        "unknown": "ru-RU-SvetlanaNeural"
    },
    "en": {
        "male": "en-US-ChristopherNeural",
        "female": "en-US-JennyNeural",
        "unknown": "en-US-JennyNeural"
    }
}

def load_users():
    if not os.path.exists(USERS_FILE):
        return set()
    try:
        with open(USERS_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except:
        return set()

def save_user(user_id):
    users = load_users()
    users.add(str(user_id))
    with open(USERS_FILE, 'w') as f:
        f.write('\n'.join(users))

def get_user_count():
    return len(load_users())


def check_openrouter_connection():
    """Проверка подключения к OpenRouter API"""
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        test_payload = {
            "model": "qwen/qwen-vl-plus",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=test_payload, timeout=10)
        return response.status_code == 200
            
    except Exception as e:
        print(f"Connection error: {e}")
        return False


def openrouter_ocr(image_path):
    """Распознавание текста с изображения через OpenRouter"""
    try:
        if not check_openrouter_connection():
            return "API_CONNECTION_ERROR"
        
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "qwen/qwen-vl-plus",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Extract all text from this image exactly as it appears. Return only the raw text without any formatting, comments or explanations."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if text and text.strip():
                text = text.strip()
                
                text = fix_russian_text(text)
                
                if text and len(text) > 2:
                    print(f"OCR Success: {text[:100]}...")
                    return text[:2000]
        
        return ""
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return "ERROR"

def fix_russian_text(text):
    """Исправление русских символов"""
 
    char_replacements = {
        '�': '?',
        '？': '?',
        '�': '!',
        '！': '!',
        '�': '.',
        '．': '.',
        '…': '...',
        'â': 'а', 'Ã': 'А',
        'â': 'а', 'ã': 'а',
        'å': 'а', 'Å': 'А',
        'ç': 'с', 'Ç': 'С',
        'è': 'е', 'È': 'Е',
        'é': 'е', 'É': 'Е',
        'ê': 'е', 'Ê': 'Е',
        'ë': 'е', 'Ë': 'Е',
        'ì': 'и', 'Ì': 'И',
        'í': 'и', 'Í': 'И',
        'î': 'и', 'Î': 'И',
        'ï': 'и', 'Ï': 'И',
        'ð': 'о', 'Ð': 'О',
        'ñ': 'н', 'Ñ': 'Н',
        'ò': 'о', 'Ò': 'О',
        'ó': 'о', 'Ó': 'О',
        'ô': 'о', 'Ô': 'О',
        'õ': 'о', 'Õ': 'О',
        'ö': 'о', 'Ö': 'О',
        'ù': 'у', 'Ù': 'У',
        'ú': 'у', 'Ú': 'У',
        'û': 'у', 'Û': 'У',
        'ü': 'у', 'Ü': 'У',
        'ý': 'ы', 'Ý': 'Ы',
        'þ': 'р', 'Þ': 'Р',
        'ÿ': 'я', 'Ÿ': 'Я'
    }
    
    for wrong_char, correct_char in char_replacements.items():
        text = text.replace(wrong_char, correct_char)
    
  
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def detect_gender(image_path):
    """Определение пола человека на фото с помощью DeepFace"""
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['gender'],
            enforce_detection=False,
            detector_backend='retinaface',
            silent=True
        )
        
        if isinstance(analysis, list):
            analysis = analysis[0]
            
        gender = analysis.get('gender', {})
        woman_score = gender.get('Woman', 0)
        man_score = gender.get('Man', 0)
        
        if woman_score > man_score:
            return "female"
        elif man_score > woman_score:
            return "male"
        else:
            return "unknown"
            
    except Exception as e:
        print(f"Gender detection error: {e}")
        return "unknown"


def detect_language_simple(text):
    """Простой детектор языка на основе символов"""
    russian_chars = len(re.findall(r'[а-яА-ЯёЁ]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    if russian_chars > english_chars:
        return "ru"
    elif english_chars > russian_chars:
        return "en"
    else:
        return "en"

async def text_to_speech_edge_tts_fast(text, output_path, voice_name):
    """Быстрая озвучка через Edge-TTS без задержек"""
    try:
        print(f"Using Edge-TTS voice: {voice_name}")
        
        communicate = edge_tts.Communicate(text, voice_name, rate="+15%")
        await communicate.save(output_path)
        
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
        
    except Exception as e:
        print(f"Edge-TTS error: {e}")
        return False

async def text_to_speech_anime(text, output_path, gender="unknown", voice_choice=None):
    """Основная функция озвучки с выбором голоса"""
    print(f"Generating voice for text: {text[:50]}...")
    

    language = detect_language_simple(text)
    print(f"Text language: {language}")
    

    voice_name = None
    if voice_choice:
        voice_name = voice_choice
        print(f"Using user-selected voice: {voice_name}")
    else:
        voice_dict = EDGE_TTS_VOICES.get(language, EDGE_TTS_VOICES["en"])
        voice_name = voice_dict.get(gender, voice_dict["unknown"])
        print(f"Using auto-selected voice: {voice_name}")
    
    return await text_to_speech_edge_tts_fast(text, output_path, voice_name)


def split_text_into_chunks(text, max_chars=200):
    """Умная разбивка текста по предложениям"""
    if len(text) <= max_chars:
        return [text]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def create_logo_image(width=1280, height=720):
    """Создание прозрачного логотипа"""
    # Создаем прозрачное изображение
    logo = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    
    try:
      
        font = ImageFont.truetype("arial.ttf", 40)
    except:
       
        font = ImageFont.load_default()
    
    # Текст логотипа
    text = "Manga Voice Acting"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    
    x = width - text_width - 20
    y = 20
    
    
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 128))  # Белый с прозрачностью
    
    return cv2.cvtColor(np.array(logo), cv2.COLOR_RGBA2BGRA)

def overlay_logo(background, logo):
    """Наложение логотипа на кадр (исправленная версия)"""
   
    if len(background.shape) == 3 and background.shape[2] == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

   
    logo_alpha = logo[:, :, 3] / 255.0
    background_alpha = 1.0 - logo_alpha

   
    for c in range(0, 3):
        background[:, :, c] = (logo_alpha * logo[:, :, c] +
                              background_alpha * background[:, :, c])

    result = cv2.cvtColor(background, cv2.COLOR_BGRA2BGR)
    return result

# -------------------------

def create_karaoke_frame(text, current_char_index, width=1280, height=720):
    """Создание кадра с подсветкой текста и логотипом (исправлено для русского языка)"""
   
    frame_bgr = np.zeros((height, width, 3), dtype=np.uint8)
  
    frame_pil = Image.fromarray(frame_bgr)
    draw = ImageDraw.Draw(frame_pil)
    
    
    try:
       
        font_paths = [
            "arial.ttf",
            "Arial Unicode MS.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf", 
            "C:/Windows/Fonts/arial.ttf" 
        ]
        font_size = 40
        font = None
        
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                print(f"Using font: {path}")
                break
            except IOError:
                continue
                
        
        if font is None:
            font = ImageFont.load_default()
            print("Using default font (no Russian support)")
    except Exception as e:
        print(f"Font loading error: {e}")
        font = ImageFont.load_default()

    max_width = width - 100
    line_height = 50
    max_lines = 4

 
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        
        test_line = " ".join(current_line + [word])
        
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width > max_width and current_line:
            lines.append(" ".join(current_line))
            current_line = [word]
        else:
            current_line.append(word)
    
    if current_line:
        lines.append(" ".join(current_line))

   
    if len(lines) > max_lines:
        lines = lines[:max_lines]

   
    total_text_height = len(lines) * line_height
    y_start = (height - total_text_height) // 2

   
    char_count = 0
    
    
    for i, line in enumerate(lines):
        
        line_bbox = draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        x_start = (width - line_width) // 2
        y_pos = y_start + i * line_height
        
       
        bg_rect = [x_start - 10, y_pos - 5, x_start + line_width + 10, y_pos + line_height - 5]
        draw.rectangle(bg_rect, fill=(0, 0, 0, 128))  
        
     
        current_x = x_start
        
        
        for char in line:
           
            color = (0, 255, 255) if char_count < current_char_index else (255, 255, 255)
            
            # Рисуем символ
            draw.text((current_x, y_pos), char, font=font, fill=color)
            
           
            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            current_x += char_width
            char_count += 1
        
      
        char_count += 1
    
    
    frame = np.array(frame_pil)
    
   
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Добавляем логотип
    logo = create_logo_image(width, height)
    if logo is not None:
        frame = overlay_logo(frame, logo)
    
    return frame

def get_audio_duration(audio_path):
    """Точное определение длительности аудио"""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ], capture_output=True, text=True, timeout=10)
        duration = float(result.stdout.strip())
        return max(duration, 0.1)
    except:
        return 5.0

def create_karaoke_video_with_audio(text, audio_path, output_path):
    """Создание караоке-видео с точной синхронизацией"""
    try:
        if not os.path.exists(audio_path):
            return False
        
        duration = get_audio_duration(audio_path)
        if duration <= 0:
            return False
            
        fps = 30
        total_frames = int(duration * fps) + 10
        
        width, height = 1280, 720
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        if not out.isOpened():
            return False

        total_chars = len(text)
        chars_per_sec = total_chars / duration if duration > 0 else total_chars / 5.0

        for frame_num in range(total_frames):
            current_time = frame_num / fps
            current_char = min(int(current_time * chars_per_sec), total_chars)
            frame = create_karaoke_frame(text, current_char, width, height)
            out.write(frame)

        out.release()

        
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-shortest',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=120)

        if os.path.exists(temp_video):
            os.remove(temp_video)

        return result.returncode == 0 and os.path.exists(output_path)

    except Exception as e:
        print(f"Video creation error: {e}")
        return False


def merge_videos(video_paths: List[str], output_path: str) -> bool:
    """Объединение нескольких видео в одно"""
    try:
        if not video_paths:
            return False
            
        if len(video_paths) == 1:
            import shutil
            shutil.copy2(video_paths[0], output_path)
            return True
            
        list_file = os.path.join(TMP_DIR, f"concat_list_{uuid4().hex}.txt")
        with open(list_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                abs_path = os.path.abspath(video_path).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")
        
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', list_file,
            '-c', 'copy', '-y', output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if os.path.exists(list_file):
            os.remove(list_file)
        
        return result.returncode == 0 and os.path.exists(output_path)
            
    except Exception as e:
        print(f"Video merge error: {e}")
        return False


async def handle_single_photo(update: Update, context, image_path: str, gender: str) -> Dict:
    """Обработка одного фото и создание видео"""
    result = {'videos': [], 'audios': [], 'text': ''}
    
    text = openrouter_ocr(image_path)
    
    if text in ["API_CONNECTION_ERROR", "ERROR"] or not text:
        return result
        
    result['text'] = text
    
    chunks = split_text_into_chunks(text, 200)
    voice_choice = context.user_data.get('voice_choice')

    for i, chunk in enumerate(chunks):
        try:
            audio_path = os.path.join(TMP_DIR, f"{uuid4().hex}_audio_{i}.mp3")
            
            if not await text_to_speech_anime(chunk, audio_path, gender, voice_choice):
                continue

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                continue
                
            result['audios'].append(audio_path)
            
            video_path = os.path.join(TMP_DIR, f"{uuid4().hex}_karaoke_{i}.mp4")
            
            if create_karaoke_video_with_audio(chunk, audio_path, video_path):
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    result['videos'].append(video_path)

        except Exception as e:
            print(f"Part {i} error: {e}")
            continue
    
    return result


async def handle_photo(update: Update, context):
    try:
        user_id = update.effective_user.id
        
        if 'processing_photos' not in context.user_data:
            context.user_data['processing_photos'] = []
        
        file = await update.message.photo[-1].get_file()
        image_id = uuid4().hex
        image_path = os.path.join(TMP_DIR, f"{image_id}.jpg")
        await file.download_to_drive(image_path)
        
        context.user_data['processing_photos'].append(image_path)
        
        count = len(context.user_data['processing_photos'])
        if count == 1:
            await update.message.reply_text("Photo added. Send more photos or /process to start.")
        else:
            await update.message.reply_text(f"{count}/5 photos. Send more or /process to start.")
        
        if count >= 5:
            await process_photos(update, context)
            
    except Exception as e:
        print(f"Error handling photo: {e}")
        await update.message.reply_text("Error processing photo.")


async def process_photos(update: Update, context):
    try:
        if 'processing_photos' not in context.user_data or not context.user_data['processing_photos']:
            await update.message.reply_text("No photos to process.")
            return
        
        photos = context.user_data['processing_photos']
        
     
        processing_msg = await update.message.reply_text("🔄 Processing your photos... Please wait.")
        
        all_videos = []
        all_audios = []
        
        for i, photo_path in enumerate(photos):
            try:
                if not os.path.exists(photo_path):
                    continue
                    
                gender = detect_gender(photo_path)
                result = await handle_single_photo(update, context, photo_path, gender)
                
                all_videos.extend([v for v in result['videos'] if os.path.exists(v) and os.path.getsize(v) > 0])
                all_audios.extend([a for a in result['audios'] if os.path.exists(a)])
                
                if os.path.exists(photo_path):
                    os.remove(photo_path)
                    
            except Exception as e:
                print(f"Error processing photo {i+1}: {e}")
                continue
        
        context.user_data['processing_photos'] = []
        
        if all_videos:
            await processing_msg.edit_text("Merging videos...")
            merged_video_path = os.path.join(TMP_DIR, f"{uuid4().hex}_merged_karaoke.mp4")
            
            if merge_videos(all_videos, merged_video_path):
                try:
                    if os.path.exists(merged_video_path) and os.path.getsize(merged_video_path) > 1024:
                        with open(merged_video_path, 'rb') as f:
                            await update.message.reply_video(
                                video=f,
                                caption="Your karaoke video is ready! 🎬",
                                supports_streaming=True,
                                read_timeout=60,
                                write_timeout=60,
                                connect_timeout=60
                            )
                except Exception as e:
                    print(f"Error sending video: {e}")
                    await update.message.reply_text("Error sending video.")
                
                if os.path.exists(merged_video_path):
                    os.remove(merged_video_path)
        else:
            await update.message.reply_text("No videos were created. Please try with different photos.")

       
        for file_path in all_videos + all_audios:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
                
        await processing_msg.delete()
        await update.message.reply_text("✅ Processing completed!")
        
    except Exception as e:
        print(f"Error processing photos: {e}")
        await update.message.reply_text("Error processing photos.")
        if 'processing_photos' in context.user_data:
            for photo_path in context.user_data['processing_photos']:
                if os.path.exists(photo_path):
                    os.remove(photo_path)
            context.user_data['processing_photos'] = []


async def clear_photos(update: Update, context):
    """Очистка очереди фото"""
    if 'processing_photos' in context.user_data:
        for photo_path in context.user_data['processing_photos']:
            if os.path.exists(photo_path):
                os.remove(photo_path)
        
        count = len(context.user_data['processing_photos'])
        context.user_data['processing_photos'] = []
        await update.message.reply_text(f"Cleared {count} photos from queue.")
    else:
        await update.message.reply_text("No photos in queue.")


async def show_queue(update: Update, context):
    """Показать текущую очередь фото"""
    if 'processing_photos' in context.user_data and context.user_data['processing_photos']:
        count = len(context.user_data['processing_photos'])
        await update.message.reply_text(f"You have {count} photo(s) in queue. Send more or /process to start.")
    else:
        await update.message.reply_text("No photos in queue.")


async def show_voices(update: Update, context):
    """Показать доступные голоса"""
    current_voice = context.user_data.get('voice_choice', 'Auto detection')
    voices_text = (
        "Available voices:\n\n"
        "Russian voices:\n"
        "/voice_ru_male - Dmitry (male)\n"
        "/voice_ru_female - Svetlana (female)\n\n"
        "English voices:\n"
        "/voice_en_male - Christopher (male)\n"
        "/voice_en_female - Jenny (female)\n\n"
        "Auto detection:\n"
        "/voice_auto - automatic selection\n\n"
        f"Current voice: {current_voice}"
    )
    await update.message.reply_text(voices_text)

async def set_voice_ru_male(update: Update, context):
    """Установка русского мужского голоса"""
    context.user_data['voice_choice'] = "ru-RU-DmitryNeural"
    await update.message.reply_text("Russian male voice selected: Dmitry")

async def set_voice_ru_female(update: Update, context):
    """Установка русского женского голоса"""
    context.user_data['voice_choice'] = "ru-RU-SvetlanaNeural"
    await update.message.reply_text("Russian female voice selected: Svetlana")

async def set_voice_en_male(update: Update, context):
    """Установка английского мужского голоса"""
    context.user_data['voice_choice'] = "en-US-ChristopherNeural"
    await update.message.reply_text("English male voice selected: Christopher")

async def set_voice_en_female(update: Update, context):
    """Установка английского женского голоса"""
    context.user_data['voice_choice'] = "en-US-JennyNeural"
    await update.message.reply_text("English female voice selected: Jenny")

async def set_voice_auto(update: Update, context):
    """Автоматический выбор голоса"""
    if 'voice_choice' in context.user_data:
        del context.user_data['voice_choice']
    await update.message.reply_text("Auto voice detection enabled")


async def start(update: Update, context):
    user_id = update.effective_user.id
    save_user(user_id)

    welcome_text = (
        "Anime Озвучка Bot\n\n"
        "Send me photos with text → I'll create karaoke videos! 🎬\n\n"
        "How to use:\n"
        "1. Choose voice (/voices) - optional\n"
        "2. Send photo(s) with text\n"
        "3. Type /process to start\n"
        "4. Get your video!\n\n"
        "Commands:\n"
        "/voices - choose voice\n"
        "/process - start processing\n"
        "/clear - clear queue\n"
        "/queue - show queue\n\n"
        "Max 5 photos at once 📸"
    )
    await update.message.reply_text(welcome_text)

def main():
    TOKEN = "8481087114:AAEG1TPHxyD7GWqzTdrsHRy0WuJSr-ekzhU"
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("process", process_photos))
    app.add_handler(CommandHandler("clear", clear_photos))
    app.add_handler(CommandHandler("queue", show_queue))
    app.add_handler(CommandHandler("voices", show_voices))
    app.add_handler(CommandHandler("voice_ru_male", set_voice_ru_male))
    app.add_handler(CommandHandler("voice_ru_female", set_voice_ru_female))
    app.add_handler(CommandHandler("voice_en_male", set_voice_en_male))
    app.add_handler(CommandHandler("voice_en_female", set_voice_en_female))
    app.add_handler(CommandHandler("voice_auto", set_voice_auto))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Anime Karaoke Bot is running...")
    app.run_polling()

if __name__ == '__main__':
    main()