from dotenv import load_dotenv, find_dotenv
from error_dict import error_words
from pydub import AudioSegment
import re, os, openai

# Load variables from the nearest .env file (walking up directories if needed)
load_dotenv(find_dotenv(), override=False)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found. Make sure it's set in your .env file.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Make sure it's set in your .env file.")


def split_audio(source: AudioSegment, length: int):
    """
    Split an AudioSegment into chunks.

    Parameters:
        source (AudioSegment): the loaded audio file.
        length (int): chunk length in milliseconds (e.g., 60_000 for 1 minute).

    Returns:
        list[AudioSegment]: list of audio chunks.
    """
    start_times = []
    chunks = []
    for i in range(0, len(source), length):
        chunks.append(source[i:i + length])
        start_times.append(i / 1000)
    return chunks, start_times


def transcribe_audio(f_path):
    """Transcribe audio using OpenAI Whisper"""
    client = openai.OpenAI()

    with open(f_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="zh",
            response_format="srt"
        )

    return transcript


def adjust_srt_timestamps(srt_content, offset_seconds):
    """Adjust SRT timestamps by adding offset"""
    if offset_seconds == 0:
        return srt_content

    def adjust_timestamp(match):
        timestamp = match.group(0)
        # Parse timestamp: HH:MM:SS,mmm
        time_parts = timestamp.replace(',', ':').split(':')
        hours, minutes, seconds, milliseconds = map(int, time_parts)

        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        total_seconds += offset_seconds

        new_hours = int(total_seconds // 3600)
        new_minutes = int((total_seconds % 3600) // 60)
        new_seconds = int(total_seconds % 60)
        new_milliseconds = int((total_seconds % 1) * 1000)

        return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},{new_milliseconds:03d}"

    # Regex to match timestamp format: HH:MM:SS,mmm
    timestamp_pattern = r'\d{2}:\d{2}:\d{2},\d{3}'
    return re.sub(timestamp_pattern, adjust_timestamp, srt_content)


def merge_srt_files(contents, start_times):
    """Merge multiple SRT contents with proper indexing and timestamps"""
    merged_content = []
    subtitle_index = 1

    for srt_content, start_offset in zip(contents, start_times):
        # Adjust timestamps for this chunk
        adjusted_content = adjust_srt_timestamps(srt_content, start_offset)

        # Split into blocks and reindex
        blocks = adjusted_content.strip().split('\n\n')

        for block in blocks:
            if block.strip():
                lines = block.strip().split('\n')
                if len(lines) >= 3:  # Valid SRT block
                    # Replace the index with our sequential one
                    lines[0] = str(subtitle_index)
                    merged_content.append('\n'.join(lines))
                    subtitle_index += 1

    return '\n\n'.join(merged_content)


def apply_error_dictionary(text: str) -> str:
    if not error_words:
        return text
    for wrong, right in error_words.items():
        # 全文直接替換，避免破壞時間戳，僅處理字幕文字行
        text = text.replace(wrong, right)
    return text


def refine_srt_with_gemini(srt_text: str) -> str:
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "音訊內容主要為台灣口音的中文。\n"
        "請直接輸出 SRT 內容，不要包含任何額外的說明文字或代碼塊標記。\n"
        "我們公司名稱是財經M平方，請判斷是否產生對的公司名稱。\n"
        "然後音檔是關於總體經濟的話題，因此會提到很多經濟、財經等名詞。\n"
        "並且也包含各國央行鷹鴿派走向、商品以及指數的走勢、行情等等的分析。\n"
        "除此之外希望可以移除贅字如還有、然後、嗯嗯等等的。\n"
        "另外重要的是，盡量不要摻雜簡體中文的用詞。\n"
        "以下是原始 SRT，請在保留原時間戳的前提下，針對文字做整理與修正：\n\n"
        f"{srt_text}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai.types.Part.from_bytes(
                data=bytes(srt_text, 'utf-8'),
                mime_type='text/plain',
            ),
            prompt])

    return response.text

### 更改為要轉檔的mp3檔案名稱
input_mp3 = 'my_CH_002.mp3'
input_path = './input_files/' + input_mp3

if not os.path.isfile(input_path):
    raise RuntimeError("MP3檔案不存在")

# 分割mp3檔，設定每分鐘做分割
mins = 20
chunk_length = 60 * 1000 * mins # 分鐘

audio = AudioSegment.from_mp3(input_path)
audio_slices, start_secs = split_audio(source=audio, length=chunk_length)

srt_contents = []
# 先產生字幕再結合
for idx, audio_slice in enumerate(audio_slices):
    print(f"Processing chunk {idx + 1}/{len(audio_slices)}")

    file_path = f"./tmp/chunk_{idx + 1}.mp3"
    audio_slice.export(file_path, format="mp3")

    srt_contents.append(transcribe_audio(file_path))

# Merge SRT contents
print('結合字幕檔')
srt_filename = './output_files/' + os.path.splitext(input_mp3)[0] + ".srt"
if len(srt_contents) == 1:
    # Single file, no need to merge
    final_srt = srt_contents[0]
else:
    # Multiple chunks, merge with timestamp adjustment
    final_srt = merge_srt_files(srt_contents, start_secs)

# 先做本地名詞/錯字替換
print('先做本地名詞/錯字替換')
final_srt = apply_error_dictionary(final_srt)
try:
    print('再交給 Gemini 清理贅字與公司名校正')
    final_srt = refine_srt_with_gemini(final_srt)
except Exception as e:
    # 若 Gemini 呼叫失敗，保留本地修正版，並輸出警告
    print(f"[Warn] Gemini refine failed: {e}")


# Write a final SRT file
with open(srt_filename, '+w', encoding='utf-8') as f:
    f.write(final_srt)

print('完成輸出, 路徑:'+srt_filename)
