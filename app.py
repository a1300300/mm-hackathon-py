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


def apply_error_dictionary2(text: str) -> str:
    if not os.path.isfile('./error_dict.txt'):
        raise RuntimeError("找無錯誤字典(error_dict.txt)")

    with open('./error_dict.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            k, v = line.split('=>')

            text = text.replace(k, v)

    return text

def refine_srt_with_gemini(srt_text: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "上面內容是繁體中文字幕，請遵守以下十點規則來修改：\n\n"
        "1. 第一點最重要: 拜託不要自行合併多行字幕句子變成一段很長的字幕;換句換說，不要擅自合併多個時間軸成單一時間軸，因為這樣一行字幕會被拉得很長。一行字幕最多不要超過5秒\n"
        "2. 第二點也很重要: 拜託不要更改每一行字幕的時間軸秒數，要跟原本的來源秒數一樣。\n"
        "3. 一行一行的檢查並視情況做修改，如果有標點符號請拿掉。\n"
        "4. 我們公司名稱是財經M平方，請判斷是否產生對的公司名稱。\n"
        "5. 常出現的英文名字名單為: Rachel, Roger, Ryan, Vivianna, Dylan, Jat, Jason, Danny, Ralice"
        "6. 然後字幕的內容是關於總體經濟的話題，因此會提到很多經濟、財經、股市、原物料、債券等等相關名詞。\n"
        "7. 並且也包含各國央行鷹鴿派走向、商品以及指數的走勢、行情等等的分析。\n"
        "8. 除此之外希望可以移除贅字如還有、然後、嗯嗯等等的。\n"
        "9. 輸出的字幕檔的格式不要跑掉，例如原本句子之間的空行不要自行拿掉。\n"
        "10. 結尾配樂的地方就不需要自行上字幕了"
    )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            system_instruction="你是一位總體經濟研究員，並且將根據我提供的影片字幕內容進行審查並修飾。"
        ),
        contents=[
            genai.types.Part.from_bytes(
                data=bytes(srt_text, 'utf-8'),
                mime_type='text/plain',
            ),
            prompt
        ])

    return response.text


if __name__ == '__main__':
    ### 更改為要轉檔的mp3檔案名稱
    input_mp3 = 'chunk_7.mp3'
    input_path = './input_files/' + input_mp3

    if not os.path.isfile(input_path):
        raise RuntimeError("MP3檔案不存在")

    # 分割mp3檔，設定每分鐘做分割
    print('分割mp3檔案')
    mins = 5
    chunk_length = 60 * 1000 * mins # 分鐘

    audio = AudioSegment.from_mp3(input_path)
    audio_slices, start_secs = split_audio(source=audio, length=chunk_length)

    raw_srt_contents = []
    srt_contents = []

    # 先產生字幕再結合
    for idx, audio_slice in enumerate(audio_slices):
        print(f"Processing chunk {idx + 1}/{len(audio_slices)}")

        file_path = f"./tmp/chunk_{idx + 1}.mp3"
        audio_slice.export(file_path, format="mp3")

        print('OpenAI 產生字幕檔')
        srt_content = transcribe_audio(file_path)

        # 備存個別srt檔案到tmp
        print('備存個別srt檔案到tmp')
        with open(f"./tmp/chunk_{idx + 1}.srt", 'w', encoding='utf-8') as f:
            f.write(srt_content)

             # 先做本地名詞/錯字替換
            print('先做本地名詞/錯字替換')
            srt_content = apply_error_dictionary2(srt_content)

        raw_srt_contents.append(srt_content)

        # 再交給 Gemini 校正文字
        is_pass = False
        while not is_pass:
            try:
                print('再交給 Gemini 校正文字')
                srt_content = refine_srt_with_gemini(srt_content)

                # 備存校正後的字幕檔至tmp
                print('備存結合的字幕檔至tmp')
                fined_srt_filename = f"./tmp/fined_{idx + 1}.srt"
                with open(fined_srt_filename, '+w', encoding='utf-8') as f:
                    f.write(srt_content)

                is_pass = True
            except Exception as e:
                # 若 Gemini 呼叫失敗，保留本地修正版，並輸出警告
                print(f"[Warn] Gemini refine failed: {e}, will redo")

        srt_contents.append(srt_content)

    # Merge SRT contents
    print('結合未修飾字幕檔')
    raw_srt_filename = './output_files/' + os.path.splitext(input_mp3)[0] + "_before.srt"
    if len(raw_srt_contents) == 1:
        # Single file, no need to merge
        raw_final_srt = raw_srt_contents[0]
    else:
        # Multiple chunks, merge with timestamp adjustment
        raw_final_srt = merge_srt_files(raw_srt_contents, start_secs)

    # Write a final SRT file
    with open(raw_srt_filename, '+w', encoding='utf-8') as f:
        f.write(raw_final_srt)

    print('結合已修飾字幕檔')
    srt_filename = './output_files/' + os.path.splitext(input_mp3)[0] + "_after.srt"
    if len(srt_contents) == 1:
        # Single file, no need to merge
        final_srt = srt_contents[0]
    else:
        # Multiple chunks, merge with timestamp adjustment
        final_srt = merge_srt_files(srt_contents, start_secs)

    # Write a final SRT file
    with open(srt_filename, '+w', encoding='utf-8') as f:
        f.write(final_srt)

    print(f'完成輸出, 未修飾字幕路徑: {raw_srt_filename}, 已修飾字幕路徑: {srt_filename}')
