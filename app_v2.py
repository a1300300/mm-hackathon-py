import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment


def split_mp3(input_path: str, dest_dir: str, minutes: int) -> list[str]:
    """將 MP3 依指定分鐘數切割，回傳輸出的檔案路徑。"""
    if minutes <= 0:
        raise ValueError("切割分鐘數必須大於 0")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"MP3檔案不存在: {input_path}")

    os.makedirs(dest_dir, exist_ok=True)

    audio = AudioSegment.from_mp3(input_path)
    chunk_length_ms = minutes * 60 * 1000
    input_name = os.path.splitext(os.path.basename(input_path))[0]
    dest_paths = []

    for index, start in enumerate(range(0, len(audio), chunk_length_ms), start=1):
        end = min(start + chunk_length_ms, len(audio))
        dest_path = os.path.join(
            dest_dir,
            f"{input_name}_part_{index:03d}.mp3",
        )

        if os.path.isfile(dest_path):
            print(f"已存在，跳過切割: {dest_path}")
            dest_paths.append(dest_path)
            continue

        chunk = audio[start:end]
        chunk.export(dest_path, format="mp3")
        dest_paths.append(dest_path)
        print(f"已輸出: {dest_path} ({(end - start) / 1000:.1f} 秒)")

    return dest_paths


def transcribe_mp3_to_srt(
    client: OpenAI,
    mp3_path: str,
    srt_path: str,
    model: str = "whisper-1",
) -> None:
    """將單一 MP3 送至 OpenAI，並將字幕直接儲存為 SRT。"""
    if model != "whisper-1":
        raise ValueError(
            f"模型 {model!r} 不支援直接輸出 SRT；請使用 whisper-1。"
        )

    print(f"開始轉錄: {mp3_path}")

    with open(mp3_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language="zh",
            response_format="srt",
        )

    # response_format=srt 會直接回傳 SRT 字串；保留 text 讀取方式以相容
    # 不同版本的 OpenAI Python SDK。
    if isinstance(transcription, str):
        srt_content = transcription
    else:
        srt_content = getattr(transcription, "text", None)
        if not srt_content:
            raise TypeError("OpenAI 回傳內容不是可寫入 SRT 的文字")

    with open(srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(srt_content)

    print(f"已輸出字幕: {srt_path}")


def transcribe_missing_srt(
    client: OpenAI,
    mp3_paths: list[str],
    model: str = "whisper-1",
) -> int:
    """只轉錄尚未產生 SRT 的 MP3，回傳本次新增的字幕數量。"""
    done_count = 0

    for mp3_path in mp3_paths:
        srt_path = os.path.splitext(mp3_path)[0] + '.srt'

        if os.path.isfile(srt_path):
            print(f"已存在，跳過轉錄: {srt_path}")
            continue

        transcribe_mp3_to_srt(
            client=client,
            mp3_path=mp3_path,
            srt_path=srt_path,
            model=model,
        )
        done_count += 1

    return done_count


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


LUNA_MODEL = 'gpt-5.6-luna'
LUNA_SYSTEM_INSTRUCTION = (
    '你是一位總體經濟研究員，請根據使用者提供的規則修飾繁體中文字幕。'
)
LUNA_PROMPT = (
    '請逐行檢查下面的繁體中文字幕，並遵守以下規則：\n\n'
    '1. 最重要：不要合併字幕，也不要把多個時間軸改成單一時間軸；每一行字幕最多不要超過 5 秒。\n'
    '2. 絕對不要修改任何字幕編號、時間軸、字幕筆數或字幕區塊的順序。\n'
    '3. 只修改字幕文字；如有標點符號請移除。\n'
    '4. 公司名稱請使用「財經M平方」。\n'
    '5. 常出現的英文名字為：Rachel、Roger、Ryan、Vivianna、Dylan、Jat、Jason、Danny、Ralice。\n'
    '6. 內容是總體經濟、財經、股市、原物料、債券、央行政策、商品與指數走勢。\n'
    '7. 移除贅字，例如「還有」、「然後」、「嗯嗯」等，但不要改變原意。\n'
    '8. 結尾配樂等沒有語音的地方不需要新增字幕。\n'
    '9. 請輸出完整的 SRT 內容，不要加 Markdown code fence 或其他說明文字。\n'
)
SRT_TIMESTAMP_PATTERN = re.compile(
    r'^\d{2}:\d{2}:\d{2},\d{3} --> '
    r'\d{2}:\d{2}:\d{2},\d{3}$',
    re.MULTILINE,
)
SRT_TIMESTAMP_TOKEN_PATTERN = re.compile(
    r'(\d{2}:\d{2}:\d{2},\d{3}) --> '
    r'(\d{2}:\d{2}:\d{2},\d{3})',
)


def refine_srt_with_luna(client: OpenAI, srt_content: str) -> str:
    """使用 GPT-5.6 Luna 修飾 SRT，但保留原始時間軸與字幕結構。"""
    response = client.responses.create(
        model=LUNA_MODEL,
        instructions=LUNA_SYSTEM_INSTRUCTION,
        input=f'{LUNA_PROMPT}\n--- SRT 開始 ---\n{srt_content}\n--- SRT 結束 ---',
    )

    refined_content = response.output_text.strip()
    if refined_content.startswith('```') and refined_content.endswith('```'):
        refined_content = re.sub(
            r'^```(?:srt)?\s*|\s*```$',
            '',
            refined_content,
            flags=re.IGNORECASE,
        ).strip()

    original_timestamps = SRT_TIMESTAMP_PATTERN.findall(srt_content)
    refined_timestamps = SRT_TIMESTAMP_PATTERN.findall(refined_content)
    if original_timestamps != refined_timestamps:
        raise ValueError('GPT-5.6 Luna 修改了字幕時間軸，拒絕寫入結果')

    return refined_content


def refine_missing_srt_with_luna(
    client: OpenAI,
    srt_paths: list[str],
) -> int:
    """只修飾尚未產生 _refined.srt 的字幕檔。"""
    refined_count = 0

    for srt_path in srt_paths:
        refined_path = os.path.splitext(srt_path)[0] + '_refined.srt'

        if os.path.isfile(refined_path):
            print(f'已存在，跳過 GPT-5.6 Luna 修飾: {refined_path}')
            continue

        if not os.path.isfile(srt_path):
            raise FileNotFoundError(f'找不到字幕檔，無法修飾: {srt_path}')

        with open(srt_path, 'r', encoding='utf-8') as srt_file:
            srt_content = srt_file.read()

        print(f'開始使用 GPT-5.6 Luna 修飾: {srt_path}')
        refined_content = refine_srt_with_luna(client, srt_content)

        with open(refined_path, 'w', encoding='utf-8') as refined_file:
            refined_file.write(refined_content + '\n')

        refined_count += 1
        print(f'已輸出修飾字幕: {refined_path}')

    return refined_count


def _srt_timestamp_to_ms(timestamp: str) -> int:
    hours, minutes, seconds_ms = timestamp.split(':', 2)
    seconds, milliseconds = seconds_ms.split(',')
    return (
        int(hours) * 60 * 60 * 1000
        + int(minutes) * 60 * 1000
        + int(seconds) * 1000
        + int(milliseconds)
    )


def _ms_to_srt_timestamp(total_ms: int) -> str:
    hours, remainder = divmod(total_ms, 60 * 60 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def _shift_srt_timestamps(srt_block: str, offset_ms: int) -> str:
    def shift_timestamp(match: re.Match[str]) -> str:
        start_ms = _srt_timestamp_to_ms(match.group(1)) + offset_ms
        end_ms = _srt_timestamp_to_ms(match.group(2)) + offset_ms
        return (
            f'{_ms_to_srt_timestamp(start_ms)} --> '
            f'{_ms_to_srt_timestamp(end_ms)}'
        )

    return SRT_TIMESTAMP_TOKEN_PATTERN.sub(shift_timestamp, srt_block)


def merge_refined_srts(mp3_paths: list[str], output_path: str) -> int:
    """合併所有 refined SRT，調整時間軸並重新編排字幕編號。"""
    merged_blocks = []
    subtitle_index = 1
    offset_ms = 0

    for mp3_path in mp3_paths:
        refined_path = os.path.splitext(mp3_path)[0] + '_refined.srt'
        if not os.path.isfile(refined_path):
            raise FileNotFoundError(f'找不到修飾後字幕檔: {refined_path}')

        with open(refined_path, 'r', encoding='utf-8') as refined_file:
            srt_content = refined_file.read().replace('\r\n', '\n').strip()

        if not srt_content:
            raise ValueError(f'修飾後字幕檔是空的: {refined_path}')

        blocks = re.split(r'\n\s*\n', srt_content)
        for block in blocks:
            lines = block.splitlines()
            # 空白字幕區塊可能只有編號與時間軸，仍需保留其時間軸。
            if len(lines) < 2 or not SRT_TIMESTAMP_PATTERN.fullmatch(lines[1]):
                raise ValueError(f'SRT 格式錯誤: {refined_path}')

            lines[0] = str(subtitle_index)
            lines[1] = _shift_srt_timestamps(lines[1], offset_ms)
            merged_blocks.append('\n'.join(lines))
            subtitle_index += 1

        offset_ms += len(AudioSegment.from_mp3(mp3_path))

    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write('\n\n'.join(merged_blocks) + '\n')

    return subtitle_index - 1


if __name__ == '__main__':
    load_dotenv()

    # 更改為要切割的 MP3 檔案名稱
    source_mp3 = '0827_Podcast.mp3'
    source_path = os.path.join('./input_files', source_mp3)

    # 每幾分鐘切割一段，可直接修改這裡
    mins = 15
    output_dir = './output_files'

    print(f'開始分割 MP3，每 {mins} 分鐘一段')
    output_paths = split_mp3(source_path, output_dir, mins)
    print(f'完成，共輸出 {len(output_paths)} 個檔案')

    client = None
    pending_srt_paths = [
        mp3_path
        for mp3_path in output_paths
        if not os.path.isfile(os.path.splitext(mp3_path)[0] + '.srt')
    ]

    if not pending_srt_paths:
        print('所有 SRT 字幕檔都已存在，跳過轉錄')
    else:
        if not os.getenv('OPENAI_API_KEY'):
            raise RuntimeError(
                '找不到 OPENAI_API_KEY，請在 .env 或環境變數中設定'
            )

        client = OpenAI()
        transcribed_count = transcribe_missing_srt(
            client=client,
            mp3_paths=output_paths,
            model='whisper-1',
        )
        print(f'完成，共新增 {transcribed_count} 個 SRT 字幕檔')

    # 對每一個字幕檔做 apply_error_dictionary2 的初步字詞替換
    print('開始套用錯誤字典')
    for mp3_path in output_paths:
        srt_path = os.path.splitext(mp3_path)[0] + '.srt'

        if not os.path.isfile(srt_path):
            raise FileNotFoundError(f'找不到字幕檔，無法套用錯誤字典: {srt_path}')

        with open(srt_path, 'r', encoding='utf-8') as srt_file:
            srt_content = srt_file.read()

        corrected_srt = apply_error_dictionary2(srt_content)

        with open(srt_path, 'w', encoding='utf-8') as srt_file:
            srt_file.write(corrected_srt)

        print(f'已套用錯誤字典: {srt_path}')

    # 對每一個字幕檔使用 GPT-5.6 Luna 修飾，結果另存為 _refined.srt。
    srt_paths = [
        os.path.splitext(mp3_path)[0] + '.srt'
        for mp3_path in output_paths
    ]
    pending_refined_paths = [
        srt_path
        for srt_path in srt_paths
        if not os.path.isfile(os.path.splitext(srt_path)[0] + '_refined.srt')
    ]

    if not pending_refined_paths:
        print('所有 GPT-5.6 Luna 修飾字幕檔都已存在，跳過修飾')
    else:
        if client is None:
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError(
                    '找不到 OPENAI_API_KEY，請在 .env 或環境變數中設定'
                )
            client = OpenAI()

        refined_count = refine_missing_srt_with_luna(
            client=client,
            srt_paths=srt_paths,
        )
        print(f'完成，共新增 {refined_count} 個 GPT-5.6 Luna 修飾字幕檔')

    # 最後結合成一個完整的 final 字幕檔。
    final_srt_path = os.path.join(
        output_dir,
        os.path.splitext(source_mp3)[0] + '_final.srt',
    )

    if os.path.isfile(final_srt_path):
        print(f'完整字幕檔已存在，跳過合併: {final_srt_path}')
    else:
        print('開始合併完整字幕檔')
        merged_count = merge_refined_srts(output_paths, final_srt_path)
        print(f'已輸出完整字幕檔: {final_srt_path}，共 {merged_count} 段字幕')
