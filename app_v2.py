import asyncio
import json
import os
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
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


async def transcribe_mp3_to_srt(
    client: AsyncOpenAI,
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
        transcription = await client.audio.transcriptions.create(
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


async def transcribe_missing_srt(
    client: AsyncOpenAI,
    mp3_paths: list[str],
    model: str = "whisper-1",
    max_concurrency: int = 3,
) -> int:
    """並行轉錄尚未產生 SRT 的 MP3，回傳本次新增的字幕數量。"""
    if max_concurrency <= 0:
        raise ValueError("max_concurrency 必須大於 0")

    pending_paths = []
    for mp3_path in mp3_paths:
        srt_path = os.path.splitext(mp3_path)[0] + '.srt'

        if os.path.isfile(srt_path):
            print(f"已存在，跳過轉錄: {srt_path}")
            continue

        pending_paths.append((mp3_path, srt_path))

    semaphore = asyncio.Semaphore(max_concurrency)

    async def transcribe_one(mp3_path: str, srt_path: str) -> None:
        async with semaphore:
            await transcribe_mp3_to_srt(
                client=client,
                mp3_path=mp3_path,
                srt_path=srt_path,
                model=model,
            )

    await asyncio.gather(
        *(transcribe_one(mp3_path, srt_path)
          for mp3_path, srt_path in pending_paths)
    )

    return len(pending_paths)


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
LUNA_SYSTEM_INSTRUCTION = """
你是財經M平方的專業繁體中文字幕校對編輯。

你的任務是校正語音辨識產生的繁體中文字幕，只能進行保守、必要且有根據的文字修正。
你不是摘要員、翻譯員，也不是內容改寫員。
不要根據外部知識擴寫、補充或改變原本的經濟觀點。
"""
LUNA_PROMPT = """
請校正輸入 JSON 中的每一筆繁體中文字幕。

# 主要目標

只修正以下內容：

1. 明顯的語音辨識錯字。
2. 明顯的人名、公司名、財經術語錯誤。
3. 沒有語意的口語贅字。
4. 不自然但不改變原意的文字表達。

如果原文已經合理，請原樣保留。
如果無法確定正確寫法，請保留原文，不要猜測。

# 結構與完整性規則

1. 每筆字幕的 position 是不可變更的唯一識別碼，必須原樣輸出。
2. 每一筆輸入必須恰好對應一筆輸出。
3. 不得新增、刪除、合併或拆分字幕。
4. 不得調換字幕順序。
5. 不得把某一筆字幕的文字移到另一筆字幕。
6. 不得遺漏任何字幕，即使該筆字幕內容是空白，也必須保留為空白。
7. 若原文內有換行，應盡量維持原本的換行結構。
8. 只能修改 text 欄位，不得修改 position。
9. 不要在 text 中加入 position、時間軸、JSON 標記或其他說明文字。

# 絕對不可任意修改的內容

以下內容除非明顯是語音辨識錯誤，否則必須原樣保留：

1. 數字、百分比、金額、日期與年份。
2. 正負號、倍數、單位與數值關係。
3. 股票代號、指數名稱、商品名稱與債券名稱。
4. 央行名稱、國家名稱與地名。
5. 否定語氣，例如「不」、「沒有」、「未」、「避免」。
6. 因果、轉折與條件關係。
7. 原文的經濟觀點、判斷與結論。
8. 英文縮寫、品牌名稱與產品名稱。
9. 已經正確的專有名詞。

不得因為句子不夠完整，就自行補充原文沒有說出的資訊。

# 公司、人名與專有名詞

1. 公司名稱應使用「財經M平方」。
2. 常出現的人名包括：
   Rachel、Roger、Ryan、Vivianna、Dylan、Jat、Jason、Danny、Ralice。
3. 人名清單只能作為參考，不可強制套用。
4. 只有在語音、上下文或文字內容足以確認時，才修正人名。
5. 財經內容常見詞彙包括：
   總體經濟、通膨、通貨膨脹、利率、降息、升息、央行、
   鷹派、鴿派、債券、殖利率、信用利差、股市、股價、
   原物料、商品、能源、黃金、石油、美元、匯率、
   內需、出口、進口、關稅、供應鏈、製造業、服務業、
   非農、失業率、GDP、CPI、PPI、PMI、AI、半導體與指數。

不要因為名詞看起來相似，就任意替換成其他財經名詞。

# 贅字與口語修正

1. 只有在「嗯」、「嗯嗯」、「喔」、「那個」等詞沒有實際語意時，才移除。
2. 「然後」、「還有」、「所以」有時具有語意，不可以一律刪除。
3. 刪除贅字後，必須維持原句意思與語氣。
4. 不要把自然的口語改成過度正式或書面化的文章。
5. 不要大幅重寫句子。
6. 不要把多個短句改寫成一個長句。

# 標點符號

1. 移除句末與句中的一般標點符號。
2. 不要因此改變句意。
3. 保留數字或英文名稱中必要的符號，例如：
   S&P 500、U.S.、AI-driven、ChatGPT。
4. 不要移除股票代號、產品名稱或英文縮寫內必要的符號。

# 無法確認時的處理方式

如果遇到以下情況，請保留原文：

1. 不確定是哪一個人名。
2. 不確定是哪一個財經名詞。
3. 不確定數字或百分比。
4. 不確定句子的真正意思。
5. 可能需要依賴外部資料才能判斷。
6. 修改後可能改變原本觀點。

# 最終要求

在輸出前，請確認：

1. 輸出字幕筆數與輸入完全相同。
2. 所有 position 都存在，且順序完全相同。
3. 沒有新增、刪除、合併或拆分字幕。
4. 只修改必要的字幕文字。
5. 不輸出任何解釋、分析、Markdown 或額外文字。
6. 只依照 API 提供的 JSON Schema 輸出結果。
"""
SRT_TIMESTAMP_PATTERN = re.compile(
    r'^\d{2}:\d{2}:\d{2},\d{3} --> '
    r'\d{2}:\d{2}:\d{2},\d{3}$',
    re.MULTILINE,
)
SRT_TIMESTAMP_TOKEN_PATTERN = re.compile(
    r'(\d{2}:\d{2}:\d{2},\d{3}) --> '
    r'(\d{2}:\d{2}:\d{2},\d{3})',
)


async def refine_srt_with_luna(
    client: AsyncOpenAI,
    srt_content: str,
) -> str:
    """只讓 Luna 修改字幕文字，再組回原始編號與時間軸。"""
    normalized_content = srt_content.replace('\r\n', '\n').strip()
    blocks = re.split(r'\n\s*\n', normalized_content)
    original_blocks = []
    subtitle_items = []

    for position, block in enumerate(blocks):
        lines = block.splitlines()
        if len(lines) < 2 or not SRT_TIMESTAMP_PATTERN.fullmatch(lines[1]):
            raise ValueError(f'SRT 第 {position + 1} 個區塊格式錯誤')

        original_blocks.append(lines)
        subtitle_items.append({
            'position': position,
            'text': '\n'.join(lines[2:]),
        })

    response = await client.responses.create(
        model=LUNA_MODEL,
        instructions=LUNA_SYSTEM_INSTRUCTION,
        input=f'{LUNA_PROMPT}\n輸入 JSON：\n{json.dumps(subtitle_items, ensure_ascii=False)}',
        text={
            'format': {
                'type': 'json_schema',
                'name': 'refined_subtitles',
                'strict': True,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'subtitles': {
                            'type': 'array',
                            'minItems': len(subtitle_items),
                            'maxItems': len(subtitle_items),
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'position': {'type': 'integer'},
                                    'text': {'type': 'string'},
                                },
                                'required': ['position', 'text'],
                                'additionalProperties': False,
                            },
                        },
                    },
                    'required': ['subtitles'],
                    'additionalProperties': False,
                },
            },
        },
    )

    result = json.loads(response.output_text)
    refined_items = result['subtitles']
    expected_positions = list(range(len(subtitle_items)))
    actual_positions = [item['position'] for item in refined_items]
    if actual_positions != expected_positions:
        raise ValueError('GPT-5.6 Luna 回傳的字幕數量或順序不正確')

    refined_blocks = []
    for original_lines, refined_item in zip(original_blocks, refined_items):
        text_lines = refined_item['text'].splitlines()
        refined_blocks.append('\n'.join(original_lines[:2] + text_lines))

    return '\n\n'.join(refined_blocks)


async def refine_missing_srt_with_luna(
    client: AsyncOpenAI,
    srt_paths: list[str],
    max_concurrency: int = 3,
) -> int:
    """並行修飾尚未產生 _refined.srt 的字幕檔。"""
    if max_concurrency <= 0:
        raise ValueError("max_concurrency 必須大於 0")

    pending_paths = []
    for srt_path in srt_paths:
        refined_path = os.path.splitext(srt_path)[0] + '_refined.srt'

        if os.path.isfile(refined_path):
            print(f'已存在，跳過 GPT-5.6 Luna 修飾: {refined_path}')
            continue

        if not os.path.isfile(srt_path):
            raise FileNotFoundError(f'找不到字幕檔，無法修飾: {srt_path}')

        pending_paths.append((srt_path, refined_path))

    semaphore = asyncio.Semaphore(max_concurrency)

    async def refine_one(srt_path: str, refined_path: str) -> None:
        async with semaphore:
            with open(srt_path, 'r', encoding='utf-8') as srt_file:
                srt_content = srt_file.read()

            print(f'開始使用 GPT-5.6 Luna 修飾: {srt_path}')
            refined_content = await refine_srt_with_luna(client, srt_content)

            with open(refined_path, 'w', encoding='utf-8') as refined_file:
                refined_file.write(refined_content + '\n')

            print(f'已輸出修飾字幕: {refined_path}')

    await asyncio.gather(
        *(refine_one(srt_path, refined_path)
          for srt_path, refined_path in pending_paths)
    )

    return len(pending_paths)


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
    source_mp3 = '2603-MEO.mp3'
    source_path = os.path.join('./input_files', source_mp3)

    # 每幾分鐘切割一段，可直接修改這裡
    mins = 15
    output_dir = './output_files'

    print(f'開始分割 MP3，每 {mins} 分鐘一段')
    output_paths = split_mp3(source_path, output_dir, mins)
    print(f'完成，共輸出 {len(output_paths)} 個檔案')

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

        async def run_transcriptions() -> int:
            async with AsyncOpenAI() as async_client:
                return await transcribe_missing_srt(
                    client=async_client,
                    mp3_paths=output_paths,
                    model='whisper-1',
                    max_concurrency=3,
                )

        transcribed_count = asyncio.run(run_transcriptions())
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
        if not os.getenv('OPENAI_API_KEY'):
            raise RuntimeError(
                '找不到 OPENAI_API_KEY，請在 .env 或環境變數中設定'
            )

        async def run_refinements() -> int:
            async with AsyncOpenAI() as async_client:
                return await refine_missing_srt_with_luna(
                    client=async_client,
                    srt_paths=srt_paths,
                    max_concurrency=3,
                )

        refined_count = asyncio.run(run_refinements())
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
