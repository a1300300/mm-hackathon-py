from app import refine_srt_with_gemini

input_file = './output_files/1009_Podcast.srt'

with open(input_file, 'r', encoding='utf-8') as f:
    final_srt = f.read()

# 再交給 Gemini 清理贅字與公司名校正
try:
    print('再交給 Gemini 清理贅字與公司名校正')
    final_srt = refine_srt_with_gemini(final_srt)
except Exception as e:
    # 若 Gemini 呼叫失敗，保留本地修正版，並輸出警告
    print(f"[Warn] Gemini refine failed: {e}")

# Write a final SRT file
tmp_file = './tmp/test1_output.srt'
with open(tmp_file, '+w', encoding='utf-8') as f:
    f.write(final_srt)