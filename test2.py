from app import apply_error_dictionary2

input_file = './output_files/1009_Podcast.srt'

with open(input_file, 'r', encoding='utf-8') as f:
    final_srt = f.read()

test = apply_error_dictionary2(final_srt)
print(test)